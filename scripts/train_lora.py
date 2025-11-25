import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import argparse
from dataclasses import dataclass, field
import os
import numpy as np
from training.data_loader import get_dataloaders
from tqdm.auto import tqdm  # 添加tqdm

@dataclass
class LoRAConfig:
    """LoRA微调配置"""
    base_model: str = "/home/yang/gpt2-moe-adapter/gpt2"
    dataset_mode: str = "mixed"  # 'mixed' 或 'single'
    batch_size: int = 4
    max_length: int = 512
    learning_rate: float = 5e-4
    num_epochs: int = 3
    warmup_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    
    # 早停配置
    early_stopping_patience: int = 3  # 容忍的评估次数
    early_stopping_threshold: float = 0.001  # 改善阈值
    
    # LoRA配置
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: tuple = ("c_attn", "c_proj")  # GPT-2的注意力投影层
    
    # 数据集混合配置
    dataset_mix: list = field(
        default_factory=lambda: [
            # 基础指令数据
            ("tatsu-lab/alpaca", None, 0.3, "instruction"),
            ("databricks/databricks-dolly-15k", None, 0.1, "instruction"),
            
            # 学科知识数据  
            ("cais/mmlu", "all", 0.15, "knowledge"),
            ("allenai/ai2_arc", "ARC-Challenge", 0.15, "knowledge"),
            ("derek-thomas/ScienceQA", None, 0.1, "knowledge"),
            
            # 推理数据
            ("gsm8k", "main", 0.1, "reasoning"),
            ("tau/commonsense_qa", None, 0.1, "reasoning"),
            
            # WikiText基础语言建模
            ("wikitext", "wikitext-2-raw-v1", 0.1, "lm")
        ]
    )
    target_total_samples: int = 50000

class EarlyStoppingCallback:
    """早停回调函数"""
    
    def __init__(self, patience=3, min_delta=0.001, save_path="./best_model"):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
        # 确保保存路径存在
        os.makedirs(save_path, exist_ok=True)
        
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """在评估后调用"""
        if metrics is None or 'eval_loss' not in metrics:
            return
        
        current_loss = metrics['eval_loss']
        
        if self.best_loss is None:
            # 第一次评估
            self.best_loss = current_loss
            self.save_checkpoint(args, state, control)
        elif current_loss < self.best_loss - self.min_delta:
            # 有显著改善
            self.best_loss = current_loss
            self.counter = 0
            self.save_checkpoint(args, state, control)
            print(f"🎯 模型改善! 验证损失: {current_loss:.4f} (最佳: {self.best_loss:.4f})")
        else:
            # 没有改善
            self.counter += 1
            print(f"⏳ 早停计数: {self.counter}/{self.patience}, 当前损失: {current_loss:.4f}, 最佳损失: {self.best_loss:.4f}")
            
            if self.counter >= self.patience:
                print("🛑 触发早停机制!")
                self.early_stop = True
                control.should_training_stop = True
    
    def save_checkpoint(self, args, state, control):
        """保存最佳模型检查点"""
        print(f"💾 保存最佳模型检查点 (损失: {self.best_loss:.4f})")
        
    def on_step_end(self, args, state, control, **kwargs):
        """在每个训练步骤结束时检查是否应该停止"""
        if self.early_stop:
            control.should_training_stop = True

def setup_lora_model(config):
    """设置LoRA模型"""
    print("🚀 初始化GPT-2 + LoRA模型...")
    
    # 加载基础模型
    model = GPT2LMHeadModel.from_pretrained(config.base_model)
    
    # 配置LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # 因果语言建模
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数
    model.print_trainable_parameters()
    
    return model

class CustomTrainer(Trainer):
    """自定义Trainer以支持早停"""
    
    def __init__(self, *args, early_stopping_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.early_stopping_callback = early_stopping_callback
    
    def evaluation_loop(self, *args, **kwargs):
        """重写评估循环以集成早停"""
        output = super().evaluation_loop(*args, **kwargs)
        
        # 调用早停回调
        if self.early_stopping_callback:
            self.early_stopping_callback.on_evaluate(
                self.args,
                self.state,
                self.control,
                output.metrics
            )
        
        return output
    
    def training_step(self, *args, **kwargs):
        """重写训练步骤以检查早停"""
        output = super().training_step(*args, **kwargs)
        
        # 检查是否应该停止
        if self.early_stopping_callback:
            self.early_stopping_callback.on_step_end(
                self.args,
                self.state,
                self.control
            )
        
        return output

def train_lora_gpt2(config):
    """训练LoRA微调的GPT-2模型（带早停）"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据 - 注意：这里直接使用数据加载器，不需要额外的collate_fn
    print("📊 加载数据集...")
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(config)
    
    # 设置模型
    model = setup_lora_model(config)
    model = model.to(device)
    
    # 早停回调
    early_stopping = EarlyStoppingCallback(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_threshold,
        save_path="./gpt2-lora-best"
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./gpt2-lora-output",
        overwrite_output_dir=True,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,  # 自动加载最佳模型
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=False,
        report_to=None,
        save_total_limit=2,  # 只保存2个检查点以节省空间
    )
    
    # 关键修改：直接使用数据加载器的数据集，但需要重新包装
    # 因为Trainer期望的是Dataset对象，而不是DataLoader
    
    # 创建自定义数据集包装器
    class DataLoaderDataset(torch.utils.data.Dataset):
        def __init__(self, dataloader):
            self.dataloader = dataloader
            # 预加载所有数据到内存
            self.data = []
            for batch in dataloader:
                for i in range(len(batch['input_ids'])):
                    self.data.append({
                        'input_ids': batch['input_ids'][i],
                        'attention_mask': batch['attention_mask'][i],
                        'labels': batch['labels'][i]
                    })
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    # 创建数据集
    train_dataset = DataLoaderDataset(train_loader)
    val_dataset = DataLoaderDataset(val_loader)
    
    # 创建自定义Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        early_stopping_callback=early_stopping,
    )
    
    # 开始训练
    print("🎯 开始LoRA微调训练...")
    print(f"⏰ 早停机制: 容忍 {config.early_stopping_patience} 次无改善评估")
    
    try:
        trainer.train()
        
        # 检查是否因早停而结束
        if early_stopping.early_stop:
            print("🏁 训练因早停机制而结束")
        else:
            print("🏁 训练正常完成")
            
    except KeyboardInterrupt:
        print("⚠️ 训练被用户中断")
    except Exception as e:
        print(f"❌ 训练出错: {e}")
        raise
    
    # 保存最终模型
    print("💾 保存最终模型...")
    trainer.save_model("./gpt2-lora-final")
    
    # 评估模型
    print("📈 评估模型...")
    eval_results = trainer.evaluate()
    print(f"最终评估结果: {eval_results}")
    
    return model, tokenizer

def train_lora_gpt2_simple(config):
    """简化的训练循环，避免复杂的Dataset包装"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("📊 加载数据集...")
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(config)
    
    # 设置模型
    model = setup_lora_model(config)
    model = model.to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # 训练状态
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0
    
    print("🎯 开始LoRA微调训练...")
    print(f"⏰ 早停机制: 容忍 {config.early_stopping_patience} 次无改善评估")
    
    # 计算总训练步数
    total_train_steps = len(train_loader) * config.num_epochs
    print(f"📊 总训练步数: {total_train_steps}")
    
    # 创建主进度条
    main_pbar = tqdm(total=total_train_steps, desc="总体训练进度", position=0)
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        # 创建epoch进度条
        epoch_pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config.num_epochs}", position=1, leave=False)
        
        for batch_idx, batch in enumerate(train_loader):
            # 移动到设备
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            
            # 前向传播
            outputs = model(**inputs)
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_train_loss += loss.item()
            train_steps += 1
            global_step += 1
            
            # 更新进度条
            current_loss = total_train_loss / train_steps
            epoch_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
            epoch_pbar.update(1)
            main_pbar.update(1)
            
            # 记录日志
            if global_step % config.logging_steps == 0:
                avg_loss = total_train_loss / train_steps
                print(f"\n📝 Step {global_step}, Loss: {avg_loss:.4f}")
                total_train_loss = 0
                train_steps = 0
        
        epoch_pbar.close()
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        val_steps = 0
        
        print("🔍 验证中...")
        # 创建验证进度条
        val_pbar = tqdm(total=len(val_loader), desc="验证进度", position=1, leave=False)
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'labels': batch['labels'].to(device)
                }
                
                outputs = model(**inputs)
                total_val_loss += outputs.loss.item()
                val_steps += 1
                
                val_pbar.update(1)
        
        val_pbar.close()
        
        avg_val_loss = total_val_loss / val_steps
        print(f"📊 Epoch {epoch+1}, 验证损失: {avg_val_loss:.4f}")
        
        # 早停检查
        if avg_val_loss < best_val_loss - config.early_stopping_threshold:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            model.save_pretrained("./gpt2-lora-best")
            tokenizer.save_pretrained("./gpt2-lora-best")
            print(f"🎯 保存最佳模型，验证损失: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ 早停计数: {patience_counter}/{config.early_stopping_patience}")
            
            if patience_counter >= config.early_stopping_patience:
                print("🛑 触发早停机制!")
                break
    
    # 关闭主进度条
    main_pbar.close()
    
    # 加载最佳模型
    try:
        model = GPT2LMHeadModel.from_pretrained("./gpt2-lora-best")
        model = get_peft_model(model, LoraConfig.from_pretrained("./gpt2-lora-best"))
        model = model.to(device)
        print("💾 加载最佳模型完成")
    except:
        print("⚠️ 无法加载最佳模型，使用当前模型")
    
    # 保存最终模型
    model.save_pretrained("./gpt2-lora-final")
    tokenizer.save_pretrained("./gpt2-lora-final")
    print("💾 保存最终模型完成")
    
    return model, tokenizer

def train_lora_gpt2_with_tqdm(config):
    """使用tqdm的增强版训练循环"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    print("📊 加载数据集...")
    train_loader, val_loader, test_loader, tokenizer = get_dataloaders(config)
    
    # 设置模型
    model = setup_lora_model(config)
    model = model.to(device)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * config.num_epochs)
    
    # 训练状态
    best_val_loss = float('inf')
    patience_counter = 0
    global_step = 0
    
    # 训练统计
    train_losses = []
    val_losses = []
    
    print("🎯 开始LoRA微调训练...")
    print(f"⏰ 早停机制: 容忍 {config.early_stopping_patience} 次无改善评估")
    
    # 计算总训练步数
    total_train_steps = len(train_loader) * config.num_epochs
    print(f"📊 总训练步数: {total_train_steps}")
    
    # 创建主进度条
    main_pbar = tqdm(total=total_train_steps, desc="总体训练进度", position=0)
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        train_batches = 0
        
        # 创建epoch进度条
        epoch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}", position=1, leave=False)
        
        for batch in epoch_pbar:
            # 移动到设备
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            
            # 前向传播
            outputs = model(**inputs)
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            epoch_train_loss += loss.item()
            train_batches += 1
            global_step += 1
            
            # 更新进度条
            current_loss = epoch_train_loss / train_batches
            current_lr = scheduler.get_last_lr()[0]
            epoch_pbar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            main_pbar.update(1)
            
            # 记录日志
            if global_step % config.logging_steps == 0:
                print(f"\n📝 Step {global_step}, Loss: {current_loss:.4f}, LR: {current_lr:.2e}")
        
        epoch_pbar.close()
        avg_train_loss = epoch_train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        val_batches = 0
        
        print("🔍 验证中...")
        # 创建验证进度条
        val_pbar = tqdm(val_loader, desc="验证进度", position=1, leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'labels': batch['labels'].to(device)
                }
                
                outputs = model(**inputs)
                epoch_val_loss += outputs.loss.item()
                val_batches += 1
                
                # 更新验证进度条
                current_val_loss = epoch_val_loss / val_batches
                val_pbar.set_postfix({'val_loss': f'{current_val_loss:.4f}'})
        
        val_pbar.close()
        
        avg_val_loss = epoch_val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        print(f"📊 Epoch {epoch+1} 结果:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证损失: {avg_val_loss:.4f}")
        print(f"  学习率: {scheduler.get_last_lr()[0]:.2e}")
        
        # 早停检查
        if avg_val_loss < best_val_loss - config.early_stopping_threshold:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            model.save_pretrained("./gpt2-lora-best")
            tokenizer.save_pretrained("./gpt2-lora-best")
            print(f"🎯 保存最佳模型，验证损失: {avg_val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ 早停计数: {patience_counter}/{config.early_stopping_patience}")
            
            if patience_counter >= config.early_stopping_patience:
                print("🛑 触发早停机制!")
                break
    
    # 关闭主进度条
    main_pbar.close()
    
    # 打印训练总结
    print("\n📈 训练总结:")
    print(f"  最佳验证损失: {best_val_loss:.4f}")
    print(f"  最终训练损失: {train_losses[-1]:.4f}")
    print(f"  训练轮次: {len(train_losses)}")
    
    # 加载最佳模型
    try:
        model = GPT2LMHeadModel.from_pretrained("./gpt2-lora-best")
        model = get_peft_model(model, LoraConfig.from_pretrained("./gpt2-lora-best"))
        model = model.to(device)
        print("💾 加载最佳模型完成")
    except:
        print("⚠️ 无法加载最佳模型，使用当前模型")
    
    # 保存最终模型
    model.save_pretrained("./gpt2-lora-final")
    tokenizer.save_pretrained("./gpt2-lora-final")
    print("💾 保存最终模型完成")
    
    return model, tokenizer

def test_generation(model, tokenizer, device):
    """测试生成效果"""
    print("\n🧪 测试文本生成...")
    
    # 测试提示
    test_prompts = [
        "Instruction: Explain the concept of machine learning in simple terms.\nResponse:",
        "Question: What is the capital of France?\nAnswer:",
        "Math Problem: If a train travels at 60 mph for 2 hours, how far does it go?\nSolution:"
    ]
    
    model.eval()
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n--- 测试 {i+1} ---")
        print(f"提示: {prompt}")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=len(inputs['input_ids'][0]) + 100,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"生成结果: {generated_text}")

def main():
    """主函数"""
    config = LoRAConfig()
    
    # 选择训练方法
    training_method = "enhanced"  # 可选: "simple", "enhanced"
    
    if training_method == "simple":
        # 使用简化训练循环
        model, tokenizer = train_lora_gpt2_simple(config)
    elif training_method == "enhanced":
        # 使用增强版训练循环（推荐）
        model, tokenizer = train_lora_gpt2_with_tqdm(config)
    else:
        # 使用Trainer版本
        model, tokenizer = train_lora_gpt2(config)
    
    # 测试生成
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_generation(model, tokenizer, device)
    
    print("✅ LoRA微调完成！")

if __name__ == "__main__":
    main()