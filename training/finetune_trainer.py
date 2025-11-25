import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, GPT2Config, GPT2Tokenizer
from tqdm import tqdm
import os
import json
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime

class LoRATrainer:
    """GPT-2 + LoRA微调训练器"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # FP16配置
        self.use_fp16 = getattr(config, 'use_fp16', True)
        self.scaler = GradScaler() if self.use_fp16 and torch.cuda.is_available() else None
        
        # 获取freeze_base_model配置，优先从模型获取，然后从配置获取
        self.freeze_base_model = getattr(model, 'freeze_base_model', getattr(config, 'freeze_base_model', True))
        
        # 优化器
        if self.freeze_base_model:
            # 只训练非基础模型参数
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            print("  仅微调非冻结参数")
        else:
            # 训练所有参数
            trainable_params = [p for p in model.parameters()]
            print("  全模型微调")
        self.optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
        
        # 学习率调度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.total_steps
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 训练统计
        self.train_stats = {
            'losses': [],
            'perplexities': [],
            'learning_rates': [],
            'test_loss': None,
            'test_perplexity': None,
            'best_val_loss': float('inf'),
            'best_epoch': -1,
            'early_stop_epoch': None,
            'fp16_enabled': self.use_fp16 and self.scaler is not None
        }
        
        # 早停相关变量
        self.patience = getattr(config, 'patience', 3)
        self.patience_counter = 0
        self.min_delta = getattr(config, 'min_delta', 1e-4)
        
        print(f"  初始化微调训练器")
        print(f"   - 设备: {self.device}")
        print(f"   - FP16: {'启用' if self.train_stats['fp16_enabled'] else '禁用'}")
        print(f"   - 可训练参数: {sum(p.numel() for p in trainable_params):,}")
        print(f"   - 早停耐心值: {self.patience} epochs")
        if self.train_stats['fp16_enabled']:
            print(f"   - 使用混合精度训练 (FP16)")

    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # FP16前向传播
            if self.use_fp16 and self.scaler is not None:
                with autocast():
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs['loss']
                
                # FP16反向传播和梯度缩放
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # 优化器步骤和缩放器更新
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                # 普通FP32训练
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录损失
            total_loss += loss.item()
            
            # 进度条更新
            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{current_lr:.2e}',
                'Patience': f'{self.patience_counter}/{self.patience}',
                'FP16': 'ON' if self.use_fp16 else 'OFF'
            })
            
            # 定期记录学习率
            if batch_idx % 100 == 0:
                self.train_stats['learning_rates'].append(current_lr)
        
        # 记录epoch统计
        avg_loss = total_loss / len(self.train_loader)
        self.train_stats['losses'].append(avg_loss)
        
        return avg_loss

    def validate(self, epoch):
        """验证"""
        self.model.eval()
        total_loss = 0
        total_perplexity = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 验证时使用FP16
                if self.use_fp16:
                    with autocast():
                        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs['loss']
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs['loss']
                
                total_loss += loss.item()
                
                # 计算困惑度
                perplexity = torch.exp(torch.tensor(loss.item()))
                total_perplexity += perplexity.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_perplexity = total_perplexity / len(self.val_loader)
        
        self.train_stats['perplexities'].append(avg_perplexity)
        
        return avg_loss, avg_perplexity

    def test(self, model_path=None):
        """在测试集上评估模型"""
        if self.test_loader is None:
            print("   未提供测试集，跳过测试评估")
            return None, None
        
        # 如果指定了模型路径，则重新加载模型
        if model_path is not None:
            self.load_model(model_path)
            print(f"  加载模型进行测试: {model_path}")
        
        self.model.eval()
        total_loss = 0
        total_perplexity = 0
        
        print("  开始在测试集上评估...")
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Testing'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 测试时使用FP16
                if self.use_fp16:
                    with autocast():
                        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs['loss']
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs['loss']
                
                total_loss += loss.item()
                
                # 计算困惑度
                perplexity = torch.exp(torch.tensor(loss.item()))
                total_perplexity += perplexity.item()
        
        avg_loss = total_loss / len(self.test_loader)
        avg_perplexity = total_perplexity / len(self.test_loader)
        
        # 保存测试结果
        self.train_stats['test_loss'] = avg_loss
        self.train_stats['test_perplexity'] = avg_perplexity
        
        print(f"  测试集结果:")
        print(f"  - 测试损失: {avg_loss:.4f}")
        print(f"  - 测试困惑度: {avg_perplexity:.4f}")
        
        return avg_loss, avg_perplexity

    def save_lora_model(self, path):
        """保存LoRA模型参数（不包含基础模型）"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        save_path = os.path.join(self.config.output_dir, path)
        os.makedirs(save_path, exist_ok=True)
        
        # 只保存LoRA参数
        model_state_dict = self.model.state_dict()
        lora_state_dict = {
            name: param for name, param in model_state_dict.items()
            if 'lora' in name.lower()
        }
        
        lora_checkpoint = {
            'lora_state_dict': lora_state_dict,
            'config': self.config.__dict__,
            'training_stats': self.train_stats,
            'fp16_enabled': self.use_fp16
        }
        
        torch.save(lora_checkpoint, os.path.join(save_path, 'lora_model.pth'))
        print(f"  保存LoRA模型到 {save_path}")
        print(f"  - LoRA参数数量: {len(lora_state_dict)}")
        total_lora_params = sum(p.numel() for p in lora_state_dict.values())
        print(f"  - LoRA参数量: {total_lora_params:,}")
        
        return save_path

    def save_merged_model(self, path="merged_model"):
        """保存合并LoRA权重后的完整模型，供lm-evaluation使用"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        save_path = os.path.join(self.config.output_dir, path)
        os.makedirs(save_path, exist_ok=True)
        
        print("开始合并LoRA权重并保存完整模型...")
        
        # 确保模型在eval模式
        self.model.eval()
        
        # 如果模型有合并方法，使用它
        if hasattr(self.model, 'merge_lora'):
            print("使用模型内置LoRA合并方法...")
            self.model.merge_lora()
        elif hasattr(self.model, 'merge_adapter'):
            print("使用adapter合并方法...")
            self.model.merge_adapter()
        else:
            print("⚠ 模型没有内置合并方法，直接保存当前状态...")
        
        # 获取模型状态字典
        model_state_dict = self.model.state_dict()
        
        # 修复权重名称：移除 'gpt2.' 前缀
        fixed_state_dict = {}
        for key, value in model_state_dict.items():
            if key.startswith('gpt2.'):
                # 移除 'gpt2.' 前缀
                new_key = key[5:]
            else:
                new_key = key
            fixed_state_dict[new_key] = value
        
        print(f"权重名称修复完成: {len(model_state_dict)} -> {len(fixed_state_dict)}")
        
        # 保存修复后的权重
        torch.save(fixed_state_dict, os.path.join(save_path, 'pytorch_model.bin'))
        
        # 保存标准配置
        if hasattr(self.model, 'config'):
            config = self.model.config
            if hasattr(config, 'to_dict'):
                config_dict = config.to_dict()
            else:
                config_dict = vars(config)
        else:
            # 创建默认GPT-2配置
            config_dict = {}
        
        # 确保使用标准GPT2Config
        standard_config = GPT2Config(**config_dict)
        standard_config.save_pretrained(save_path)
        
        # 保存tokenizer
        try:
            tokenizer = GPT2Tokenizer.from_pretrained("/home/yang/gpt2-moe-adapter/gpt2")
            tokenizer.save_pretrained(save_path)
            print("✓ Tokenizer保存成功")
        except Exception as e:
            print(f"⚠ Tokenizer保存失败: {e}")
        
        # 生成使用说明
        self._generate_usage_instructions(save_path)
        
        print(f"✅ 合并后的完整模型已保存到: {save_path}")
        print(f"  可用于lm-evaluation-harness测试的命令:")
        print(f"  lm_eval --model hf --model_args pretrained={save_path} --tasks [task_name]")
        
        return save_path

    def _generate_usage_instructions(self, model_path):
        """生成使用说明文件"""
        instructions = {
            "model_type": "gpt2",
            "training_method": "LoRA fine-tuning",
            "usage": {
                "lm_evaluation_harness": f"lm_eval --model hf --model_args pretrained={model_path} --tasks [task_name]",
                "transformers": "from transformers import GPT2LMHeadModel, GPT2Tokenizer\nmodel = GPT2LMHeadModel.from_pretrained('path/to/model')\ntokenizer = GPT2Tokenizer.from_pretrained('path/to/model')",
                "example_tasks": ["hellaswag", "winogrande", "piqa", "lambada", "mmlu"]
            },
            "note": "This model includes base GPT-2 weights merged with LoRA adapter weights"
        }
        
        import json
        with open(os.path.join(model_path, 'usage_instructions.json'), 'w') as f:
            json.dump(instructions, f, indent=2)

    def validate_model_loading(self, model_path):
        """验证保存的模型是否能正确加载"""
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        
        print(f"验证模型加载: {model_path}")
        
        try:
            # 尝试加载模型
            model = GPT2LMHeadModel.from_pretrained(model_path)
            tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            
            # 检查模型参数
            total_params = sum(p.numel() for p in model.parameters())
            print(f"✅ 模型加载成功")
            print(f"   - 总参数量: {total_params:,}")
            print(f"   - 模型类型: {type(model)}")
            
            # 简单推理测试
            inputs = tokenizer("Hello, how are you?", return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
            print(f"   - 输出形状: {logits.shape}")
            print(f"   - 推理测试: ✓ 通过")
            
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False

    def load_model(self, model_path):
        """加载LoRA模型"""
        checkpoint_path = os.path.join(model_path, 'lora_model.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"  LoRA模型文件不存在: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 获取当前模型状态
        model_state_dict = self.model.state_dict()
        
        # 只加载LoRA参数
        lora_params = checkpoint['lora_state_dict']
        for name, param in lora_params.items():
            if name in model_state_dict:
                model_state_dict[name].copy_(param)
            else:
                print(f"   跳过不匹配的LoRA参数: {name}")
        
        # 加载更新后的状态
        self.model.load_state_dict(model_state_dict)
        
        # 更新训练统计
        if 'training_stats' in checkpoint:
            self.train_stats.update(checkpoint['training_stats'])
        
        print(f"  从 {model_path} 加载LoRA模型")
        return True

    def check_early_stop(self, current_val_loss, best_val_loss, epoch):
        """检查是否应该早停"""
        improvement = best_val_loss - current_val_loss
        
        if improvement > self.min_delta:
            # 有显著改善，重置计数器
            self.patience_counter = 0
            print(f"  验证损失改善: {improvement:.6f} > {self.min_delta}")
            return False
        else:
            # 没有显著改善，增加计数器
            self.patience_counter += 1
            print(f"  验证损失未改善，耐心计数: {self.patience_counter}/{self.patience}")
            
            # 检查是否达到耐心限制
            if self.patience_counter >= self.patience:
                print(f"  早停触发！连续 {self.patience} 个epoch验证损失未改善")
                self.train_stats['early_stop_epoch'] = epoch
                return True
            
            return False

    def train(self):
        """完整训练流程 - 只负责训练和保存模型，不进行外部评估"""
        best_val_loss = float('inf')
        best_epoch = -1
        
        print(" 开始微调训练")
        
        for epoch in range(self.config.num_epochs):
            print(f"\n  Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_perplexity = self.validate(epoch)
            
            # 训练统计
            print(f"  训练统计:")
            print(f"  - 训练损失: {train_loss:.4f}")
            print(f"  - 验证损失: {val_loss:.4f}")
            print(f"  - 验证困惑度: {val_perplexity:.4f}")
            
            # 检查是否有改善
            has_improvement = val_loss < best_val_loss - self.min_delta
            
            # 保存最佳模型
            if has_improvement:
                best_val_loss = val_loss
                best_epoch = epoch
                
                # 保存最佳LoRA模型
                self.save_lora_model("best_lora_model")
                
                self.train_stats['best_val_loss'] = best_val_loss
                self.train_stats['best_epoch'] = best_epoch
                print(f"  保存最佳LoRA模型 (验证损失: {val_loss:.4f}, Epoch: {epoch})")
            else:
                print(f"  验证损失未改善，跳过保存 (当前最佳: {best_val_loss:.4f})")
            
            # 检查早停条件
            if not has_improvement and self.check_early_stop(val_loss, best_val_loss, epoch):
                print(f"   训练在 Epoch {epoch} 提前停止")
                break
            
            # 保存训练统计
            self.save_training_stats()
        
        # 训练结束后在测试集上评估最佳模型
        print(f"\n{'='*50}")
        print("  训练完成，开始在测试集上评估最佳模型...")
        print(f"{'='*50}")
        
        test_loss, test_perplexity = self.test("best_lora_model")
        
        # 保存合并后的完整模型供外部评估使用
        print(f"\n{'='*50}")
        print("  保存合并LoRA权重后的完整模型...")
        print(f"{'='*50}")
        
        merged_model_path = self.save_merged_model("final_merged_model")
        
        # 验证合并后的模型
        print(f"\n验证合并后的模型...")
        if self.validate_model_loading(merged_model_path):
            print("✅ 模型验证通过，可以用于lm-evaluation-harness评估")
        else:
            print("⚠ 模型验证失败，请检查保存的模型")
        
        # 最终报告
        print(f"\n{'='*50}")
        print("  最终微调训练报告:")
        print(f"{'='*50}")
        print(f"  最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch})")
        print(f"  最终训练轮数: {len(self.train_stats['losses'])}")
        if self.train_stats['early_stop_epoch'] is not None:
            print(f"   早停触发于: Epoch {self.train_stats['early_stop_epoch']}")
        if test_loss is not None:
            print(f"  测试集损失: {test_loss:.4f}")
            print(f"  测试集困惑度: {test_perplexity:.4f}")
        
        print(f"\n{'='*50}")
        print("  后续评估说明:")
        print(f"{'='*50}")
        print(f"  1. LoRA模型已保存到: {os.path.join(self.config.output_dir, 'best_lora_model')}")
        print(f"  2. 合并后的完整模型已保存到: {merged_model_path}")
        print(f"  3. 使用以下命令进行lm-evaluation评估:")
        print(f"     lm_eval --model hf --model_args pretrained={merged_model_path} --tasks [task_name]")
        print(f"  4. 示例任务: hellaswag, winogrande, piqa, mmlu, lambada")
        
        # 保存最终报告
        self.save_final_report(best_val_loss, best_epoch, test_loss, test_perplexity, merged_model_path)
        
        return best_val_loss

    def save_training_stats(self):
        """保存训练统计"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(os.path.join(self.config.output_dir, 'finetune_training_stats.json'), 'w') as f:
            json.dump(self.train_stats, f, indent=2)
    
    def save_final_report(self, best_val_loss, best_epoch, test_loss, test_perplexity, merged_model_path):
        """保存最终LoRA训练报告"""
        # 从模型获取LoRA配置
        lora_config = getattr(self.model, 'config', None)
        
        report = {
            'training_summary': {
                'best_validation_loss': best_val_loss,
                'best_epoch': best_epoch,
                'test_loss': test_loss,
                'test_perplexity': test_perplexity,
                'total_training_epochs': len(self.train_stats['losses']),
                'early_stop_epoch': self.train_stats['early_stop_epoch'],
                'final_learning_rate': self.train_stats['learning_rates'][-1] if self.train_stats['learning_rates'] else 0,
                'patience_used': self.patience_counter
            },
            'model_info': {
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'device': str(self.device),
                'merged_model_path': merged_model_path
            },
            'lora_config': {
                'rank': getattr(lora_config, 'adapter_rank', None),
                'alpha': getattr(lora_config, 'adapter_alpha', None),
                'base_model': getattr(lora_config, 'base_model', None)
            },
            'evaluation_instructions': {
                'lm_eval_command': f"lm_eval --model hf --model_args pretrained={merged_model_path} --tasks [task_name]",
                'suggested_tasks': ["hellaswag", "winogrande", "piqa", "mmlu"],
                'note': "Run lm-evaluation-harness separately after training completes"
            },
            'early_stop_config': {
                'patience': self.patience,
                'min_delta': self.min_delta
            },
            'config': self.config.__dict__
        }
        
        report_path = os.path.join(self.config.output_dir, 'final_lora_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  最终报告已保存到: {report_path}")