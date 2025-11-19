import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import json
import numpy as np
from torch.cuda.amp import autocast, GradScaler

class LongBenchEvaluator:
    """LongBench评估器 - 专门测试长文本理解能力 + FP16支持"""
    
    def __init__(self, model, tokenizer, device, use_fp16=True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_fp16 = use_fp16 and torch.cuda.is_available()
        self.results = {}
        
        print(f"  LongBench评估器初始化 - FP16: {'启用' if self.use_fp16 else '禁用'}")

    def evaluate_single_task(self, task_name, dataset, max_samples=20):
        """评估单个LongBench任务 - 支持FP16"""
        print(f"  评估任务: {task_name}")
        
        if max_samples and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        task_results = {
            'exact_match': [],
            'rouge_scores': [],
            'perplexity': []
        }
        
        for i, sample in enumerate(tqdm(dataset, desc=f"评估 {task_name}")):
            try:
                # 构建输入
                context = sample.get('context', '')
                question = sample.get('input', '') or sample.get('question', '')
                ground_truth = sample.get('answers', [''])[0] if sample.get('answers') else sample.get('target', '')
                
                # 根据任务类型构建提示
                if 'qa' in task_name.lower():
                    prompt = f"基于以下文档回答问题：\n\n文档：{context}\n\n问题：{question}\n\n答案："
                elif 'summar' in task_name.lower():
                    prompt = f"为以下文档生成摘要：\n\n{context}\n\n摘要："
                else:
                    prompt = f"{context}\n\n{question}"
                
                # 分词
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048  # 限制输入长度
                ).to(self.device)
                
                # 生成 - 支持FP16
                with torch.no_grad():
                    if self.use_fp16:
                        with autocast():
                            outputs = self.model.generate(
                                inputs.input_ids,
                                max_new_tokens=256,
                                do_sample=False,
                                pad_token_id=self.tokenizer.eos_token_id,
                                num_return_sequences=1
                            )
                    else:
                        outputs = self.model.generate(
                            inputs.input_ids,
                            max_new_tokens=256,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id,
                            num_return_sequences=1
                        )
                
                # 解码生成结果
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                prediction = generated_text[len(prompt):].strip()
                
                # 计算精确匹配
                exact_match = 1.0 if prediction.strip() == ground_truth.strip() else 0.0
                task_results['exact_match'].append(exact_match)
                
                # 计算困惑度（如果适用）
                if context:
                    perplexity = self.calculate_perplexity(context)
                    task_results['perplexity'].append(perplexity)
                
                # 打印前几个样本的示例
                if i < 2:
                    print(f"\n--- {task_name} 样本 {i} ---")
                    print(f"输入: {prompt[:200]}...")
                    print(f"预测: {prediction[:100]}...")
                    print(f"真实: {ground_truth[:100]}...")
                    print(f"精确匹配: {exact_match}")
                    print(f"FP16: {'启用' if self.use_fp16 else '禁用'}")
                    
            except Exception as e:
                print(f"  评估样本 {i} 时出错: {e}")
                continue
        
        # 汇总任务结果
        if task_results['exact_match']:
            task_summary = {
                'exact_match': np.mean(task_results['exact_match']),
                'samples_evaluated': len(task_results['exact_match']),
                'fp16_enabled': self.use_fp16
            }
            if task_results['perplexity']:
                task_summary['avg_perplexity'] = np.mean(task_results['perplexity'])
            
            self.results[task_name] = task_summary
            return task_summary
        else:
            print(f"  任务 {task_name} 无有效结果")
            return None

    def calculate_perplexity(self, text):
        """计算文本的困惑度 - 支持FP16"""
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            with torch.no_grad():
                if self.use_fp16:
                    with autocast():
                        outputs = self.model(inputs.input_ids, labels=inputs.input_ids)
                        loss = outputs.loss
                else:
                    outputs = self.model(inputs.input_ids, labels=inputs.input_ids)
                    loss = outputs.loss
                
                perplexity = torch.exp(loss).item()
            return perplexity
        except:
            return float('inf')
    
    def evaluate_all_tasks(self, max_samples_per_task=20):
        """评估所有LongBench任务"""
        print("  开始LongBench任务评估")
        
        # 加载LongBench数据集
        try:
            longbench_tasks = {
                'single_doc_qa': load_dataset("THUDM/LongBench", "single_doc_qa", split="test"),
                'multi_doc_qa': load_dataset("THUDM/LongBench", "multi_doc_qa", split="test"),
                'summarization': load_dataset("THUDM/LongBench", "summarization", split="test")
            }
        except Exception as e:
            print(f"  加载LongBench失败: {e}")
            return self.results
        
        # 评估每个任务
        for task_name, dataset in longbench_tasks.items():
            if dataset is not None:
                self.evaluate_single_task(task_name, dataset, max_samples_per_task)
        
        # 生成评估报告
        self.generate_report()
        return self.results
    
    def generate_report(self):
        """生成评估报告"""
        if not self.results:
            print("  无评估结果可报告")
            return
        
        print("\n" + "="*60)
        print("  LongBench评估报告")
        print("="*60)
        
        overall_scores = []
        for task_name, scores in self.results.items():
            em_score = scores.get('exact_match', 0)
            overall_scores.append(em_score)
            print(f"  {task_name}:")
            print(f"   - 精确匹配: {em_score:.4f}")
            print(f"   - 评估样本: {scores.get('samples_evaluated', 0)}")
            if 'avg_perplexity' in scores:
                print(f"   - 平均困惑度: {scores['avg_perplexity']:.4f}")
        
        if overall_scores:
            avg_score = np.mean(overall_scores)
            print(f"\n  总体平均精确匹配: {avg_score:.4f}")
        
        # 保存结果
        import json
        with open("longbench_evaluation_results.json", "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"  详细结果已保存至: longbench_evaluation_results.json")

class SmearTrainer:
    """SMEAR训练器 - 修复早停机制 + FP16支持"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        
        # FP16配置
        self.use_fp16 = getattr(config, 'use_fp16', True)  # 默认为True
        self.scaler = GradScaler() if self.use_fp16 and torch.cuda.is_available() else None
        
        # 优化器 - 只训练Adapter参数
        trainable_params = [p for p in model.parameters() if p.requires_grad]
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
            'fp16_enabled': self.use_fp16 and self.scaler is not None  # 记录FP16状态
        }
        
        # 早停相关变量
        self.patience = getattr(config, 'patience', 3)
        self.patience_counter = 0
        self.min_delta = getattr(config, 'min_delta', 1e-4)
        
        print(f"  初始化SMEAR训练器")
        print(f"   - 设备: {self.device}")
        print(f"   - FP16: {'启用' if self.train_stats['fp16_enabled'] else '禁用'}")
        print(f"   - 可训练参数: {sum(p.numel() for p in trainable_params):,}")
        print(f"   - 早停耐心值: {self.patience} epochs")
        if self.train_stats['fp16_enabled']:
            print(f"   - 使用混合精度训练 (FP16)")

    def train_epoch(self, epoch):
        """训练一个epoch - 支持FP16"""
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
                
                # 梯度裁剪（在缩放后的梯度上）
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
            
            # 学习率调度（两种模式都需要）
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
        """验证 - 支持FP16"""
        self.model.eval()
        total_loss = 0
        total_perplexity = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # 验证时也使用FP16以减少内存占用
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
        """在测试集上评估模型 - 支持FP16"""
        if self.test_loader is None:
            print("   未提供测试集，跳过测试评估")
            return None, None
        
        # 如果指定了模型路径，则重新加载完整模型
        if model_path is not None:
            self.load_complete_model(model_path)
            print(f"  加载完整模型进行测试: {model_path}")
        
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

    def save_complete_model(self, path):
        """保存完整模型（基础模型 + SMEAR适配器）"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        save_path = os.path.join(self.config.output_dir, path)
        os.makedirs(save_path, exist_ok=True)
        
        # 保存完整模型状态
        complete_state_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'training_stats': self.train_stats,
            'smear_adapters_only': False,
            'fp16_enabled': self.use_fp16  # 保存FP16状态
        }
        
        torch.save(complete_state_dict, os.path.join(save_path, 'complete_model.pth'))
        print(f"  保存完整模型到 {save_path}")
    
    def save_smear_adapters_only(self, path):
        """仅保存SMEAR适配器参数（用于继续训练）"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        save_path = os.path.join(self.config.output_dir, path)
        os.makedirs(save_path, exist_ok=True)
        
        # 保存SMEAR适配器参数
        model_state_dict = self.model.state_dict()
        smear_state_dict = {
            name: param for name, param in model_state_dict.items()
            if any(key in name for key in ['smear_adapters', 'adapter_alpha'])
        }
        
        adapter_only_state_dict = {
            'smear_adapters': smear_state_dict,
            'config': self.config.__dict__,
            'training_stats': self.train_stats,
            'smear_adapters_only': True  # 标记这是仅适配器
        }
        
        torch.save(adapter_only_state_dict, os.path.join(save_path, 'smear_adapters.pth'))
        print(f"💾 保存 {len(smear_state_dict)} 个SMEAR适配器参数到 {save_path}")
    
    def load_complete_model(self, model_path):
        """加载完整模型"""
        checkpoint_path = os.path.join(model_path, 'complete_model.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"  完整模型文件不存在: {checkpoint_path}")
            print("   尝试加载仅适配器版本...")
            return self.load_smear_adapters_only(model_path)
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载完整模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 更新配置和训练统计（可选）
        if 'training_stats' in checkpoint:
            self.train_stats.update(checkpoint['training_stats'])
        
        print(f"  从 {model_path} 加载完整模型")
        return True
    
    def load_smear_adapters_only(self, model_path):
        """仅加载SMEAR适配器参数"""
        checkpoint_path = os.path.join(model_path, 'smear_adapters.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"  适配器文件不存在: {checkpoint_path}")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 获取当前模型状态字典
        model_state_dict = self.model.state_dict()
        
        # 只更新SMEAR相关的参数
        smear_adapters = checkpoint['smear_adapters']
        for name, param in smear_adapters.items():
            if name in model_state_dict:
                model_state_dict[name].copy_(param)
            else:
                print(f"   跳过不匹配的参数: {name}")
        
        # 加载更新后的状态字典
        self.model.load_state_dict(model_state_dict)
        
        # 更新训练统计（可选）
        if 'training_stats' in checkpoint:
            self.train_stats.update(checkpoint['training_stats'])
        
        print(f"  从 {model_path} 加载SMEAR适配器参数")
        return True
    
    def check_early_stop(self, current_val_loss, best_val_loss, epoch):
        """检查是否应该早停 - 修复版本"""
        # 检查是否有显著改善（超过最小阈值）
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
        """完整训练流程 - 早停机制"""
        best_val_loss = float('inf')
        best_epoch = -1
        
        print(" 开始训练")
        
        for epoch in range(self.config.num_epochs):
            print(f"\n📍 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_perplexity = self.validate(epoch)
            
            # 简化的训练统计输出
            print(f"  训练统计:")
            print(f"  - 训练损失: {train_loss:.4f}")
            print(f"  - 验证损失: {val_loss:.4f}")
            print(f"  - 验证困惑度: {val_perplexity:.4f}")
            
            # 检查是否有改善
            has_improvement = val_loss < best_val_loss - self.min_delta
            
            # 保存最佳模型（只在性能提升时保存）
            if has_improvement:
                best_val_loss = val_loss
                best_epoch = epoch
                
                # 保存完整模型用于测试
                self.save_complete_model("best_smear_model")
                # 同时保存适配器参数用于继续训练
                self.save_smear_adapters_only("best_smear_adapters")
                
                self.train_stats['best_val_loss'] = best_val_loss
                self.train_stats['best_epoch'] = best_epoch
                print(f"  保存最佳模型 (验证损失: {val_loss:.4f}, Epoch: {epoch})")
            else:
                print(f"  验证损失未改善，跳过保存 (当前最佳: {best_val_loss:.4f})")
            
            # 检查早停条件 - 只在没有改善时检查
            if not has_improvement and self.check_early_stop(val_loss, best_val_loss, epoch):
                print(f"   训练在 Epoch {epoch} 提前停止")
                break
            
            # 保存训练统计
            self.save_training_stats()
        
        # 训练结束后在测试集上评估最佳模型
        print(f"\n{'='*50}")
        print("  训练完成，开始在测试集上评估最佳模型...")
        print(f"{'='*50}")
        
        test_loss, test_perplexity = self.test("best_smear_model")
        
        # 最终报告
        print(f"\n{'='*50}")
        print("  最终训练报告:")
        print(f"{'='*50}")
        print(f"  最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch})")
        print(f"  最终训练轮数: {len(self.train_stats['losses'])}")
        if self.train_stats['early_stop_epoch'] is not None:
            print(f"   早停触发于: Epoch {self.train_stats['early_stop_epoch']}")
        if test_loss is not None:
            print(f"  测试集损失: {test_loss:.4f}")
            print(f"  测试集困惑度: {test_perplexity:.4f}")
        
        # 保存最终报告
        self.save_final_report(best_val_loss, best_epoch, test_loss, test_perplexity)
        
        return best_val_loss
    
    def save_training_stats(self):
        """保存训练统计"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(os.path.join(self.config.output_dir, 'smear_training_stats.json'), 'w') as f:
            json.dump(self.train_stats, f, indent=2)
    
    def save_final_report(self, best_val_loss, best_epoch, test_loss, test_perplexity):
        """保存最终训练报告"""
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
                'device': str(self.device)
            },
            'early_stop_config': {
                'patience': self.patience,
                'min_delta': self.min_delta
            },
            'config': self.config.__dict__
        }
        
        report_path = os.path.join(self.config.output_dir, 'final_training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  最终报告已保存到: {report_path}")

class IntegratedSmearTrainer(SmearTrainer):
    """集成LongBench评估的SMEAR训练器"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, config):
        super().__init__(model, train_loader, val_loader, test_loader, config)
        self.longbench_evaluator = None
    
    def setup_longbench_evaluator(self):
        """设置LongBench评估器"""
        self.longbench_evaluator = LongBenchEvaluator(
            self.model, 
            self.tokenizer,  # 需要确保tokenizer可用
            self.device
        )
        print("  LongBench评估器已设置")
    
    def evaluate_on_longbench(self, max_samples_per_task=10):
        """在LongBench上评估模型"""
        if self.longbench_evaluator is None:
            self.setup_longbench_evaluator()
        
        print("\n" + "="*50)
        print("  开始在LongBench上评估模型...")
        print("="*50)
        
        results = self.longbench_evaluator.evaluate_all_tasks(max_samples_per_task)
        
        # 将LongBench结果集成到训练统计中
        if 'longbench_results' not in self.train_stats:
            self.train_stats['longbench_results'] = {}
        
        self.train_stats['longbench_results'][f'epoch_{len(self.train_stats["losses"])}'] = results
        
        return results
    
    def train_with_longbench_eval(self, longbench_eval_interval=2):
        """带LongBench评估的训练流程"""
        best_val_loss = float('inf')
        best_epoch = -1
        
        print("  开始带LongBench评估的SMEAR训练")
        
        for epoch in range(self.config.num_epochs):
            print(f"\n  Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_perplexity = self.validate(epoch)
            
            print(f"  训练统计:")
            print(f"  - 训练损失: {train_loss:.4f}")
            print(f"  - 验证损失: {val_loss:.4f}")
            print(f"  - 验证困惑度: {val_perplexity:.4f}")
            
            # 定期在LongBench上评估
            if (epoch + 1) % longbench_eval_interval == 0:
                self.evaluate_on_longbench(max_samples_per_task=10)
            
            # 检查是否有改善
            has_improvement = val_loss < best_val_loss - self.min_delta
            
            if has_improvement:
                best_val_loss = val_loss
                best_epoch = epoch
                
                # 保存最佳模型
                self.save_complete_model("best_smear_model")
                self.save_smear_adapters_only("best_smear_adapters")
                
                self.train_stats['best_val_loss'] = best_val_loss
                self.train_stats['best_epoch'] = best_epoch
                print(f"  保存最佳模型 (验证损失: {val_loss:.4f}, Epoch: {epoch})")
            else:
                print(f"  验证损失未改善，跳过保存 (当前最佳: {best_val_loss:.4f})")
            
            # 检查早停
            if not has_improvement and self.check_early_stop(val_loss, best_val_loss, epoch):
                print(f"   训练在 Epoch {epoch} 提前停止")
                break
            
            # 保存训练统计
            self.save_training_stats()
        
        # 最终评估
        print(f"\n{'='*50}")
        print("  训练完成，开始最终评估...")
        print(f"{'='*50}")
        
        # 在BookCorpus测试集上评估
        test_loss, test_perplexity = self.test("best_smear_model")
        
        # 在LongBench上最终评估
        final_longbench_results = self.evaluate_on_longbench(max_samples_per_task=20)
        
        # 最终报告
        print(f"\n{'='*50}")
        print("  最终训练报告:")
        print(f"{'='*50}")
        print(f"  最佳验证损失: {best_val_loss:.4f} (Epoch {best_epoch})")
        print(f"  最终训练轮数: {len(self.train_stats['losses'])}")
        if test_loss is not None:
            print(f"  BookCorpus测试集:")
            print(f"  - 测试损失: {test_loss:.4f}")
            print(f"  - 测试困惑度: {test_perplexity:.4f}")
        
        if final_longbench_results:
            print(f"  LongBench评估:")
            for task, scores in final_longbench_results.items():
                print(f"  - {task}: 精确匹配 = {scores.get('exact_match', 0):.4f}")
        
        # 保存最终报告
        self.save_final_report_with_longbench(
            best_val_loss, best_epoch, test_loss, test_perplexity, final_longbench_results
        )
        
        return best_val_loss
    
    def save_final_report_with_longbench(self, best_val_loss, best_epoch, test_loss, test_perplexity, longbench_results):
        """保存包含LongBench结果的最终报告"""
        report = {
            'training_summary': {
                'best_validation_loss': best_val_loss,
                'best_epoch': best_epoch,
                'test_loss': test_loss,
                'test_perplexity': test_perplexity,
                'total_training_epochs': len(self.train_stats['losses']),
                'early_stop_epoch': self.train_stats['early_stop_epoch'],
            },
            'longbench_evaluation': longbench_results,
            'model_info': {
                'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'device': str(self.device)
            },
            'dataset_info': {
                'training_data': 'BookCorpus',
                'evaluation_data': 'LongBench + BookCorpus Test Set'
            }
        }
        
        report_path = os.path.join(self.config.output_dir, 'final_training_report_with_longbench.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  最终报告已保存到: {report_path}")