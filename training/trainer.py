import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import json

class MoETrainer:
    """MoE Adapter训练器"""
    
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
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
            'lm_losses': [],
            'load_balancing_losses': [],
            'perplexities': []
        }
        
        print(f"🚀 初始化训练器")
        print(f"   - 设备: {self.device}")
        print(f"   - 可训练参数: {sum(p.numel() for p in trainable_params):,}")
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        total_lm_loss = 0
        total_balance_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # 统计
            total_loss += loss.item()
            if outputs['lm_loss'] is not None:
                total_lm_loss += outputs['lm_loss'].item()
            if outputs['load_balancing_loss'] is not None:
                total_balance_loss += outputs['load_balancing_loss'].item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LM Loss': f'{outputs["lm_loss"].item() if outputs["lm_loss"] is not None else 0:.4f}',
                'Balance Loss': f'{outputs["load_balancing_loss"].item() if outputs["load_balancing_loss"] is not None else 0:.4f}'
            })
            
            # 记录路由统计信息（每100个batch）
            if batch_idx % 100 == 0 and outputs['router_metrics']:
                self._log_router_metrics(outputs['router_metrics'], epoch, batch_idx)
        
        # 记录epoch统计
        avg_loss = total_loss / len(self.train_loader)
        avg_lm_loss = total_lm_loss / len(self.train_loader)
        avg_balance_loss = total_balance_loss / len(self.train_loader)
        
        self.train_stats['losses'].append(avg_loss)
        self.train_stats['lm_losses'].append(avg_lm_loss)
        self.train_stats['load_balancing_losses'].append(avg_balance_loss)
        
        return avg_loss, avg_lm_loss, avg_balance_loss
    
    def _log_router_metrics(self, router_metrics, epoch, batch_idx):
        """记录路由器统计信息"""
        print(f"\n📊 Epoch {epoch}, Batch {batch_idx} - 专家使用情况:")
        for metric in router_metrics:
            layer = metric['layer']
            expert_usage = metric['expert_usage']
            print(f"  层 {layer}: {expert_usage.cpu().detach().numpy().round(4)}")
    
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
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                total_loss += outputs['loss'].item()
                
                # 计算困惑度
                if outputs['lm_loss'] is not None:
                    perplexity = torch.exp(torch.tensor(outputs['lm_loss']))
                    total_perplexity += perplexity.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_perplexity = total_perplexity / len(self.val_loader)
        
        self.train_stats['perplexities'].append(avg_perplexity)
        
        return avg_loss, avg_perplexity
    
    def train(self):
        """完整训练流程"""
        best_val_loss = float('inf')
        
        print("🚀 开始训练MoE Adapter模型")
        
        for epoch in range(self.config.num_epochs):
            print(f"\n📍 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # 训练
            train_loss, train_lm_loss, train_balance_loss = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_perplexity = self.validate(epoch)
            
            print(f"📈 训练统计:")
            print(f"  - 总损失: {train_loss:.4f}")
            print(f"  - LM损失: {train_lm_loss:.4f}") 
            print(f"  - 负载均衡损失: {train_balance_loss:.4f}")
            print(f"  - 验证损失: {val_loss:.4f}")
            print(f"  - 验证困惑度: {val_perplexity:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f"best_model_epoch_{epoch}")
                print(f"💾 保存最佳模型 (验证损失: {val_loss:.4f})")
            
            # 保存训练统计
            self.save_training_stats()
    
    def save_model(self, path):
        """保存模型"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        save_path = os.path.join(self.config.output_dir, path)
        os.makedirs(save_path, exist_ok=True)
        
        # 只保存Adapter参数
        adapter_state_dict = {
            name: param for name, param in self.model.state_dict().items()
            if 'moe_adapters' in name
        }
        
        torch.save(adapter_state_dict, os.path.join(save_path, 'adapter_weights.pth'))
        
        # 保存配置
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    def save_training_stats(self):
        """保存训练统计信息"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(os.path.join(self.config.output_dir, 'training_stats.json'), 'w') as f:
            json.dump(self.train_stats, f, indent=2)