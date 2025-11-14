import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
import json

class SmearTrainer:
    """专门为SMEAR方法设计的训练器"""
    
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
            'perplexities': [],
            'routing_diversity': []  # SMEAR特有的路由多样性统计
        }
        
        print(f"🚀 初始化SMEAR训练器")
        print(f"   - 设备: {self.device}")
        print(f"   - 可训练参数: {sum(p.numel() for p in trainable_params):,}")
    
    def train_epoch(self, epoch):
        """训练一个epoch - SMEAR专用"""
        self.model.train()
        total_loss = 0
        total_lm_loss = 0
        total_routing_diversity = 0
        
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
            
            # 统计 - SMEAR专用
            total_loss += loss.item()
            
            lm_loss = outputs.get('lm_loss', None)
            if lm_loss is not None:
                total_lm_loss += lm_loss.item()
            
            # 计算路由多样性（SMEAR特有）
            routing_diversity = self._compute_routing_diversity(outputs.get('routing_info', []))
            total_routing_diversity += routing_diversity
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LM Loss': f'{lm_loss.item() if lm_loss is not None else 0:.4f}',
                'Routing Diversity': f'{routing_diversity:.4f}'
            })
            
            # 记录路由信息
            routing_info = outputs.get('routing_info', [])
            if batch_idx % 100 == 0 and routing_info:
                self._log_smear_routing_info(routing_info, epoch, batch_idx)
        
        # 记录epoch统计
        avg_loss = total_loss / len(self.train_loader)
        avg_lm_loss = total_lm_loss / len(self.train_loader) if total_lm_loss > 0 else 0
        avg_routing_diversity = total_routing_diversity / len(self.train_loader)
        
        self.train_stats['losses'].append(avg_loss)
        self.train_stats['lm_losses'].append(avg_lm_loss)
        self.train_stats['routing_diversity'].append(avg_routing_diversity)
        
        return avg_loss, avg_lm_loss, avg_routing_diversity
    
    def _compute_routing_diversity(self, routing_info):
        """计算SMEAR路由多样性（专家权重分布的熵）"""
        if not routing_info:
            return 0.0
        
        total_diversity = 0.0
        count = 0
        
        for info in routing_info:
            if 'routing_weights' in info:
                routing_weights = info['routing_weights']  # [batch_size, num_experts]
                
                # 计算平均路由权重
                avg_weights = torch.mean(routing_weights, dim=0)
                
                # 计算熵作为多样性指标
                entropy = -torch.sum(avg_weights * torch.log(avg_weights + 1e-8))
                total_diversity += entropy.item()
                count += 1
        
        return total_diversity / count if count > 0 else 0.0
    
    def _log_smear_routing_info(self, routing_info, epoch, batch_idx):
        """记录SMEAR路由信息"""
        print(f"\n📊 Epoch {epoch}, Batch {batch_idx} - SMEAR路由信息:")
        
        for info in routing_info:
            layer = info.get('layer', 'unknown')
            
            if 'routing_weights' in info:
                routing_weights = info['routing_weights']
                avg_weights = torch.mean(routing_weights, dim=0)
                weights_str = avg_weights.cpu().detach().numpy().round(4)
                
                # 计算每个专家的使用强度
                expert_strength = torch.mean(routing_weights, dim=0)
                dominant_expert = torch.argmax(expert_strength).item()
                
                print(f"  层 {layer}:")
                print(f"    - 路由权重: {weights_str}")
                print(f"    - 主导专家: {dominant_expert} (权重: {expert_strength[dominant_expert]:.4f})")
    
    def validate(self, epoch):
        """验证 - SMEAR专用"""
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
                lm_loss = outputs.get('lm_loss', None)
                if lm_loss is not None:
                    perplexity = torch.exp(torch.tensor(lm_loss.item()))
                    total_perplexity += perplexity.item()
        
        avg_loss = total_loss / len(self.val_loader)
        avg_perplexity = total_perplexity / len(self.val_loader) if total_perplexity > 0 else float('inf')
        
        self.train_stats['perplexities'].append(avg_perplexity)
        
        return avg_loss, avg_perplexity
    
    def train(self):
        """完整训练流程"""
        best_val_loss = float('inf')
        
        print("🚀 开始训练SMEAR适配器模型")
        
        for epoch in range(self.config.num_epochs):
            print(f"\n📍 Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # 训练
            train_loss, train_lm_loss, train_routing_diversity = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_perplexity = self.validate(epoch)
            
            print(f"📈 SMEAR训练统计:")
            print(f"  - 总损失: {train_loss:.4f}")
            print(f"  - LM损失: {train_lm_loss:.4f}") 
            print(f"  - 路由多样性: {train_routing_diversity:.4f}")
            print(f"  - 验证损失: {val_loss:.4f}")
            print(f"  - 验证困惑度: {val_perplexity:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(f"best_smear_model_epoch_{epoch}")
                print(f"💾 保存最佳SMEAR模型 (验证损失: {val_loss:.4f})")
            
            # 保存训练统计
            self.save_training_stats()
    
    def save_model(self, path):
        """保存SMEAR模型"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        save_path = os.path.join(self.config.output_dir, path)
        os.makedirs(save_path, exist_ok=True)
        
        # 保存SMEAR适配器参数
        model_state_dict = self.model.state_dict()
        smear_state_dict = {
            name: param for name, param in model_state_dict.items()
            if any(key in name for key in ['smear_adapters', 'adapters'])
        }
        
        # 如果没有找到，保存所有可训练参数
        if not smear_state_dict:
            smear_state_dict = {
                name: param for name, param in model_state_dict.items()
                if param.requires_grad
            }
        
        torch.save(smear_state_dict, os.path.join(save_path, 'smear_weights.pth'))
        
        # 保存配置
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        print(f"💾 保存了 {len(smear_state_dict)} 个SMEAR适配器参数")
    
    def save_training_stats(self):
        """保存训练统计信息"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        with open(os.path.join(self.config.output_dir, 'smear_training_stats.json'), 'w') as f:
            json.dump(self.train_stats, f, indent=2)