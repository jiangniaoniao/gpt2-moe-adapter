import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class SmearExpert(nn.Module):
    """SMEAR专家，支持参数聚合和Top-K稀疏激活两种策略"""
    
    def __init__(self, hidden_size, expert_size, num_experts):
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_size = expert_size
        self.num_experts = num_experts
        
        # 专家参数池 [num_experts, hidden_size, expert_size]
        self.down_weights = nn.Parameter(torch.randn(num_experts, hidden_size, expert_size))
        self.down_biases = nn.Parameter(torch.randn(num_experts, expert_size))
        self.up_weights = nn.Parameter(torch.randn(num_experts, expert_size, hidden_size))
        
        # 初始化
        nn.init.xavier_uniform_(self.down_weights)
        nn.init.xavier_uniform_(self.up_weights)
        nn.init.zeros_(self.down_biases)
        
        self.activation = nn.GELU()
    
    def forward_parameter_merging(self, x, routing_weights):
        """
        参数聚合策略：在参数空间加权平均
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # 处理不同的路由权重形状
        if len(routing_weights.shape) == 3:
            # 每token路由 [batch_size, seq_len, num_experts]
            if routing_weights.shape[1] == seq_len:
                routing_weights_flat = routing_weights.reshape(-1, self.num_experts)
                x_flat = x.reshape(-1, hidden_size)
                
                # 参数软合并
                merged_down_w = torch.einsum('be,ehd->hd', routing_weights_flat, self.down_weights)
                merged_down_b = torch.einsum('be,ed->d', routing_weights_flat, self.down_biases)
                merged_up_w = torch.einsum('be,edh->dh', routing_weights_flat, self.up_weights)
                
                # 应用合并后的专家
                z = torch.matmul(x_flat, merged_down_w) + merged_down_b
                z = self.activation(z)
                output_flat = torch.matmul(z, merged_up_w)
                
                output = output_flat.reshape(batch_size, seq_len, hidden_size)
                return output
            else:
                raise ValueError(f"路由权重形状不匹配: {routing_weights.shape} vs 输入 {x.shape}")
        
        else:
            # 每段/序列路由 [batch_size, num_experts]
            if len(routing_weights.shape) == 1:
                routing_weights = routing_weights.unsqueeze(0)
            
            # 参数软合并
            merged_down_w = torch.einsum('be,ehd->hd', routing_weights, self.down_weights)
            merged_down_b = torch.einsum('be,ed->d', routing_weights, self.down_biases)
            merged_up_w = torch.einsum('be,edh->dh', routing_weights, self.up_weights)
            
            # 应用专家
            z = torch.matmul(x, merged_down_w) + merged_down_b
            z = self.activation(z)
            output = torch.matmul(z, merged_up_w)
            
            return output
    
    def forward_top_k_sparse(self, x, routing_weights, top_k=2):
        """
        Top-K稀疏激活策略：在输出空间加权求和
        - 修复：确保每个位置严格只激活top_k个专家
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # 应用Top-K选择
        if len(routing_weights.shape) == 3:
            # 每token路由 [batch_size, seq_len, num_experts]
            # 获取Top-K权重和索引
            topk_weights, topk_indices = torch.topk(routing_weights, top_k, dim=-1)
            
            # 创建严格稀疏的权重矩阵 - 只保留Top-K，其他清零
            sparse_weights = torch.zeros_like(routing_weights)
            sparse_weights.scatter_(-1, topk_indices, topk_weights)
            
            # 重新归一化 - 确保每个位置的权重和为1
            sparse_weights_sum = sparse_weights.sum(dim=-1, keepdim=True) + 1e-8
            sparse_weights = sparse_weights / sparse_weights_sum
            
            # 验证稀疏性 - 现在应该严格满足
            non_zero_count = (sparse_weights > 1e-8).sum(dim=-1)
            if not torch.all(non_zero_count <= top_k):
                print(f"警告: 发现位置激活专家数超过{top_k}: {non_zero_count.max().item()}")
                # 强制修正：重新创建稀疏权重
                sparse_weights = torch.zeros_like(routing_weights)
                sparse_weights.scatter_(-1, topk_indices, 1.0)  # 使用1.0而不是原始权重
                sparse_weights_sum = sparse_weights.sum(dim=-1, keepdim=True) + 1e-8
                sparse_weights = sparse_weights / sparse_weights_sum
            
            # 优化实现：避免显存爆炸
            output = torch.zeros_like(x)
            
            # 重塑输入以便处理
            x_flat = x.reshape(-1, hidden_size)  # [batch_size * seq_len, hidden_size]
            sparse_weights_flat = sparse_weights.reshape(-1, self.num_experts)  # [batch_size * seq_len, num_experts]
            
            # 逐专家计算
            for expert_idx in range(self.num_experts):
                # 获取当前专家的权重 [batch_size * seq_len]
                expert_weights_flat = sparse_weights_flat[:, expert_idx]
                
                # 只处理有激活的位置
                active_mask = expert_weights_flat > 1e-8
                if not active_mask.any():
                    continue
                    
                # 获取激活位置的输入和权重
                active_x = x_flat[active_mask]  # [num_active, hidden_size]
                active_weights = expert_weights_flat[active_mask]  # [num_active]
                
                # 计算当前专家的输出
                z = torch.matmul(active_x, self.down_weights[expert_idx]) + self.down_biases[expert_idx]
                z = self.activation(z)
                expert_output_flat = torch.matmul(z, self.up_weights[expert_idx])  # [num_active, hidden_size]
                
                # 加权累加到输出
                output_flat = output.reshape(-1, hidden_size)
                output_flat[active_mask] += expert_output_flat * active_weights.unsqueeze(1)
                output = output_flat.reshape(batch_size, seq_len, hidden_size)
                
        else:
            # 每段/序列路由 [batch_size, num_experts]
            if len(routing_weights.shape) == 1:
                routing_weights = routing_weights.unsqueeze(0)
            
            # 应用Top-K选择
            topk_weights, topk_indices = torch.topk(routing_weights, top_k, dim=-1)
            
            # 创建严格稀疏的权重矩阵
            sparse_weights = torch.zeros_like(routing_weights)
            sparse_weights.scatter_(-1, topk_indices, topk_weights)
            
            # 重新归一化
            sparse_weights_sum = sparse_weights.sum(dim=-1, keepdim=True) + 1e-8
            sparse_weights = sparse_weights / sparse_weights_sum
            
            # 验证稀疏性
            non_zero_count = (sparse_weights > 1e-8).sum(dim=-1)
            if not torch.all(non_zero_count <= top_k):
                print(f"警告: 发现样本激活专家数超过{top_k}: {non_zero_count.max().item()}")
                # 强制修正
                sparse_weights = torch.zeros_like(routing_weights)
                sparse_weights.scatter_(-1, topk_indices, 1.0)
                sparse_weights_sum = sparse_weights.sum(dim=-1, keepdim=True) + 1e-8
                sparse_weights = sparse_weights / sparse_weights_sum
            
            # 计算所有专家的输出
            expert_outputs = []
            for i in range(self.num_experts):
                z = torch.matmul(x, self.down_weights[i]) + self.down_biases[i]
                z = self.activation(z)
                expert_out = torch.matmul(z, self.up_weights[i])  # [batch_size, seq_len, hidden_size]
                expert_outputs.append(expert_out)
            
            # 堆叠专家输出 [batch_size, seq_len, hidden_size, num_experts]
            expert_outputs = torch.stack(expert_outputs, dim=-1)
            
            # 在输出空间加权求和
            sparse_weights_expanded = sparse_weights.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, num_experts]
            output = torch.sum(expert_outputs * sparse_weights_expanded, dim=-1)
        
        return output, sparse_weights
    
    def forward(self, x, routing_weights, routing_strategy="parameter_merging", top_k=2):
        """
        统一前向传播接口
        """
        if routing_strategy == "parameter_merging":
            output = self.forward_parameter_merging(x, routing_weights)
            return output, routing_weights  # 返回原始路由权重
        elif routing_strategy == "top_k_sparse":
            output, sparse_weights = self.forward_top_k_sparse(x, routing_weights, top_k)
            return output, sparse_weights  # 返回稀疏化后的路由权重
        else:
            raise ValueError(f"未知的路由策略: {routing_strategy}")

class SmearRouter(nn.Module):
    """SMEAR路由器，支持多种路由粒度"""
    
    def __init__(self, hidden_size, num_experts, temperature=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.temperature = temperature
        
        self.router_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, num_experts)
        )
    
    def forward(self, hidden_states, routing_granularity="sequence"):
        """
        根据路由粒度生成路由权重
        """
        if routing_granularity == "token":
            # 每token路由: [batch_size, seq_len, num_experts]
            router_logits = self.router_network(hidden_states) / self.temperature
            routing_weights = F.softmax(router_logits, dim=-1)
            return routing_weights
        else:
            # 每段/序列路由: [batch_size, num_experts]
            if len(hidden_states.shape) == 3:
                context_vector = torch.mean(hidden_states, dim=1)
            else:
                context_vector = hidden_states
                
            router_logits = self.router_network(context_vector) / self.temperature
            routing_weights = F.softmax(router_logits, dim=-1)
            return routing_weights

class CausalSegmentRouting(nn.Module):
    """因果分段路由策略（Lory论文方法）"""
    
    def __init__(self, segment_length: int = 256, num_experts: int = 8):
        super().__init__()
        self.segment_length = segment_length
        self.num_experts = num_experts
    
    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        实现因果分段路由
        """
        batch_size, seq_len, hidden_size = x.shape
        num_segments = seq_len // self.segment_length
        
        if num_segments == 0:
            return x, routing_weights.unsqueeze(1)
        
        # 创建因果路由权重
        initial_routing = routing_weights.unsqueeze(1)  # [batch_size, 1, num_experts]
        expanded_routing = initial_routing.expand(batch_size, num_segments, self.num_experts)
        
        # 因果移位: 使用前一个段的路由权重
        causal_routing_weights = torch.roll(expanded_routing, shifts=1, dims=1)
        causal_routing_weights[:, 0, :] = initial_routing.squeeze(1)  # 第一个段用初始路由
        
        # 对第一个段应用stop gradient
        causal_routing_weights = causal_routing_weights.clone()
        if self.training:
            causal_routing_weights[:, 0, :] = causal_routing_weights[:, 0, :].detach()
        
        return x, causal_routing_weights

class SmearAdapterLayer(nn.Module):
    """SMEAR适配器层，支持多种路由粒度和策略"""
    
    def __init__(self, hidden_size, expert_size, num_experts, 
                 routing_granularity="causal_segment", segment_length=256, 
                 use_residual=True, routing_strategy="parameter_merging", top_k=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_size = expert_size
        self.num_experts = num_experts
        self.routing_granularity = routing_granularity
        self.segment_length = segment_length
        self.use_residual = use_residual
        self.routing_strategy = routing_strategy
        self.top_k = top_k
        
        self.router = SmearRouter(hidden_size, num_experts)
        self.expert = SmearExpert(hidden_size, expert_size, num_experts)
        
        if routing_granularity == "causal_segment":
            self.causal_routing = CausalSegmentRouting(segment_length, num_experts)
        
        if self.use_residual:
            self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        if self.routing_granularity == "token":
            # 每token路由
            routing_weights = self.router(x, routing_granularity="token")
            adapter_output, final_routing_weights = self.expert(
                x, routing_weights, 
                routing_strategy=self.routing_strategy,
                top_k=self.top_k
            )
            
        elif self.routing_granularity == "segment":
            # 普通分段路由
            num_segments = (seq_len + self.segment_length - 1) // self.segment_length
            adapter_output = torch.zeros_like(x)
            all_routing_weights = []
            
            for seg_idx in range(num_segments):
                start_idx = seg_idx * self.segment_length
                end_idx = min((seg_idx + 1) * self.segment_length, seq_len)
                
                segment = x[:, start_idx:end_idx, :]
                routing_weights = self.router(segment, routing_granularity="segment")
                segment_output, seg_routing_weights = self.expert(
                    segment, routing_weights,
                    routing_strategy=self.routing_strategy,
                    top_k=self.top_k
                )
                
                adapter_output[:, start_idx:end_idx, :] = segment_output
                all_routing_weights.append(seg_routing_weights)
            
            # 合并分段路由权重
            if all_routing_weights:
                final_routing_weights = torch.stack(all_routing_weights, dim=1)  # [batch_size, num_segments, num_experts]
            else:
                final_routing_weights = routing_weights
            
        elif self.routing_granularity == "causal_segment":
            # 因果分段路由
            initial_routing_weights = self.router(x, routing_granularity="segment")
            _, causal_routing_weights = self.causal_routing(x, initial_routing_weights)
            
            num_segments = causal_routing_weights.shape[1]
            adapter_output = torch.zeros_like(x)
            all_routing_weights = []
            
            for seg_idx in range(num_segments):
                start_idx = seg_idx * self.segment_length
                end_idx = min((seg_idx + 1) * self.segment_length, seq_len)
                
                if start_idx >= seq_len:
                    break
                    
                segment = x[:, start_idx:end_idx, :]
                seg_routing = causal_routing_weights[:, seg_idx, :]  # [batch_size, num_experts]
                segment_output, final_seg_routing = self.expert(
                    segment, seg_routing,
                    routing_strategy=self.routing_strategy,
                    top_k=self.top_k
                )
                
                adapter_output[:, start_idx:end_idx, :] = segment_output
                all_routing_weights.append(final_seg_routing)
            
            # 合并分段路由权重
            if all_routing_weights:
                final_routing_weights = torch.stack(all_routing_weights, dim=1)  # [batch_size, num_segments, num_experts]
            else:
                final_routing_weights = causal_routing_weights
            
        else:
            raise ValueError(f"未知的路由粒度: {self.routing_granularity}")
        
        # 残差连接
        if self.use_residual:
            adapter_output = self.layer_norm(adapter_output + x)
        else:
            adapter_output = self.layer_norm(adapter_output)
        
        return adapter_output, final_routing_weights
    
    def get_expert_utilization(self, routing_weights):
        """计算专家利用率 - 修复：更准确的计算"""
        if len(routing_weights.shape) == 3:
            # [batch_size, num_segments, num_experts]
            # 计算每个段（每个路由决策）的平均活跃专家数
            active_mask = (routing_weights > 1e-8).float()
            avg_active_per_segment = active_mask.sum(dim=-1).mean()  # 每个段平均活跃专家数
            utilization = avg_active_per_segment / self.num_experts
            return utilization
        else:
            # [batch_size, num_experts]
            active_mask = (routing_weights > 1e-8).float()
            avg_active_per_sample = active_mask.sum(dim=-1).mean()  # 每个样本平均活跃专家数
            utilization = avg_active_per_sample / self.num_experts
            return utilization