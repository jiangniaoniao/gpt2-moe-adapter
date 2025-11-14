import torch
import torch.nn as nn
import torch.nn.functional as F

from models.router import Router

class Expert(nn.Module):
    """单个专家网络"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.expert_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.expert_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.expert_size, config.hidden_size),
            nn.Dropout(0.1)
        )
        
        # 初始化
        nn.init.normal_(self.expert_network[0].weight, std=config.expert_init_std)
        nn.init.normal_(self.expert_network[3].weight, std=config.expert_init_std)
        
    def forward(self, x):
        return self.expert_network(x)

class MoEAdapterLayer(nn.Module):
    """MoE Adapter层"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 专家池
        self.experts = nn.ModuleList([
            Expert(config) for _ in range(config.num_experts)
        ])
        
        # 路由器
        self.router = Router(config)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, hidden_states):
        """
        参数:
            hidden_states: [batch_size, seq_len, hidden_size]
        返回:
            output: [batch_size, seq_len, hidden_size]
            router_metrics: 路由器统计信息
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 计算路由权重 - 使用序列平均表示
        sequence_repr = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_size]
        routing_weights, router_logits = self.router(sequence_repr)  # [batch_size, num_experts]
        
        # 专家输出加权和
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(hidden_states)  # [batch_size, seq_len, hidden_size]
            weight = routing_weights[:, i].unsqueeze(-1).unsqueeze(-1)  # [batch_size, 1, 1]
            weighted_expert = weight * expert_out
            expert_outputs.append(weighted_expert)
        
        # 合并专家输出
        combined_output = sum(expert_outputs)
        
        # 应用层归一化
        output = self.layer_norm(combined_output)
        
        # 收集路由器统计信息
        router_metrics = {
            'router_logits': router_logits,
            'routing_weights': routing_weights,
            'expert_usage': routing_weights.mean(dim=0)  # 平均使用率
        }
        
        return output, router_metrics
    
    def load_balancing_loss(self, router_metrics):
        """计算负载均衡损失"""
        if not self.config.use_load_balancing:
            return torch.tensor(0.0, device=next(self.parameters()).device)
            
        routing_weights = router_metrics['routing_weights']  # [batch_size, num_experts]
        
        # 计算每个专家的使用频率
        expert_usage = routing_weights.mean(dim=0)  # [num_experts]
        
        # 计算负载均衡损失 (专家使用率的熵)
        load_balancing_loss = -torch.sum(expert_usage * torch.log(expert_usage + 1e-8))
        load_balancing_loss = load_balancing_loss * self.config.load_balancing_alpha
        
        return load_balancing_loss