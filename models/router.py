import torch
import torch.nn as nn
import torch.nn.functional as F

class Router(nn.Module):
    """MoE路由器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, config.num_experts)
        
        # 初始化
        nn.init.normal_(self.router.weight, std=0.02)
        
    def forward(self, hidden_states):
        """计算路由权重"""
        router_logits = self.router(hidden_states)
        
        if self.config.router_type == "soft":
            # 软路由 - 所有专家加权平均
            routing_weights = F.softmax(router_logits / self.config.temperature, dim=-1)
            return routing_weights, router_logits
            
        elif self.config.router_type == "topk":
            # Top-K 路由
            routing_weights = F.softmax(router_logits / self.config.temperature, dim=-1)
            top_k_weights, top_k_indices = torch.topk(routing_weights, self.config.top_k, dim=-1)
            
            # 重新归一化
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
            
            # 创建稀疏权重矩阵
            sparse_weights = torch.zeros_like(routing_weights)
            sparse_weights.scatter_(-1, top_k_indices, top_k_weights)
            
            return sparse_weights, router_logits
            
        else:
            raise ValueError(f"Unknown router type: {self.config.router_type}")