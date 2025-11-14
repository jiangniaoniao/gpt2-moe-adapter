import torch
import torch.nn as nn
import torch.nn.functional as F

class SmearExpert(nn.Module):
    """SMEAR风格的专家，支持参数软合并"""
    
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
    
    def forward(self, x, routing_weights):
        """
        参数:
            x: [batch_size, seq_len, hidden_size]
            routing_weights: [batch_size, num_experts] 路由权重
        返回:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # 确保路由权重正确形状
        if len(routing_weights.shape) == 1:
            routing_weights = routing_weights.unsqueeze(0)  # [1, num_experts]
        
        # 参数软合并 - 关键SMEAR操作
        # 计算加权平均的参数 [hidden_size, expert_size]
        merged_down_w = torch.einsum('be,ehd->hd', routing_weights, self.down_weights)
        merged_down_b = torch.einsum('be,ed->d', routing_weights, self.down_biases)
        merged_up_w = torch.einsum('be,edh->dh', routing_weights, self.up_weights)
        
        # 应用合并后的专家
        z = torch.matmul(x, merged_down_w) + merged_down_b
        z = self.activation(z)
        output = torch.matmul(z, merged_up_w)
        
        return output

class SmearRouter(nn.Module):
    """SMEAR路由器，生成专家权重"""
    
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
    
    def forward(self, hidden_states):
        """
        参数:
            hidden_states: [batch_size, seq_len, hidden_size] 或 [batch_size, hidden_size]
        返回:
            routing_weights: [batch_size, num_experts]
        """
        if len(hidden_states.shape) == 3:
            # 使用序列平均表示
            context_vector = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_size]
        else:
            context_vector = hidden_states
        
        router_logits = self.router_network(context_vector) / self.temperature
        routing_weights = F.softmax(router_logits, dim=-1)
        
        return routing_weights

class SmearAdapterLayer(nn.Module):
    """增强的SMEAR适配器层，支持真正的旁路"""
    
    def __init__(self, hidden_size, expert_size, num_experts, use_residual=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.expert_size = expert_size
        self.num_experts = num_experts
        self.use_residual = use_residual
        
        self.router = SmearRouter(hidden_size, num_experts)
        self.expert = SmearExpert(hidden_size, expert_size, num_experts)
        
        # 适配器内部的残差连接
        if self.use_residual:
            self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x):
        # 计算路由权重
        routing_weights = self.router(x)
        
        # 应用软合并专家
        adapter_output = self.expert(x, routing_weights)
        
        # 适配器内部的残差连接（可选）
        if self.use_residual:
            adapter_output = self.layer_norm(adapter_output + x)
        else:
            adapter_output = self.layer_norm(adapter_output)
        
        return adapter_output, routing_weights