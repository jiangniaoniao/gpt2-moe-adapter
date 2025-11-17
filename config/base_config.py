from dataclasses import dataclass
from typing import List, Optional
import torch
import random
import numpy as np

@dataclass
class SmearAdapterConfig:
    """SMEAR适配器配置"""
    hidden_size: int = 768
    expert_size: int = 512
    num_experts: int = 8
    router_temperature: float = 1.0
    use_load_balancing: bool = False
    
    # 路由粒度配置
    routing_granularity: str = "causal_segment"  # "token", "segment", "causal_segment"
    segment_length: int = 256
    
    # 路由策略配置
    routing_strategy: str = "parameter_merging"  # "parameter_merging" 或 "top_k_sparse"
    top_k: int = 2  # Top-K值
    
    seed: int = 42
    
    def __post_init__(self):
        # 验证路由粒度
        valid_granularities = ["token", "segment", "causal_segment"]
        if self.routing_granularity not in valid_granularities:
            raise ValueError(f"路由粒度必须是: {valid_granularities}")
        
        # 验证路由策略
        valid_strategies = ["parameter_merging", "top_k_sparse"]
        if self.routing_strategy not in valid_strategies:
            raise ValueError(f"路由策略必须是: {valid_strategies}")
        
        if self.routing_strategy == "top_k_sparse" and self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) 不能大于专家数量 ({self.num_experts})")
        
        # 设置随机种子
        self._set_seed()
    
    def _set_seed(self):
        """设置随机种子保证可复现性"""
        seed = self.seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

@dataclass
class GPT2SmearConfig:
    """GPT-2 + SMEAR配置"""
    base_model: str = "gpt2"
    num_adapter_layers: int = 6
    adapter_layers: Optional[List[int]] = None
    freeze_base_model: bool = True
    smear_config: SmearAdapterConfig = None
    
    def __post_init__(self):
        if self.smear_config is None:
            self.smear_config = SmearAdapterConfig()
        
        if self.adapter_layers is None:
            total_layers = 12
            self.adapter_layers = list(range(min(self.num_adapter_layers, total_layers)))
        
        # 验证配置
        self.smear_config.__post_init__()