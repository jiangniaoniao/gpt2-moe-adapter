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
    use_load_balancing: bool = False  # SMEAR通常不需要显式的负载均衡损失
    seed: int = 42  # 随机种子

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
        
        # 设置随机种子
        self._set_seed()
    
    def _set_seed(self):
        """设置随机种子保证可复现性"""
        seed = self.smear_config.seed
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)