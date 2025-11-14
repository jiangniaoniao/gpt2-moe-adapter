from dataclasses import dataclass
from typing import List, Optional

@dataclass
class MoEAdapterConfig:
    """MoE Adapter配置"""
    # 基础配置
    hidden_size: int = 768
    num_experts: int = 8
    expert_size: int = 512
    reduction_factor: int = 4
    
    # 路由器配置
    router_type: str = "soft"  # "soft", "topk"
    top_k: int = 2
    temperature: float = 1.0
    
    # 负载均衡
    use_load_balancing: bool = True
    load_balancing_alpha: float = 0.01
    
    # 初始化配置
    expert_init_std: float = 0.02

@dataclass
class GPT2MoEConfig:
    """GPT-2 MoE模型配置"""
    base_model: str = "gpt2"
    num_adapter_layers: int = 6  # 插入适配器的层数
    adapter_layers: Optional[List[int]] = None  # 指定插入层索引
    freeze_base_model: bool = True
    adapter_config: MoEAdapterConfig = None
    
    def __post_init__(self):
        if self.adapter_config is None:
            self.adapter_config = MoEAdapterConfig()
        
        # 如果没有指定具体层，使用前N层
        if self.adapter_layers is None:
            total_layers = 12  # GPT-2 small有12层
            self.adapter_layers = list(range(min(self.num_adapter_layers, total_layers)))

@dataclass
class SmearAdapterConfig:
    """SMEAR适配器配置"""
    hidden_size: int = 768
    expert_size: int = 512
    num_experts: int = 8
    router_temperature: float = 1.0
    use_load_balancing: bool = False  # SMEAR通常不需要显式的负载均衡损失

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