from dataclasses import dataclass

@dataclass
class LoRAConfig:
    """LoRA微调配置"""
    base_model: str = "gpt2"
    freeze_base_model: bool = True  # 冻结基础模型参数
    adapter_rank: int = 16  # LoRA rank
    adapter_alpha: float = 32.0  # LoRA alpha