from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据配置
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"  # 或 "wikitext-103-raw-v1"
    max_length: int = 1024
    
    # 训练配置
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # 学习率调度
    warmup_steps: int = 1000
    total_steps: int = 10000
    
    # 保存配置
    output_dir: str = "./output"
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100