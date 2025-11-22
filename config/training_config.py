from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """训练配置"""
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    max_length: int = 1024
    learning_rate: float = 1e-4
    num_epochs: int = 20
    batch_size: int = 4
    warmup_steps: int = 1000
    total_steps: int = 100000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    output_dir: str = "./smear_output"
    use_fp16: bool = True
    fp16_opt_level: str = "O1"
    patience: int = 3
    min_delta: float = 0.001