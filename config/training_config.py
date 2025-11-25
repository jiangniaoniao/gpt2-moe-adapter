from dataclasses import dataclass, field

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
    dataset_mode: str= "mixed"
    target_total_samples = 50000 
    dataset_mix: list[tuple[str, str, float, str]] = field(
        default_factory=lambda: [
            # 基础指令数据
            ("tatsu-lab/alpaca", None, 0.3, "instruction"),
            ("databricks/databricks-dolly-15k", None, 0.1, "instruction"),
            
            # 学科知识数据  
            ("cais/mmlu", "all", 0.15, "knowledge"),
            ("allenai/ai2_arc", "ARC-Challenge", 0.15, "knowledge"),
            ("derek-thomas/ScienceQA", None, 0.1, "knowledge"),
            
            # 推理数据
            ("gsm8k", "main", 0.1, "reasoning"),
            ("tau/commonsense_qa", None, 0.1, "reasoning"),
            
            # WikiText基础语言建模
            ("wikitext", "wikitext-2-raw-v1", 0.1, "lm")
        ]
    )