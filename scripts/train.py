import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.base_config import GPT2MoEConfig, MoEAdapterConfig
from models.gpt2_moe_model import GPT2WithMoEAdapter
from training.trainer import MoETrainer
from training.data_loader import create_data_loaders

def main():
    # 配置
    adapter_config = MoEAdapterConfig(
        num_experts=8,
        expert_size=512,
        router_type="topk",
        top_k=2,
        use_load_balancing=True
    )
    
    model_config = GPT2MoEConfig(
        base_model="home/yang/gpt2-moe-adapter/gpt2",
        num_adapters=6,
        freeze_base_model=True,
        adapter_config=adapter_config
    )
    
    # 训练配置
    training_config = type('TrainingConfig', (), {
        'learning_rate': 1e-4,
        'num_epochs': 10,
        'warmup_steps': 1000,
        'total_steps': 10000,
        'max_grad_norm': 1.0,
        'batch_size': 4,
        'accumulation_steps': 4
    })()
    
    # 创建模型
    model = GPT2WithMoEAdapter(model_config)
    
    # 数据加载器
    train_loader, val_loader = create_data_loaders(
        dataset_path="your_dataset_path",
        batch_size=training_config.batch_size,
        max_length=1024
    )
    
    # 训练器
    trainer = MoETrainer(model, train_loader, val_loader, training_config)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()