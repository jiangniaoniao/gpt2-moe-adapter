import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.base_config import GPT2SmearConfig, SmearAdapterConfig
from config.training_config import TrainingConfig
from models.gpt2_smear_model import GPT2WithSmearAdapter
from training.data_loader import get_wikitext_dataloaders
from training.smear_trainer import SmearTrainer  # 使用专门的SMEAR训练器


def main():
    # SMEAR配置
    smear_config = SmearAdapterConfig(
        num_experts=8,
        expert_size=512,
        router_temperature=1.0
    )
    
    # 模型配置
    model_config = GPT2SmearConfig(
        base_model="/home/yang/gpt2-moe-adapter/gpt2",
        num_adapter_layers=6,
        adapter_layers=[2, 4, 6, 8, 10, 12],
        freeze_base_model=True,
        smear_config=smear_config
    )
    
    # 训练配置
    training_config = TrainingConfig(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_length=1024,
        learning_rate=1e-4,
        num_epochs=10,
        batch_size=4,
        output_dir="./smear_output"
    )
    
    # 创建模型
    model = GPT2WithSmearAdapter(model_config)
    
    # 数据加载器
    train_loader, val_loader, _ = get_wikitext_dataloaders(training_config)
    
    # 使用专门的SMEAR训练器
    trainer = SmearTrainer(model, train_loader, val_loader, training_config)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()