import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.finetune_config import LoRAConfig
from config.training_config import TrainingConfig
from models.gpt2_lora_model import GPT2WithLoRA
from training.finetune_trainer import LoRATrainer
from training.data_loader import get_dataloaders

def test_lora_strategies():
    """测试不同的LoRA配置"""
    
    strategies = [
        # (rank, alpha, 描述)
        (8, 16.0, "小规模LoRA (rank=8)"),
        (16, 32.0, "中等规模LoRA (rank=16)"),
        (32, 64.0, "大规模LoRA (rank=32)"),
        (16, 16.0, "低alpha LoRA (rank=16, alpha=16)"),
        (16, 64.0, "高alpha LoRA (rank=16, alpha=64)"),
    ]
    
    for rank, alpha, description in strategies:
        print(f"\n{'='*60}")
        print(f"测试: {description}")
        print(f"{'='*60}")
        
        # LoRA配置
        lora_config = LoRAConfig(
            base_model="gpt2",
            freeze_base_model=True,
            adapter_rank=rank,
            adapter_alpha=alpha
        )
        
        # 创建模型
        model = GPT2WithLoRA(lora_config)
        
        # 训练配置
        training_config = TrainingConfig(
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            max_length=1024,
            learning_rate=1e-4,
            num_epochs=3,
            batch_size=4,
            output_dir=f"./lora_output_rank{rank}_alpha{alpha}"
        )
        
        # 数据加载器
        train_loader, val_loader, _ = get_dataloaders(training_config)
        
        # 训练器
        trainer = LoRATrainer(model, train_loader, val_loader, training_config)
        
        # 开始训练
        trainer.train()

def main():
    """主LoRA微调函数"""
    
    # LoRA配置 - 推荐配置
    lora_config = LoRAConfig(
        base_model="gpt2",
        freeze_base_model=True,  # 冻结基础模型
        adapter_rank=64,       # rank=16，平衡效果和参数量
        adapter_alpha=128.0      # alpha=32，增强表达能力
    )
    
    # 创建模型
    model = GPT2WithLoRA(lora_config)
    
    # 训练配置
    training_config = TrainingConfig(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_length=1024,
        learning_rate=2e-4,      # LoRA通常用较大的学习率
        num_epochs=50,
        batch_size=4,
        output_dir="./",
        use_fp16=True,
        fp16_opt_level="O1",
        patience=3,
        min_delta=0.001,
    )
    
    # 数据加载器
    train_loader, val_loader, test_loader, _ = get_dataloaders(training_config)
    
    # 训练器
    trainer = LoRATrainer(model, train_loader, val_loader, test_loader, training_config)
    
    # 开始LoRA微调训练
    trainer.train()

if __name__ == "__main__":
    main()