import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config.base_config import GPT2SmearConfig, SmearAdapterConfig
from config.training_config import TrainingConfig
from models.gpt2_smear_model import GPT2WithSmearAdapter
from training.smear_trainer import SmearTrainer, IntegratedSmearTrainer
from training.data_loader import get_dataloaders

def test_routing_strategies():
    """测试不同的路由策略组合"""
    
    strategies = [
        # (路由粒度, 路由策略, top_k, 描述)
        ("token", "parameter_merging", None, "Token级参数聚合"),
        ("token", "top_k_sparse", 2, "Token级Top-2稀疏"),
        ("segment", "parameter_merging", None, "Segment级参数聚合"),
        ("segment", "top_k_sparse", 2, "Segment级Top-2稀疏"),
        ("causal_segment", "parameter_merging", None, "因果分段参数聚合"),
        ("causal_segment", "top_k_sparse", 2, "因果分段Top-2稀疏"),
    ]
    
    for routing_granularity, routing_strategy, top_k, description in strategies:
        print(f"\n{'='*60}")
        print(f"测试: {description}")
        print(f"{'='*60}")
        
        # SMEAR配置
        smear_config = SmearAdapterConfig(
            num_experts=8,
            expert_size=512,
            router_temperature=1.0,
            routing_granularity=routing_granularity,
            segment_length=256,
            routing_strategy=routing_strategy,
            top_k=top_k if top_k else 2
        )
        
        # 模型配置
        model_config = GPT2SmearConfig(
            base_model="gpt2",
            num_adapter_layers=3,
            adapter_layers=[2, 4, 6],
            freeze_base_model=True,
            smear_config=smear_config
        )
        
        # 训练配置
        training_config = TrainingConfig(
            dataset_name="wikitext",
            dataset_config="wikitext-2-raw-v1",
            max_length=1024,
            learning_rate=1e-4,
            num_epochs=3,
            batch_size=4,
            output_dir=f"./smear_output_{description.replace(' ', '_')}"
        )
        
        # 创建模型
        model = GPT2WithSmearAdapter(model_config)
        
        # 数据加载器
        train_loader, val_loader, _ = get_dataloaders(training_config)
        
        # 训练器
        trainer = SmearTrainer(model, train_loader, val_loader, training_config)
        
        # 开始训练
        trainer.train()

def main():
    """主训练函数"""
    
    # SMEAR配置
    smear_config = SmearAdapterConfig(
        num_experts=8,
        expert_size=512,
        router_temperature=1.0,
        routing_granularity="causal_segment",  # 因果分段路由
        segment_length=256,
        routing_strategy="top_k_sparse",       # Top-K稀疏激活
        top_k=2                                # Top-2
    )
    
    # 模型配置
    model_config = GPT2SmearConfig(
        base_model="gpt2",
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
        num_epochs=0,
        batch_size=16,
        output_dir="./smear_output_recommended",
        use_fp16=True,
        fp16_opt_level="O1",
        patience=3,
        min_delta=0.001,
    )
    
    # 创建模型
    model = GPT2WithSmearAdapter(model_config)
    
    # 数据加载器
    train_loader, val_loader, test_loader, _ = get_dataloaders(training_config)
    
    # 训练器
    trainer = IntegratedSmearTrainer(model, train_loader, val_loader, test_loader, training_config)
    
    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()