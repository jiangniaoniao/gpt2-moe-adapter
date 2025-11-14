import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from transformers import GPT2Tokenizer

from config.base_config import GPT2MoEConfig, MoEAdapterConfig
from models.gpt2_moe_model import GPT2WithMoEAdapter
from training.data_loader import get_wikitext_dataloaders

def evaluate_model():
    """评估训练好的模型"""
    
    # 加载配置
    adapter_config = MoEAdapterConfig(
        num_experts=8,
        expert_size=512,
        router_type="soft"
    )
    
    model_config = GPT2MoEConfig(
        base_model="gpt2",
        num_adapter_layers=6,
        adapter_layers=[2, 4, 6, 8, 10, 12],
        freeze_base_model=True,
        adapter_config=adapter_config
    )
    
    # 创建模型
    model = GPT2WithMoEAdapter(model_config)
    
    # 加载训练好的适配器权重
    adapter_weights = torch.load("./output/best_model_epoch_0/adapter_weights.pth")
    model.load_state_dict(adapter_weights, strict=False)
    
    # 移动到设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 获取数据加载器
    training_config = type('Config', (), {
        'dataset_name': 'wikitext',
        'dataset_config': 'wikitext-2-raw-v1',
        'max_length': 1024,
        'batch_size': 4
    })()
    
    _, val_loader, _ = get_wikitext_dataloaders(training_config)
    
    # 评估
    total_loss = 0
    total_perplexity = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs['loss']
            
            total_loss += loss.item()
            perplexity = torch.exp(torch.tensor(loss.item()))
            total_perplexity += perplexity.item()
    
    avg_loss = total_loss / len(val_loader)
    avg_perplexity = total_perplexity / len(val_loader)
    
    print(f"📊 评估结果:")
    print(f"  - 平均损失: {avg_loss:.4f}")
    print(f"  - 平均困惑度: {avg_perplexity:.4f}")

if __name__ == "__main__":
    evaluate_model()