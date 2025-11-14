import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch
from transformers import GPT2Tokenizer

from config.base_config import GPT2MoEConfig, MoEAdapterConfig
from models.gpt2_moe_model import GPT2WithMoEAdapter

def generate_text():
    """使用训练好的模型生成文本"""
    
    # 加载配置
    adapter_config = MoEAdapterConfig(
        num_experts=8,
        expert_size=512,
        router_type="soft"
    )
    
    # 使用本地GPT-2权重路径
    local_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gpt2")
    
    model_config = GPT2MoEConfig(
        base_model=local_model_path,  # 使用本地路径
        num_adapter_layers=6,
        adapter_layers=[2, 4, 6, 8, 10, 12],
        freeze_base_model=True,
        adapter_config=adapter_config
    )
    
    # 创建模型和tokenizer
    model = GPT2WithMoEAdapter(model_config)
    tokenizer = GPT2Tokenizer.from_pretrained(local_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载训练好的适配器权重
    adapter_weights = torch.load("./output/best_model_epoch_0/adapter_weights.pth")
    model.load_state_dict(adapter_weights, strict=False)
    
    # 移动到设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # 生成文本
    prompt = "The future of artificial intelligence is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=100,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"🤖 生成的文本:")
    print(f"{generated_text}")

if __name__ == "__main__":
    generate_text()