import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config
import os

from .moe_adapter import MoEAdapterLayer

class GPT2WithMoEAdapter(nn.Module):
    """集成MoE Adapter的GPT-2模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        adapter_config = config.adapter_config
        
        # 检查是否是本地路径
        if os.path.exists(config.base_model):
            # 加载本地GPT-2模型
            print(f"📁 从本地路径加载模型: {config.base_model}")
            self.gpt2 = GPT2LMHeadModel.from_pretrained(config.base_model)
        else:
            # 从HuggingFace加载模型
            print(f"🌐 从HuggingFace加载模型: {config.base_model}")
            self.gpt2 = GPT2LMHeadModel.from_pretrained(config.base_model)
        
        gpt2_config = self.gpt2.config
        
        # 冻结基础模型参数
        if config.freeze_base_model:
            for param in self.gpt2.parameters():
                param.requires_grad = False
        
        # 创建MoE Adapter层
        self.moe_adapters = nn.ModuleList([
            MoEAdapterLayer(adapter_config) for _ in range(len(config.adapter_layers))
        ])
        
        print(f"✅ 初始化GPT-2 + MoE Adapter模型")
        print(f"   - 基础模型: {config.base_model}")
        print(f"   - 适配器层: {config.adapter_layers}")
        print(f"   - 专家数量: {adapter_config.num_experts}")
        print(f"   - 路由器类型: {adapter_config.router_type}")
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        参数:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len]
        返回:
            dict: 包含损失、logits和路由器统计信息
        """
        # GPT-2前向传播，获取所有隐藏状态
        outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.hidden_states  # 所有层的隐藏状态
        
        # 应用MoE Adapter
        adapter_outputs = []
        total_load_loss = torch.tensor(0.0, device=input_ids.device)
        router_metrics_list = []
        
        current_adapter_idx = 0
        for layer_idx, hidden_state in enumerate(hidden_states):
            if layer_idx in self.config.adapter_layers and current_adapter_idx < len(self.moe_adapters):
                # 应用MoE Adapter
                adapted_output, router_metrics = self.moe_adapters[current_adapter_idx](hidden_state)
                adapter_outputs.append(adapted_output)
                
                # 累加负载均衡损失
                load_loss = self.moe_adapters[current_adapter_idx].load_balancing_loss(router_metrics)
                total_load_loss = total_load_loss + load_loss
                
                # 收集路由器统计信息
                router_metrics_list.append({
                    'layer': layer_idx,
                    'expert_usage': router_metrics['expert_usage']
                })
                
                current_adapter_idx += 1
            else:
                adapter_outputs.append(hidden_state)
        
        # 使用最后一个隐藏状态
        last_hidden_state = adapter_outputs[-1]
        
        # 通过GPT-2的LM头计算logits
        lm_logits = self.gpt2.lm_head(last_hidden_state)
        
        # 计算损失
        total_loss = None
        lm_loss = None
        
        if labels is not None:
            # 移位logits和labels用于语言建模
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss = lm_loss + total_load_loss
        
        return {
            'loss': total_loss,
            'logits': lm_logits,
            'lm_loss': lm_loss,
            'load_balancing_loss': total_load_loss,
            'hidden_states': adapter_outputs,
            'router_metrics': router_metrics_list
        }
    
    def generate(self, input_ids, **kwargs):
        """生成文本 - 直接使用基础GPT-2的生成方法"""
        return self.gpt2.generate(input_ids, **kwargs)