import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from .smear_adapter import SmearAdapterLayer

class GPT2WithSmearAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        smear_config = config.smear_config
        
        # 加载预训练GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(config.base_model)
        
        # 冻结基础模型
        if config.freeze_base_model:
            for param in self.gpt2.parameters():
                param.requires_grad = False
        
        # 创建SMEAR适配器层 - 作为旁路
        self.smear_adapters = nn.ModuleList([
            SmearAdapterLayer(
                hidden_size=smear_config.hidden_size,
                expert_size=smear_config.expert_size,
                num_experts=smear_config.num_experts
            ) for _ in range(len(config.adapter_layers))
        ])
        
        # 旁路缩放因子（可学习）
        self.adapter_alpha = nn.Parameter(torch.ones(len(config.adapter_layers)) * 0.1)
        
        print(f"✅ 初始化GPT-2 + SMEAR旁路适配器")
        print(f"   - 旁路适配器层: {config.adapter_layers}")
        print(f"   - 可训练参数: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # 获取GPT-2隐藏状态
        outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        hidden_states = outputs.hidden_states
        adapter_outputs = []
        routing_info = []
        
        current_adapter_idx = 0
        for layer_idx, hidden_state in enumerate(hidden_states):
            if layer_idx in self.config.adapter_layers and current_adapter_idx < len(self.smear_adapters):
                # 🎯 关键修改：旁路结构
                # 原始GPT-2输出 + 缩放后的适配器输出
                original_output = hidden_state
                adapter_output, routing_weights = self.smear_adapters[current_adapter_idx](hidden_state)
                
                # 应用旁路连接：output = original + alpha * adapter
                alpha = self.adapter_alpha[current_adapter_idx]
                combined_output = original_output + alpha * adapter_output
                
                adapter_outputs.append(combined_output)
                routing_info.append({
                    'layer': layer_idx,
                    'routing_weights': routing_weights,
                    'alpha': alpha
                })
                current_adapter_idx += 1
            else:
                adapter_outputs.append(hidden_state)
        
        # 最终隐藏状态
        last_hidden_state = adapter_outputs[-1]
        
        # 通过LM头计算logits
        lm_logits = self.gpt2.lm_head(last_hidden_state)
        
        # 计算损失
        total_loss = None
        lm_loss = None
        
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss = lm_loss
        
        return {
            'loss': total_loss,
            'logits': lm_logits,
            'lm_loss': lm_loss,
            'routing_info': routing_info,
            'hidden_states': adapter_outputs
        }