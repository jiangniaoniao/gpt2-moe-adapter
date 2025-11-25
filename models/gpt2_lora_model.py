import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class LoRALayer(nn.Module):
    """LoRA适配器层"""
    
    def __init__(self, in_features, out_features, rank=16, alpha=32.0, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA权重矩阵 A 和 B
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 初始化权重
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        # LoRA前向传播: Wx + BAx * scaling
        lora_output = self.lora_B(self.dropout(self.lora_A(x)))
        return x + lora_output * self.scaling

class GPT2WithLoRA(nn.Module):
    """GPT-2 + LoRA微调模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.freeze_base_model = getattr(config, 'freeze_base_model', False)
        
        # 加载预训练GPT-2
        self.gpt2 = GPT2LMHeadModel.from_pretrained(config.base_model)
        
        # 冻结基础模型参数
        if self.freeze_base_model:
            print("  冻结GPT-2基础模型参数")
            for param in self.gpt2.parameters():
                param.requires_grad = False
        
        # 在注意力层添加LoRA适配器
        self.lora_layers = nn.ModuleList()
        
        # GPT-2的transformer层数量（对于gpt2是12层）
        num_layers = self.gpt2.config.num_hidden_layers
        
        for i in range(num_layers):
            # 为每层的qkv投影添加LoRA
            lora_q = LoRALayer(
                in_features=self.gpt2.config.hidden_size,
                out_features=self.gpt2.config.n_embd,
                rank=config.adapter_rank,
                alpha=config.adapter_alpha
            )
            lora_v = LoRALayer(
                in_features=self.gpt2.config.hidden_size,
                out_features=self.gpt2.config.n_embd,
                rank=config.adapter_rank,
                alpha=config.adapter_alpha
            )
            
            self.lora_layers.append(lora_q)
            self.lora_layers.append(lora_v)
        
        print(f"  初始化GPT-2 + LoRA模型")
        print(f"   - 基础模型: {config.base_model}")
        print(f"   - LoRA rank: {config.adapter_rank}")
        print(f"   - LoRA alpha: {config.adapter_alpha}")
        print(f"   - LoRA层数量: {len(self.lora_layers)}")
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.gpt2.parameters()) + trainable_params
        print(f"   - 可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"   - 总参数量: {total_params:,}")
    
    def apply_lora(self, hidden_states, layer_idx):
        """对特定层应用LoRA"""
        if layer_idx < len(self.lora_layers):
            # 简化：对每个隐藏状态应用相同的LoRA层
            # 实际中应该更精确地匹配Q、K、V投影
            lora_output = self.lora_layers[layer_idx](hidden_states)
            return hidden_states + lora_output
        return hidden_states
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """前向传播"""
        # 获取GPT-2的transformer输出
        outputs = self.gpt2.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # 对隐藏状态应用LoRA
        modified_hidden_states = []
        for i, hidden_state in enumerate(outputs.hidden_states):
            if i > 0:  # 跳过embedding层
                # 应用LoRA（每层一个）
                lora_idx = min(i - 1, len(self.lora_layers) - 1)
                modified_state = self.apply_lora(hidden_state, lora_idx)
                modified_hidden_states.append(modified_state)
            else:
                modified_hidden_states.append(hidden_state)
        
        # 最后的隐藏状态
        last_hidden_state = modified_hidden_states[-1]
        
        # 通过LM头计算logits
        lm_logits = self.gpt2.lm_head(last_hidden_state)
        
        # 计算损失
        total_loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            total_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {
            'loss': total_loss,
            'logits': lm_logits,
            'hidden_states': modified_hidden_states
        }
    
    def generate(self, input_ids, **kwargs):
        """生成文本 - 委托给GPT-2模型"""
        self.eval()
        return self.gpt2.generate(input_ids, **kwargs)
    
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """为生成准备输入 - 委托给GPT-2模型"""
        return self.gpt2.prepare_inputs_for_generation(input_ids, **kwargs)