import torch
import torch.nn as nn

def compute_moe_loss(lm_loss, load_balancing_loss, lm_loss_weight=1.0, balance_loss_weight=1.0):
    """计算MoE总损失"""
    if lm_loss is None:
        return load_balancing_loss
    
    total_loss = lm_loss_weight * lm_loss + balance_loss_weight * load_balancing_loss
    return total_loss

def compute_perplexity(lm_loss):
    """计算困惑度"""
    if lm_loss is None:
        return None
    return torch.exp(lm_loss)