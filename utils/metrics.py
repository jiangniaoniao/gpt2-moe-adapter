import torch

def calculate_perplexity(loss):
    """计算困惑度"""
    return torch.exp(torch.tensor(loss))

def calculate_expert_utilization(router_metrics):
    """计算专家利用率"""
    if not router_metrics:
        return 0.0
    
    total_usage = 0
    count = 0
    
    for metric in router_metrics:
        expert_usage = metric['expert_usage']
        total_usage += expert_usage.sum().item()
        count += len(expert_usage)
    
    return total_usage / count if count > 0 else 0.0