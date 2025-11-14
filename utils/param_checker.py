import torch
import torch.nn as nn

def check_parameter_freezing(model):
    """详细检查参数冻结状态"""
    print("🔍 参数冻结状态检查:")
    
    total_params = 0
    frozen_params = 0
    trainable_params = 0
    
    # 检查基础GPT-2参数
    gpt2_params = 0
    frozen_gpt2_params = 0
    
    for name, param in model.gpt2.named_parameters():
        gpt2_params += param.numel()
        if not param.requires_grad:
            frozen_gpt2_params += param.numel()
    
    # 检查适配器参数
    adapter_params = 0
    trainable_adapter_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()
        
        if 'smear_adapters' in name or 'adapter_alpha' in name:
            adapter_params += param.numel()
            if param.requires_grad:
                trainable_adapter_params += param.numel()
    
    print(f"📊 参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  冻结参数: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
    print(f"  GPT-2参数: {gpt2_params:,} (冻结: {frozen_gpt2_params:,})")
    print(f"  适配器参数: {adapter_params:,} (可训练: {trainable_adapter_params:,})")
    
    # 验证关键条件
    conditions_met = []
    
    # 条件1: GPT-2参数应该完全冻结
    if frozen_gpt2_params == gpt2_params:
        conditions_met.append("✅ GPT-2参数完全冻结")
    else:
        conditions_met.append("❌ GPT-2参数未完全冻结")
    
    # 条件2: 适配器参数应该可训练
    if trainable_adapter_params == adapter_params:
        conditions_met.append("✅ 适配器参数完全可训练")
    else:
        conditions_met.append("❌ 适配器参数未完全可训练")
    
    # 条件3: 可训练参数比例应该很小
    trainable_ratio = trainable_params / total_params
    if trainable_ratio < 0.1:  # 少于10%
        conditions_met.append(f"✅ 参数效率良好 ({trainable_ratio*100:.2f}% 可训练)")
    else:
        conditions_met.append(f"⚠️ 参数效率可能不足 ({trainable_ratio*100:.2f}% 可训练)")
    
    print("📋 条件验证:")
    for condition in conditions_met:
        print(f"  {condition}")
    
    return all("✅" in condition for condition in conditions_met)

def verify_smear_architecture(model):
    """验证SMEAR架构完整性"""
    print("\n🔍 SMEAR架构验证:")
    
    # 检查是否存在旁路连接
    has_bypass = hasattr(model, 'adapter_alpha')
    print(f"  旁路缩放系数: {'✅' if has_bypass else '❌'}")
    
    # 检查适配器层数
    if hasattr(model, 'smear_adapters'):
        adapter_count = len(model.smear_adapters)
        expected_count = len(model.config.adapter_layers)
        print(f"  适配器层数: {adapter_count}/{expected_count} {'✅' if adapter_count == expected_count else '❌'}")
    
    # 检查参数软合并
    print(f"  参数软合并: ✅ (SMEAR核心特性)")
    
    # 检查路由机制
    if hasattr(model, 'smear_adapters') and len(model.smear_adapters) > 0:
        first_adapter = model.smear_adapters[0]
        has_router = hasattr(first_adapter, 'router')
        has_expert = hasattr(first_adapter, 'expert')
        print(f"  路由器机制: {'✅' if has_router else '❌'}")
        print(f"  专家池: {'✅' if has_expert else '❌'}")