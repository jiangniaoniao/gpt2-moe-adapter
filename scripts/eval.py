import torch
import numpy as np
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import json
from tqdm import tqdm
import re
from typing import Dict, Any, Optional
import os

class AutoEvaluator:
    """自动化评估器，使用多个benchmark数据集"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def evaluate_mmlu(self, subjects=None, num_samples=50):
        """评估MMLU学科知识（简化版）"""
        if subjects is None:
            subjects = ['abstract_algebra', 'anatomy', 'astronomy', 'business_ethics']
        
        results = {}
        
        for subject in subjects:
            try:
                # 加载MMLU子集
                dataset = load_dataset("cais/mmlu", subject)
                test_data = dataset['test']
                
                if len(test_data) > num_samples:
                    test_data = test_data.select(range(num_samples))
                
                correct = 0
                total = 0
                
                for example in tqdm(test_data, desc=f"MMLU-{subject}"):
                    question = example['question']
                    choices = example['choices']
                    answer_key = example['answer']
                    
                    # 构建提示
                    prompt = f"问题: {question}\n选项:\n"
                    for i, choice in enumerate(choices):
                        prompt += f"{chr(65+i)}. {choice}\n"
                    prompt += "正确答案是:"
                    
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=inputs['input_ids'].shape[1] + 10,
                            temperature=0.1,
                            do_sample=False,
                            pad_token_id=self.tokenizer.eos_token_id
                        )
                    
                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    answer_text = generated.replace(prompt, "").strip()
                    
                    # 提取第一个字母作为答案
                    if answer_text and answer_text[0].upper() in ['A', 'B', 'C', 'D']:
                        predicted = ord(answer_text[0].upper()) - 65
                        if predicted == answer_key:
                            correct += 1
                    
                    total += 1
                
                accuracy = correct / total if total > 0 else 0
                results[subject] = {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total
                }
                
            except Exception as e:
                print(f"MMLU {subject} 评估失败: {e}")
                results[subject] = {'accuracy': 0, 'correct': 0, 'total': 0}
        
        return results
    
    def evaluate_gsm8k(self, num_samples=50):
        """评估数学推理能力（GSM8K）"""
        try:
            dataset = load_dataset("gsm8k", "main")
            test_data = dataset['test']
            
            if len(test_data) > num_samples:
                test_data = test_data.select(range(num_samples))
            
            correct = 0
            total = 0
            
            for example in tqdm(test_data, desc="GSM8K"):
                question = example['question']
                answer = example['answer'].split("#### ")[-1].strip()
                
                prompt = f"数学问题: {question}\n让我们一步步推理:\n"
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 150,
                        temperature=0.3,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                reasoning = generated.replace(prompt, "").strip()
                
                # 提取最终答案数字
                numbers = re.findall(r'\d+\.?\d*', reasoning)
                if numbers:
                    predicted_answer = numbers[-1]  # 取最后一个数字
                    if predicted_answer == answer:
                        correct += 1
                
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
            
        except Exception as e:
            print(f"GSM8K 评估失败: {e}")
            return {'accuracy': 0, 'correct': 0, 'total': 0}
    
    def evaluate_arc(self, num_samples=50):
        """评估科学推理能力（ARC）"""
        try:
            dataset = load_dataset("ai2_arc", "ARC-Challenge")
            test_data = dataset['test']
            
            if len(test_data) > num_samples:
                test_data = test_data.select(range(num_samples))
            
            correct = 0
            total = 0
            
            for example in tqdm(test_data, desc="ARC"):
                question = example['question']
                choices = example['choices']['text']
                labels = example['choices']['label']
                answer_key = example['answerKey']
                
                prompt = f"科学问题: {question}\n选项:\n"
                for label, choice in zip(labels, choices):
                    prompt += f"{label}. {choice}\n"
                prompt += "正确答案是:"
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 10,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer_text = generated.replace(prompt, "").strip()
                
                # 提取答案标签
                if answer_text and answer_text[0].upper() in labels:
                    predicted = answer_text[0].upper()
                    if predicted == answer_key:
                        correct += 1
                
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
            
        except Exception as e:
            print(f"ARC 评估失败: {e}")
            return {'accuracy': 0, 'correct': 0, 'total': 0}
    
    def evaluate_commonsense_qa(self, num_samples=50):
        """评估常识推理能力"""
        try:
            dataset = load_dataset("tau/commonsense_qa")
            test_data = dataset['validation']
            
            if len(test_data) > num_samples:
                test_data = test_data.select(range(num_samples))
            
            correct = 0
            total = 0
            
            for example in tqdm(test_data, desc="CommonsenseQA"):
                question = example['question']
                choices = example['choices']['text']
                labels = example['choices']['label']
                answer_key = example['answerKey']
                
                prompt = f"常识问题: {question}\n选项:\n"
                for label, choice in zip(labels, choices):
                    prompt += f"{label}. {choice}\n"
                prompt += "正确答案是:"
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 10,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer_text = generated.replace(prompt, "").strip()
                
                # 提取答案标签
                if answer_text and answer_text[0].upper() in labels:
                    predicted = answer_text[0].upper()
                    if predicted == answer_key:
                        correct += 1
                
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
            
        except Exception as e:
            print(f"CommonsenseQA 评估失败: {e}")
            return {'accuracy': 0, 'correct': 0, 'total': 0}
    
    def evaluate_humanities(self, num_samples=50):
        """评估人文科学知识"""
        try:
            dataset = load_dataset("boolq")
            test_data = dataset['validation']
            
            if len(test_data) > num_samples:
                test_data = test_data.select(range(num_samples))
            
            correct = 0
            total = 0
            
            for example in tqdm(test_data, desc="BoolQ"):
                passage = example['passage']
                question = example['question']
                answer = example['answer']
                
                prompt = f"阅读以下文章:\n{passage}\n\n问题: {question}\n请回答是或否:"
                
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=inputs['input_ids'].shape[1] + 10,
                        temperature=0.1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer_text = generated.replace(prompt, "").strip().lower()
                
                # 检查是否包含"是"或"否"
                if answer and ('是' in answer_text or 'yes' in answer_text or 'true' in answer_text):
                    predicted = True
                elif not answer and ('否' in answer_text or 'no' in answer_text or 'false' in answer_text):
                    predicted = False
                else:
                    continue
                
                if predicted == answer:
                    correct += 1
                
                total += 1
            
            accuracy = correct / total if total > 0 else 0
            return {
                'accuracy': accuracy,
                'correct': correct,
                'total': total
            }
            
        except Exception as e:
            print(f"BoolQ 评估失败: {e}")
            return {'accuracy': 0, 'correct': 0, 'total': 0}
    
    def evaluate_language_modeling(self, num_samples=1000):
        """评估语言建模能力（困惑度）"""
        try:
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
            test_data = dataset['test']
            
            if len(test_data) > num_samples:
                test_data = test_data.select(range(num_samples))
            
            total_loss = 0
            total_tokens = 0
            
            self.model.eval()
            
            for example in tqdm(test_data, desc="语言建模"):
                text = example['text']
                if len(text.strip()) < 10:
                    continue
                
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss.item()
                
                total_loss += loss * inputs['input_ids'].shape[1]
                total_tokens += inputs['input_ids'].shape[1]
            
            avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
            perplexity = np.exp(avg_loss)
            
            return {
                'perplexity': perplexity,
                'loss': avg_loss,
                'total_tokens': total_tokens
            }
            
        except Exception as e:
            print(f"语言建模评估失败: {e}")
            return {'perplexity': float('inf'), 'loss': float('inf'), 'total_tokens': 0}
    
    def comprehensive_evaluation(self, num_samples=50):
        """综合评估所有能力"""
        print("  开始综合评估...")
        
        results = {
            '学科知识': self.evaluate_mmlu(num_samples=num_samples),
            '数学推理': self.evaluate_gsm8k(num_samples=num_samples),
            '科学推理': self.evaluate_arc(num_samples=num_samples),
            '常识推理': self.evaluate_commonsense_qa(num_samples=num_samples),
            '人文理解': self.evaluate_humanities(num_samples=num_samples),
            '语言建模': self.evaluate_language_modeling(num_samples=min(num_samples*20, 1000))
        }
        
        # 计算综合得分
        accuracies = []
        for category, result in results.items():
            if category == '学科知识':
                sub_accuracies = [v['accuracy'] for v in result.values() if 'accuracy' in v]
                if sub_accuracies:
                    accuracies.append(np.mean(sub_accuracies))
            elif category == '语言建模':
                perplexity = result.get('perplexity', float('inf'))
                if perplexity < float('inf'):
                    lm_score = max(0, 1 - (perplexity / 100))
                    accuracies.append(lm_score)
            elif 'accuracy' in result:
                accuracies.append(result['accuracy'])
        
        if accuracies:
            results['综合得分'] = np.mean(accuracies)
        
        return results

def load_model(model_path: str, base_model_path: Optional[str] = None, model_type="auto"):
    """
    加载模型，支持多种模型类型
    
    Args:
        model_path: 模型路径
        base_model_path: 基础模型路径（仅用于LoRA/MoE模型）
        model_type: 模型类型，可选值: "auto", "original", "lora", "moe"
    """
    print(f"  加载模型从: {model_path}")
    
    # 自动检测模型类型
    if model_type == "auto":
        # 检查是否有adapter_config.json文件 (LoRA)
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            model_type = "lora"
        # 检查是否有MoE相关的文件
        elif os.path.exists(os.path.join(model_path, "complete_model.pth")) or \
             os.path.exists(os.path.join(model_path, "smear_adapters.pth")):
            model_type = "moe"
        else:
            model_type = "original"
    
    print(f"  检测到模型类型: {model_type}")
    
    # 加载tokenizer
    if model_type == "moe" and base_model_path:
        # MoE模型使用基础模型的tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(base_model_path)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  使用设备: {device}")
    
    if model_type == "lora":
        # 加载LoRA模型
        if base_model_path is None:
            base_model_path = model_path
            print(f"  警告: 未提供基础模型路径，使用 {model_path} 作为基础模型")
        
        print(f"  加载基础模型: {base_model_path}")
        base_model = GPT2LMHeadModel.from_pretrained(base_model_path)
        
        print(f"  加载LoRA适配器")
        try:
            model = PeftModel.from_pretrained(base_model, model_path)
            model = model.merge_and_unload()
            print("  LoRA模型加载并合并成功")
        except Exception as e:
            print(f"  LoRA模型加载失败: {e}")
            print("  尝试直接加载为完整模型...")
            model = GPT2LMHeadModel.from_pretrained(model_path)
    
    elif model_type == "moe":
        # 加载MoE Adapter模型
        model = load_moe_adapter_model(model_path, base_model_path, device)
    
    else:
        # 加载原模型
        print(f"  加载原GPT-2模型")
        model = GPT2LMHeadModel.from_pretrained(model_path)
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device

def load_moe_adapter_model(model_path, base_model_path, device):
    """
    加载MoE Adapter模型
    """
    print("  加载MoE Adapter模型...")
    
    # 首先尝试导入你的MoE Adapter类
    try:
        # 添加项目路径到sys.path
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        sys.path.append("/home/yang/gpt2-moe-adapter")
        
        # 尝试导入你的MoE Adapter模型
        try:
            from models.gpt2_smear_model import GPT2WithSmearAdapter
            from config.base_config import GPT2SmearConfig, SmearAdapterConfig
            print("  ✅ 成功导入MoE Adapter模块")
        except ImportError as e:
            print(f"  ❌ 导入MoE Adapter模块失败: {e}")
            print("  尝试加载为普通GPT-2模型...")
            return GPT2LMHeadModel.from_pretrained(model_path)
        
        # 检查模型文件
        model_files = os.listdir(model_path)
        
        if "complete_model.pth" in model_files:
            # 加载完整模型
            print("  检测到完整模型格式，加载中...")
            checkpoint_path = os.path.join(model_path, "complete_model.pth")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            # 从检查点获取配置
            if 'config' in checkpoint:
                config_dict = checkpoint['config']
                
                # 重建配置对象
                if 'smear_config' in config_dict:
                    smear_config_dict = config_dict['smear_config']
                    smear_config = SmearAdapterConfig(**smear_config_dict)
                    
                    # 创建模型配置
                    model_config = GPT2SmearConfig(
                        base_model=config_dict.get('base_model', 'gpt2'),
                        smear_config=smear_config,
                        **{k: v for k, v in config_dict.items() 
                           if k not in ['base_model', 'smear_config']}
                    )
                else:
                    # 使用默认配置
                    smear_config = SmearAdapterConfig(
                        num_experts=4,
                        expert_size=512,
                        routing_granularity="token",
                        segment_length=256,
                        routing_strategy="top_k_sparse",
                        top_k=2
                    )
                    model_config = GPT2SmearConfig(
                        base_model="gpt2",
                        num_adapter_layers=5,
                        adapter_layers=[2, 4, 6, 8, 10],
                        freeze_base_model=True,
                        smear_config=smear_config
                    )
                
                # 创建模型
                model = GPT2WithSmearAdapter(model_config)
                
                # 加载权重
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("  ✅ 完整MoE模型加载成功")
                else:
                    print("  ⚠️ 检查点中没有模型权重，使用随机初始化")
            
            else:
                print("  ⚠️ 检查点中没有配置信息，使用默认配置")
                smear_config = SmearAdapterConfig(
                    num_experts=4,
                    expert_size=512,
                    routing_granularity="token",
                    segment_length=256,
                    routing_strategy="top_k_sparse",
                    top_k=2
                )
                model_config = GPT2SmearConfig(
                    base_model="gpt2",
                    num_adapter_layers=5,
                    adapter_layers=[2, 4, 6, 8, 10],
                    freeze_base_model=True,
                    smear_config=smear_config
                )
                model = GPT2WithSmearAdapter(model_config)
        
        elif "smear_adapters.pth" in model_files:
            # 加载仅适配器权重
            print("  检测到仅适配器格式，加载中...")
            adapter_path = os.path.join(model_path, "smear_adapters.pth")
            adapter_checkpoint = torch.load(adapter_path, map_location=device)
            
            # 从适配器文件获取配置或使用默认
            if 'config' in adapter_checkpoint:
                config_dict = adapter_checkpoint['config']
                if 'smear_config' in config_dict:
                    smear_config = SmearAdapterConfig(**config_dict['smear_config'])
                    model_config = GPT2SmearConfig(
                        base_model=config_dict.get('base_model', 'gpt2'),
                        smear_config=smear_config,
                        **{k: v for k, v in config_dict.items() 
                           if k not in ['base_model', 'smear_config']}
                    )
                else:
                    # 使用默认配置
                    smear_config = SmearAdapterConfig(
                        num_experts=4,
                        expert_size=512,
                        routing_granularity="token",
                        segment_length=256,
                        routing_strategy="top_k_sparse",
                        top_k=2
                    )
                    model_config = GPT2SmearConfig(
                        base_model="gpt2",
                        num_adapter_layers=5,
                        adapter_layers=[2, 4, 6, 8, 10],
                        freeze_base_model=True,
                        smear_config=smear_config
                    )
            else:
                # 使用默认配置
                smear_config = SmearAdapterConfig(
                    num_experts=4,
                    expert_size=512,
                    routing_granularity="token",
                    segment_length=256,
                    routing_strategy="top_k_sparse",
                    top_k=2
                )
                model_config = GPT2SmearConfig(
                    base_model="gpt2",
                    num_adapter_layers=5,
                    adapter_layers=[2, 4, 6, 8, 10],
                    freeze_base_model=True,
                    smear_config=smear_config
                )
            
            # 创建模型
            model = GPT2WithSmearAdapter(model_config)
            
            # 加载适配器权重
            if 'smear_adapters' in adapter_checkpoint:
                adapter_weights = adapter_checkpoint['smear_adapters']
                
                # 获取当前模型状态
                model_state = model.state_dict()
                
                # 只更新MoE相关的参数
                for name, param in adapter_weights.items():
                    if name in model_state:
                        model_state[name] = param
                    else:
                        print(f"   跳过不匹配的参数: {name}")
                
                # 加载更新后的状态
                model.load_state_dict(model_state, strict=False)
                print("  ✅ MoE适配器权重加载成功")
            else:
                print("  ⚠️ 适配器文件中没有权重，使用随机初始化")
        
        else:
            # 没有找到MoE文件，加载为普通模型
            print(f"  ⚠️ 未找到MoE模型文件，尝试加载为普通模型")
            try:
                model = GPT2LMHeadModel.from_pretrained(model_path)
            except:
                # 如果失败，创建一个新的GPT-2模型
                print(f"  ⚠️ 无法加载模型，创建新的GPT-2模型")
                model = GPT2LMHeadModel.from_pretrained("gpt2")
        
        return model
        
    except Exception as e:
        print(f"  ❌ MoE模型加载失败: {e}")
        print("  回退到普通GPT-2模型...")
        return GPT2LMHeadModel.from_pretrained("gpt2")

def print_evaluation_results(results, model_name=""):
    """打印评估结果"""
    print("\n" + "="*60)
    print(f"  {model_name}评估结果汇总")
    print("="*60)
    
    for category, result in results.items():
        if category == '学科知识':
            print(f"\n{category}:")
            for subject, metrics in result.items():
                accuracy = metrics.get('accuracy', 0)
                correct = metrics.get('correct', 0)
                total = metrics.get('total', 0)
                print(f"  {subject:20}: {accuracy:.1%} ({correct}/{total})")
            
            sub_accuracies = [v.get('accuracy', 0) for v in result.values() if 'accuracy' in v]
            if sub_accuracies:
                avg_accuracy = np.mean(sub_accuracies)
                print(f"  {'平均':20}: {avg_accuracy:.1%}")
            
        elif category == '语言建模':
            perplexity = result.get('perplexity', float('inf'))
            loss = result.get('loss', float('inf'))
            print(f"\n{category}:")
            print(f"  困惑度: {perplexity:.2f}")
            print(f"  损失: {loss:.4f}")
            
        elif category == '综合得分':
            print(f"\n{category}: {result:.1%}")
            
        else:
            accuracy = result.get('accuracy', 0)
            correct = result.get('correct', 0)
            total = result.get('total', 0)
            print(f"\n{category}: {accuracy:.1%} ({correct}/{total})")

def save_results(results, filename="evaluation_results.json"):
    """保存评估结果到文件"""
    def convert_types(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        else:
            return obj
    
    results = convert_types(results)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"  评估结果已保存到: {filename}")

def compare_models(original_model_path, trained_model_path, base_model_path=None, 
                   trained_model_type="auto", num_samples=30):
    """比较原模型和训练后模型的性能"""
    print("="*60)
    print("  开始模型对比评估")
    print("="*60)
    
    # 加载原模型
    print("\n1. 评估原GPT-2模型:")
    original_model, original_tokenizer, device = load_model(
        original_model_path, 
        model_type="original"
    )
    original_evaluator = AutoEvaluator(original_model, original_tokenizer, device)
    original_results = original_evaluator.comprehensive_evaluation(num_samples=num_samples)
    original_results['模型类型'] = '原始GPT-2模型'
    original_results['模型路径'] = original_model_path
    
    # 加载训练后模型
    print(f"\n2. 评估训练后模型 (类型: {trained_model_type}):")
    trained_model, trained_tokenizer, device = load_model(
        trained_model_path, 
        base_model_path=base_model_path,
        model_type=trained_model_type
    )
    trained_evaluator = AutoEvaluator(trained_model, trained_tokenizer, device)
    trained_results = trained_evaluator.comprehensive_evaluation(num_samples=num_samples)
    trained_results['模型类型'] = '训练后模型'
    trained_results['模型路径'] = trained_model_path
    trained_results['模型具体类型'] = trained_model_type
    
    # 打印对比结果
    print("\n" + "="*60)
    print("  模型性能对比")
    print("="*60)
    
    categories = ['数学推理', '科学推理', '常识推理', '人文理解']
    
    for category in categories:
        if category in original_results and category in trained_results:
            orig_acc = original_results[category].get('accuracy', 0)
            trained_acc = trained_results[category].get('accuracy', 0)
            improvement = trained_acc - orig_acc
            
            print(f"\n{category}:")
            print(f"  原模型: {orig_acc:.1%}")
            print(f"  训练后: {trained_acc:.1%}")
            print(f"  提升: {improvement:+.1%}")
    
    # MMLU对比
    if '学科知识' in original_results and '学科知识' in trained_results:
        print(f"\n学科知识 (MMLU):")
        orig_subjects = [v.get('accuracy', 0) for v in original_results['学科知识'].values() if 'accuracy' in v]
        trained_subjects = [v.get('accuracy', 0) for v in trained_results['学科知识'].values() if 'accuracy' in v]
        
        if orig_subjects and trained_subjects:
            orig_mmlu = np.mean(orig_subjects)
            trained_mmlu = np.mean(trained_subjects)
            improvement = trained_mmlu - orig_mmlu
            print(f"  原模型平均: {orig_mmlu:.1%}")
            print(f"  训练后平均: {trained_mmlu:.1%}")
            print(f"  提升: {improvement:+.1%}")
    
    # 综合得分对比
    if '综合得分' in original_results and '综合得分' in trained_results:
        print(f"\n综合得分:")
        orig_overall = original_results['综合得分']
        trained_overall = trained_results['综合得分']
        improvement = trained_overall - orig_overall
        print(f"  原模型: {orig_overall:.1%}")
        print(f"  训练后: {trained_overall:.1%}")
        print(f"  提升: {improvement:+.1%}")
    
    # 保存结果
    comparison_results = {
        '原模型': original_results,
        '训练后模型': trained_results,
        '对比时间': str(np.datetime64('now')),
        '训练后模型类型': trained_model_type
    }
    
    timestamp = str(np.datetime64('now')).replace(':', '-').replace(' ', '_')
    filename = f"model_comparison_{timestamp}.json"
    save_results(comparison_results, filename)
    
    return original_results, trained_results

def main():
    """主评估函数"""
    # 配置参数
    ORIGINAL_MODEL_PATH = "gpt2"  # 原GPT-2模型
    TRAINED_MODEL_PATH = "/home/yang/gpt2-moe-adapter/best_smear_model"  # 训练好的模型路径
    BASE_MODEL_PATH = "gpt2"  # LoRA/MoE基础模型路径
    TRAINED_MODEL_TYPE = "auto"  # 训练后模型类型: "auto", "original", "lora", "moe"
    NUM_SAMPLES = 30  # 每个测试集的样本数量
    
    print("  模型评估")
    print("="*60)
    print(f"  开始对比评估原模型和训练后模型")
    print(f"  训练后模型路径: {TRAINED_MODEL_PATH}")
    print(f"  训练后模型类型: {TRAINED_MODEL_TYPE}")
    print("="*60)
    
    try:
        # 对比评估
        original_results, trained_results = compare_models(
            original_model_path=ORIGINAL_MODEL_PATH,
            trained_model_path=TRAINED_MODEL_PATH,
            base_model_path=BASE_MODEL_PATH,
            trained_model_type=TRAINED_MODEL_TYPE,
            num_samples=NUM_SAMPLES
        )
        
        print("\n" + "="*60)
        print("  评估完成!")
        print("="*60)
        
    except Exception as e:
        print(f"  评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()