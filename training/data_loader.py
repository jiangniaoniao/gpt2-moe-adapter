import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import GPT2Tokenizer
import random

def get_dataloaders(config):
    """支持多样化数据集混合的数据加载器"""
    tokenizer = GPT2Tokenizer.from_pretrained("/home/yang/gpt2-moe-adapter/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset_mode = getattr(config, 'dataset_mode', 'single')
    
    if dataset_mode == 'mixed':
        return get_mixed_dataloaders(config, tokenizer)
    else:
        return get_wikitext_dataloaders(config, tokenizer)

def get_mixed_dataloaders(config, tokenizer):
    """修复样本数量问题的混合数据集加载器"""
    print("🚀 加载混合多样化数据集...")
    
    # 数据集配置
    dataset_mix = getattr(config, 'dataset_mix', [
        # 基础指令数据 (40%)
        ("tatsu-lab/alpaca", None, 0.25, "instruction"),
        ("databricks/databricks-dolly-15k", None, 0.15, "instruction"),
        
        # 学科知识数据 (40%)
        ("qwedsacf/ivi-mmlu", None, 0.2, "knowledge"),
        ("allenai/sciq", None, 0.2, "knowledge"),
        
        # 推理数据 (20%)
        ("gsm8k", "main", 0.1, "reasoning"),
        ("tau/commonsense_qa", None, 0.1, "reasoning"),
    ])
    
    # 目标总样本数
    target_total_samples = getattr(config, 'target_total_samples', 50000)
    
    all_datasets = []
    dataset_info = []
    
    for i, (dataset_name, dataset_config, weight, dataset_type) in enumerate(dataset_mix):
        try:
            print(f"  📂 加载数据集 {i+1}/{len(dataset_mix)}: {dataset_name}")
            
            # 加载数据集
            if dataset_config:
                dataset = load_dataset(dataset_name, dataset_config)
            else:
                dataset = load_dataset(dataset_name)
            
            # 获取训练分割
            if "train" in dataset:
                train_data = dataset["train"]
            elif "training" in dataset:
                train_data = dataset["training"]
            else:
                first_split = list(dataset.keys())[0]
                train_data = dataset[first_split]
            
            print(f"    原始数据集大小: {len(train_data)}")
            
            # 格式化数据集 - 使用新的处理函数
            formatted_dataset = format_and_process_dataset(
                train_data, dataset_type, tokenizer, config
            )
            
            if len(formatted_dataset) == 0:
                print(f"    ⚠️  格式化后无有效样本，跳过")
                continue
            
            # 计算目标样本数 - 更合理的采样策略
            target_samples = min(
                int(target_total_samples * weight),  # 直接按权重计算
                len(formatted_dataset)
            )
            
            # 采样
            if target_samples < len(formatted_dataset):
                indices = random.sample(range(len(formatted_dataset)), target_samples)
                formatted_dataset = formatted_dataset.select(indices)
            
            all_datasets.append(formatted_dataset)
            dataset_info.append({
                'name': dataset_name,
                'type': dataset_type,
                'weight': weight,
                'samples': len(formatted_dataset),
                'original_samples': len(train_data)
            })
            
            print(f"    ✅ 成功加载 {len(formatted_dataset)}/{len(train_data)} 个样本")
            
        except Exception as e:
            print(f"    ❌ 加载失败: {e}")
            continue
    
    if not all_datasets:
        raise ValueError("没有成功加载任何数据集！")
    
    print("🔗 合并数据集...")
    combined_dataset = concatenate_datasets(all_datasets)
    
    # 如果总样本数仍太少，考虑重复采样
    if len(combined_dataset) < 20000:
        print(f"⚠️  总样本数较少 ({len(combined_dataset)})，考虑重复采样...")
        repeat_times = max(1, 30000 // len(combined_dataset))
        combined_dataset = concatenate_datasets([combined_dataset] * repeat_times)
        print(f"    重复采样后: {len(combined_dataset)} 样本")
    
    # 打乱数据
    combined_dataset = combined_dataset.shuffle(seed=42)
    
    # 打印数据集统计
    print("\n📊 数据集混合统计:")
    for info in dataset_info:
        actual_weight = info['samples'] / len(combined_dataset) if len(combined_dataset) > 0 else 0
        print(f"   - {info['name']} ({info['type']}): {info['samples']} 样本 "
              f"(目标权重: {info['weight']:.2f}, 实际权重: {actual_weight:.2f})")
    print(f"   📈 总样本数: {len(combined_dataset)}")
    
    # 分割数据集
    total_size = len(combined_dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset = combined_dataset.select(range(train_size))
    val_dataset = combined_dataset.select(range(train_size, train_size + val_size))
    test_dataset = combined_dataset.select(range(train_size + val_size, total_size))
    
    print(f"\n📋 最终数据集分割:")
    print(f"   - 训练集: {len(train_dataset)} 样本")
    print(f"   - 验证集: {len(val_dataset)} 样本") 
    print(f"   - 测试集: {len(test_dataset)} 样本")
    
    # 创建数据加载器
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
            'labels': torch.stack([torch.tensor(item['labels']) for item in batch])
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print("🎉 混合数据加载器创建完成！")
    return train_loader, val_loader, test_loader, tokenizer

def format_and_process_dataset(dataset, dataset_type, tokenizer, config):
    """新的格式化处理函数 - 保持样本独立性"""
    
    # 先格式化文本
    formatted_dataset = format_dataset_text(dataset, dataset_type)
    
    if len(formatted_dataset) == 0:
        return formatted_dataset
    
    # 然后进行分词处理
    return process_dataset_with_padding(formatted_dataset, tokenizer, config)

def format_dataset_text(dataset, dataset_type):
    """仅格式化文本内容，保持样本独立性"""
    
    def instruction_format(example):
        # Alpaca格式
        if 'instruction' in example and 'output' in example:
            input_text = f"Instruction: {example['instruction']}\n"
            if example.get('input', '').strip():
                input_text += f"Input: {example['input']}\n"
            input_text += f"Response: {example['output']}"
            return {"text": input_text}
        
        # Dolly格式
        elif 'instruction' in example and 'response' in example:
            input_text = f"Instruction: {example['instruction']}\n"
            if example.get('context', '').strip():
                input_text += f"Context: {example['context']}\n"
            input_text += f"Response: {example['response']}"
            return {"text": input_text}
        
        # 其他格式
        elif 'question' in example and 'answer' in example:
            return {"text": f"Q: {example['question']}\nA: {example['answer']}"}
        elif 'text' in example:
            return {"text": example['text']}
        
        return {"text": str(example)}
    
    def knowledge_format(example):
        # MMLU替代格式
        if 'input' in example and 'target' in example:
            return {"text": f"Question: {example['input']}\nAnswer: {example['target']}"}
        
        # SciQ格式
        elif 'question' in example and 'correct_answer' in example:
            question = example['question']
            choices = [example['distractor1'], example['distractor2'], 
                      example['distractor3'], example['correct_answer']]
            random.shuffle(choices)
            answer = example['correct_answer']
            
            choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
            text = f"Question: {question}\nOptions:\n{choices_text}\nAnswer: {answer}"
            return {"text": text}
        
        elif 'question' in example and 'answer' in example:
            return {"text": f"Q: {example['question']}\nA: {example['answer']}"}
        
        return {"text": str(example)}
    
    def reasoning_format(example):
        # GSM8K格式
        if 'question' in example and 'answer' in example:
            return {"text": f"Math Problem: {example['question']}\nSolution: {example['answer']}"}
        
        # CommonsenseQA格式
        elif 'question' in example and 'choices' in example and 'answerKey' in example:
            question = example['question']
            choices = example['choices']
            answer_key = example['answerKey']
            
            if isinstance(choices, dict) and 'text' in choices:
                choices = choices['text']
            
            if isinstance(choices, list):
                choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
                answer_idx = ord(answer_key) - ord('A') if len(answer_key) == 1 else int(answer_key)
                answer = choices[answer_idx] if answer_idx < len(choices) else answer_key
                text = f"Question: {question}\nOptions:\n{choices_text}\nAnswer: {answer}"
                return {"text": text}
        
        return {"text": str(example)}
    
    def lm_format(example):
        if 'text' in example:
            text = example['text'].strip()
            if text and (not text.startswith("=") or len(text) > 10):
                return {"text": text}
        return {"text": ""}
    
    # 选择格式化函数
    if dataset_type == "instruction":
        format_func = instruction_format
    elif dataset_type == "knowledge":
        format_func = knowledge_format
    elif dataset_type == "reasoning":
        format_func = reasoning_format
    else:
        format_func = lm_format
    
    try:
        formatted = dataset.map(format_func)
        # 过滤空文本
        formatted = formatted.filter(lambda x: x['text'] and x['text'].strip())
        return formatted
    except Exception as e:
        print(f"    ⚠️  数据格式化失败: {e}, 使用原始文本")
        # 尝试使用原始文本
        if 'text' in dataset.column_names:
            return dataset.filter(lambda x: x['text'] and x['text'].strip())
        else:
            # 如果连text列都没有，创建一个
            return dataset.map(lambda x: {"text": str(x)}).filter(lambda x: x['text'] and x['text'].strip())

def process_dataset_with_padding(dataset, tokenizer, config):
    """使用填充而不是分组来处理数据集 - 保持样本数量"""
    
    def tokenize_function(examples):
        texts = [text for text in examples["text"] if text and text.strip()]
        
        if not texts:
            return {"input_ids": [], "attention_mask": []}
        
        # 使用填充到最大长度
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding='max_length',  # 关键修改：使用填充而不是分组
            max_length=config.max_length,
            return_tensors=None
        )
        
        # 为语言建模设置labels
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # 分词处理
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset with padding"
    )
    
    # 过滤掉太短的序列（如果有的话）
    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x.get('input_ids', [])) > 10
    )
    
    return tokenized_dataset

def format_dataset(dataset, dataset_type, tokenizer, config):
    """向后兼容的包装函数"""
    return format_and_process_dataset(dataset, dataset_type, tokenizer, config)

def format_instruction_data(dataset, tokenizer, config):
    return format_and_process_dataset(dataset, "instruction", tokenizer, config)

def format_knowledge_data(dataset, tokenizer, config):
    return format_and_process_dataset(dataset, "knowledge", tokenizer, config)

def format_reasoning_data(dataset, tokenizer, config):
    return format_and_process_dataset(dataset, "reasoning", tokenizer, config)

def format_lm_data(dataset, tokenizer, config):
    return format_and_process_dataset(dataset, "lm", tokenizer, config)

# 保留原有的WikiText数据加载器（保持不变）
def get_wikitext_dataloaders(config, tokenizer):
    """WikiText数据加载器"""
    print("  加载WikiText数据集...")
    
    dataset_config = getattr(config, 'dataset_config', 'wikitext-2-raw-v1')
    dataset = load_dataset("wikitext", dataset_config)
    
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"] if "test" in dataset else dataset["validation"]
    
    print(f"  成功加载WikiText:")
    print(f"   - 训练集: {len(train_dataset)} 样本")
    print(f"   - 验证集: {len(val_dataset)} 样本")
    print(f"   - 测试集: {len(test_dataset)} 样本")
    
    def wikitext_tokenize_function(examples):
        texts = []
        for text in examples["text"]:
            if text.strip() and not text.strip().startswith("="):
                texts.append(text.strip())
        
        if not texts:
            return {"input_ids": [], "attention_mask": []}
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding='max_length',  # 也改为填充
            max_length=config.max_length,
            return_tensors=None
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    tokenized_train = train_dataset.map(
        wikitext_tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing WikiText training set"
    )
    
    tokenized_val = val_dataset.map(
        wikitext_tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing WikiText validation set"
    )
    
    tokenized_test = test_dataset.map(
        wikitext_tokenize_function,
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing WikiText test set"
    )
    
    # 移除分组步骤，直接使用填充后的数据
    
    def wikitext_collate_fn(batch):
        valid_batch = [item for item in batch if len(item['input_ids']) > 0]
        
        if not valid_batch:
            return {
                'input_ids': torch.empty((0, config.max_length), dtype=torch.long),
                'attention_mask': torch.empty((0, config.max_length), dtype=torch.long),
                'labels': torch.empty((0, config.max_length), dtype=torch.long)
            }
        
        return {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in valid_batch]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in valid_batch]),
            'labels': torch.stack([torch.tensor(item['labels']) for item in valid_batch])
        }
    
    train_loader = DataLoader(
        tokenized_train,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=wikitext_collate_fn
    )
    
    val_loader = DataLoader(
        tokenized_val,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=wikitext_collate_fn
    )
    
    test_loader = DataLoader(
        tokenized_test,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=wikitext_collate_fn
    )
    
    print("  WikiText数据加载器创建完成")
    return train_loader, val_loader, test_loader, tokenizer