import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer

def get_dataloaders(config):
    """加载数据集 - 包含训练集、验证集和测试集"""
    tokenizer = GPT2Tokenizer.from_pretrained("/home/yang/gpt2-moe-adapter/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    dataset = load_dataset(config.dataset_name, config.dataset_config)
    
    def tokenize_function(examples):
        # 连接文本并分词
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=config.max_length,
            return_tensors=None
        )
        return tokenized
    
    # 分词处理
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # 为语言建模准备labels
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        
        # 我们丢弃剩余部分，但如果数据集足够大则没问题
        if total_length >= config.max_length:
            total_length = (total_length // config.max_length) * config.max_length
        
        result = {
            k: [t[i : i + config.max_length] for i in range(0, total_length, config.max_length)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc="Grouping texts in chunks of 1024",
    )
    
    # 创建三个数据加载器：训练集、验证集、测试集
    train_dataset = tokenized_datasets["train"]
    
    # 优先使用validation作为验证集，如果没有则使用部分test集
    if "validation" in tokenized_datasets:
        val_dataset = tokenized_datasets["validation"]
        test_dataset = tokenized_datasets["test"] if "test" in tokenized_datasets else None
    else:
        # 如果没有validation，将test集分割为验证集和测试集
        test_split = tokenized_datasets["test"]
        split_ratio = getattr(config, 'val_test_split_ratio', 0.5)
        split_idx = int(len(test_split) * split_ratio)
        
        val_dataset = test_split.select(range(split_idx))
        test_dataset = test_split.select(range(split_idx, len(test_split)))
    
    # 训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
            'labels': torch.stack([torch.tensor(item['labels']) for item in batch])
        }
    )
    
    # 验证数据加载器
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
            'labels': torch.stack([torch.tensor(item['labels']) for item in batch])
        }
    )
    
    # 测试数据加载器（如果测试集存在）
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=lambda batch: {
                'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
                'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
                'labels': torch.stack([torch.tensor(item['labels']) for item in batch])
            }
        )
    
    # 打印数据集信息
    print(f"📊 数据集信息:")
    print(f"   - 训练集样本数: {len(train_dataset)}")
    print(f"   - 验证集样本数: {len(val_dataset)}")
    if test_loader:
        print(f"   - 测试集样本数: {len(test_dataset)}")
    else:
        print(f"   - 测试集: 未提供")
    
    return train_loader, val_loader, test_loader, tokenizer

def get_wikitext_dataloaders_with_custom_split(config, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """可选：自定义数据集分割比例"""
    assert train_ratio + val_ratio + test_ratio == 1.0, "分割比例之和必须为1"
    
    tokenizer = GPT2Tokenizer.from_pretrained("/home/yang/gpt2-moe-adapter/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 加载数据集
    dataset = load_dataset(config.dataset_name, config.dataset_config)
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=config.max_length,
            return_tensors=None
        )
        return tokenized
    
    # 分词处理
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    
    # 为语言建模准备labels
    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        
        if total_length >= config.max_length:
            total_length = (total_length // config.max_length) * config.max_length
        
        result = {
            k: [t[i : i + config.max_length] for i in range(0, total_length, config.max_length)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    tokenized_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc="Grouping texts in chunks of 1024",
    )
    
    # 自定义分割
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
            'labels': torch.stack([torch.tensor(item['labels']) for item in batch])
        }
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
            'labels': torch.stack([torch.tensor(item['labels']) for item in batch])
        }
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=lambda batch: {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
            'labels': torch.stack([torch.tensor(item['labels']) for item in batch])
        }
    )
    
    print(f"📊 自定义数据集分割:")
    print(f"   - 训练集样本数: {len(train_dataset)}")
    print(f"   - 验证集样本数: {len(val_dataset)}")
    print(f"   - 测试集样本数: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, tokenizer