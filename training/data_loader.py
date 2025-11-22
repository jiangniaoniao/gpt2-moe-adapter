import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer

def get_dataloaders(config):
    """通用数据加载器，支持BookCorpus和WikiText"""
    tokenizer = GPT2Tokenizer.from_pretrained("/home/yang/gpt2-moe-adapter/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # 根据配置选择数据集
    dataset_name = getattr(config, 'dataset_name', 'wikitext')
    dataset_config = getattr(config, 'dataset_config', 'wikitext-2-raw-v1')
    
    print(f"  加载数据集: {dataset_name} ({dataset_config})...")
    
    if dataset_name.lower() == "bookcorpus":
        return get_bookcorpus_dataloaders(config, tokenizer)
    elif dataset_name.lower() == "wikitext":
        return get_wikitext_dataloaders(config, tokenizer)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

def get_bookcorpus_dataloaders(config, tokenizer):
    """BookCorpus数据加载器"""
    print("  加载BookCorpus数据集...")
    
    # 加载BookCorpus
    dataset = load_dataset("bookcorpus", split="train", trust_remote_code=True)
    print(f"  成功加载BookCorpus，共 {len(dataset)} 个样本")
    
    # 自定义分割：训练集80%，验证集10%，测试集10%
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # 分割数据集
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, train_size + val_size))
    test_dataset = dataset.select(range(train_size + val_size, total_size))
    
    print(f"  数据集分割:")
    print(f"   - 训练集: {len(train_dataset)} 样本")
    print(f"   - 验证集: {len(val_dataset)} 样本") 
    print(f"   - 测试集: {len(test_dataset)} 样本")
    
    def tokenize_function(examples):
        """分词函数 - 针对BookCorpus优化"""
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
    print("  对数据集进行分词处理...")
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training set"
    )
    
    tokenized_val = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation set"
    )
    
    tokenized_test = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=test_dataset.column_names,
        desc="Tokenizing test set"
    )
    
    # 为语言建模准备labels
    def group_texts(examples):
        """将文本分组为固定长度的块"""
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        
        # 丢弃剩余部分
        if total_length >= config.max_length:
            total_length = (total_length // config.max_length) * config.max_length
        
        result = {
            k: [t[i : i + config.max_length] for i in range(0, total_length, config.max_length)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    # 应用文本分组
    print("  分组文本为固定长度块...")
    tokenized_train = tokenized_train.map(
        group_texts,
        batched=True,
        desc="Grouping training texts"
    )
    
    tokenized_val = tokenized_val.map(
        group_texts,
        batched=True,
        desc="Grouping validation texts"
    )
    
    tokenized_test = tokenized_test.map(
        group_texts,
        batched=True,
        desc="Grouping test texts"
    )
    
    # 创建数据加载器
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([torch.tensor(item['input_ids']) for item in batch]),
            'attention_mask': torch.stack([torch.tensor(item['attention_mask']) for item in batch]),
            'labels': torch.stack([torch.tensor(item['labels']) for item in batch])
        }
    
    train_loader = DataLoader(
        tokenized_train,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        tokenized_val,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        tokenized_test,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    print("  BookCorpus数据加载器创建完成")
    return train_loader, val_loader, test_loader, tokenizer

def get_wikitext_dataloaders(config, tokenizer):
    """WikiText数据加载器"""
    print("  加载WikiText数据集...")
    
    # 加载WikiText数据集
    dataset_config = getattr(config, 'dataset_config', 'wikitext-2-raw-v1')
    dataset = load_dataset("wikitext", dataset_config)
    
    # 获取数据集分割
    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"] if "test" in dataset else dataset["validation"]
    
    print(f"  成功加载WikiText:")
    print(f"   - 训练集: {len(train_dataset)} 样本")
    print(f"   - 验证集: {len(val_dataset)} 样本")
    print(f"   - 测试集: {len(test_dataset)} 样本")
    
    def wikitext_tokenize_function(examples):
        """WikiText分词函数"""
        # 处理WikiText的特殊格式
        texts = []
        for text in examples["text"]:
            # 跳过空行和标题行
            if text.strip() and not text.strip().startswith("="):
                texts.append(text.strip())
        
        if not texts:
            # 如果没有有效文本，返回空结果
            return {"input_ids": [], "attention_mask": []}
        
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=config.max_length,
            return_tensors=None
        )
        return tokenized
    
    # 分词处理
    print("  对WikiText数据集进行分词处理...")
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
    
    # 为语言建模准备labels
    def group_texts(examples):
        """将文本分组为固定长度的块"""
        # 过滤掉空的token序列
        valid_indices = [i for i, ids in enumerate(examples["input_ids"]) if len(ids) > 0]
        
        if not valid_indices:
            return {"input_ids": [], "attention_mask": [], "labels": []}
        
        filtered_examples = {k: [examples[k][i] for i in valid_indices] for k in examples.keys()}
        
        concatenated = {k: sum(filtered_examples[k], []) for k in filtered_examples.keys()}
        total_length = len(concatenated[list(filtered_examples.keys())[0]])
        
        # 丢弃剩余部分
        if total_length >= config.max_length:
            total_length = (total_length // config.max_length) * config.max_length
        
        result = {
            k: [t[i : i + config.max_length] for i in range(0, total_length, config.max_length)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    # 应用文本分组
    print("  分组WikiText文本为固定长度块...")
    tokenized_train = tokenized_train.map(
        group_texts,
        batched=True,
        desc="Grouping WikiText training texts"
    )
    
    tokenized_val = tokenized_val.map(
        group_texts,
        batched=True,
        desc="Grouping WikiText validation texts"
    )
    
    tokenized_test = tokenized_test.map(
        group_texts,
        batched=True,
        desc="Grouping WikiText test texts"
    )
    
    # 创建数据加载器
    def wikitext_collate_fn(batch):
        """WikiText数据整理函数"""
        # 过滤空批次
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