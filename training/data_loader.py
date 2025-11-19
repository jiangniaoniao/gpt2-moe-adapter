import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer

def get_bookcorpus_dataloaders(config):
    """专门为BookCorpus设计的数据加载器"""
    tokenizer = GPT2Tokenizer.from_pretrained("/home/yang/gpt2-moe-adapter/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
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