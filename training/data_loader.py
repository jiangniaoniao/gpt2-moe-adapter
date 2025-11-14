import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer

def get_wikitext_dataloaders(config):
    """加载Wikitext数据集"""
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
    
    # 创建数据加载器
    train_dataset = tokenized_datasets["train"]
    val_dataset = tokenized_datasets["validation"] if "validation" in tokenized_datasets else tokenized_datasets["test"]
    
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
    
    return train_loader, val_loader, tokenizer