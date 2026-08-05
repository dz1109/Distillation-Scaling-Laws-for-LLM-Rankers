"""
bert_train.py
=============
BERT Cross-Encoder 训练脚本（hard label）。

输入 query+doc pair，输出相关性分类（0/1 或 0/1/2/3）。
使用 AutoModelForSequenceClassification，兼容任意 BERT-like 模型。

数据格式 (jsonl):
  {"qid": "...", "query": "...", "title": "...", "passage": "...", "label": 2}

用法:
  deepspeed --num_gpus=8 bert_train.py \
      --model_name_or_path /path/to/bert \
      --train_data_path /path/to/train.jsonl \
      --num_labels 4 \
      --output_dir ./output/bert_ce \
      --deepspeed ds_config_zero2.json
"""

import logging
import os
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    PreTrainedTokenizerBase,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="bert-base-chinese")
    num_labels: int = field(default=4, metadata={"help": "分类标签数: 2 或 4"})
    max_length: int = field(default=512, metadata={"help": "最大序列长度"})


@dataclass
class DataArguments:
    train_data_path: str = field(default="")
    eval_data_path: Optional[str] = field(default=None)


# ============================================================================
# Dataset
# ============================================================================

class CrossEncoderDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, num_labels=4):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels
        self.samples = []

        logger.info(f"Loading data from {data_path}...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line.strip())
                query = obj.get("query", "")
                title = obj.get("title", "")
                passage = obj.get("passage", "")
                label = int(obj.get("label", 0))
                if self.num_labels == 2:
                    label = min(label, 1)
                doc = (title + " " + passage).strip()
                self.samples.append((query, doc, label))

        logger.info(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query, doc, label = self.samples[idx]
        encoded = self.tokenizer(
            query, doc,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_token_type_ids=True,
        )
        encoded["labels"] = label
        return encoded


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [f.pop("labels") for f in features]
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


# ============================================================================
# Main
# ============================================================================

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Num labels: {model_args.num_labels}")
    logger.info(f"Data: {data_args.train_data_path}")
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=model_args.num_labels,
    )

    train_dataset = CrossEncoderDataset(
        data_args.train_data_path, tokenizer,
        max_length=model_args.max_length,
        num_labels=model_args.num_labels,
    )

    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = CrossEncoderDataset(
            data_args.eval_data_path, tokenizer,
            max_length=model_args.max_length,
            num_labels=model_args.num_labels,
        )

    collator = DataCollator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
