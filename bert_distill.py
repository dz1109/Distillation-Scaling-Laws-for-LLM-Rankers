"""
bert_distill.py
===============
BERT Cross-Encoder 蒸馏脚本。

Teacher: 已有 LLM 的 teacher_probs（来自 parquet 数据）。
Student: BERT AutoModelForSequenceClassification。
Loss: JSD(student_probs, teacher_probs)

数据格式 (jsonl/parquet):
  query, title, passage, label, teacher_probs
  teacher_probs 是 JSON 字符串: '{"0": 0.1, "1": 0.2, "2": 0.5, "3": 0.2}'

用法:
  deepspeed --num_gpus=8 bert_distill.py \
      --model_name_or_path /path/to/bert \
      --train_data_path /path/to/distill_data.parquet_dir \
      --num_labels 4 \
      --output_dir ./output/bert_distill \
      --deepspeed ds_config_zero2.json
"""

import logging
import os
import sys
import json
import glob
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import pyarrow.parquet as pq
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
    train_data_path: str = field(default="", metadata={"help": "parquet 目录或 jsonl 文件"})
    eval_data_path: Optional[str] = field(default=None)
    teacher_prob_key: str = field(default="teacher_probs")


@dataclass
class DistillArguments(TrainingArguments):
    distill_type: str = field(
        default="jsd",
        metadata={
            "help": (
                "蒸馏类型: "
                "'soft'/'jsd' 使用 JS 散度; "
                "'fkl' 使用 forward KL = KL(teacher || student); "
                "'rkl' 使用 reverse KL = KL(student || teacher); "
                "'hard' 用 teacher 概率 argmax 作为硬标签, 交叉熵训练"
            ),
        },
    )
    distill_weight: float = field(
        default=1.0, metadata={"help": "蒸馏 loss 权重 (乘到最终 loss 上)"}
    )


# ============================================================================
# Dataset
# ============================================================================

class DistillDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, num_labels=4,
                 teacher_prob_key="teacher_probs"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_labels = num_labels
        self.samples = []

        logger.info(f"Loading data from {data_path}...")

        if os.path.isdir(data_path):
            parquet_files = sorted(glob.glob(os.path.join(data_path, "*.parquet")))
            if not parquet_files:
                parquet_files = sorted(glob.glob(os.path.join(data_path, "**/*.parquet"), recursive=True))
            for pf in parquet_files:
                table = pq.read_table(pf)
                for i in range(len(table)):
                    row = {col: table.column(col)[i].as_py() for col in table.column_names}
                    self._add_row(row, teacher_prob_key)
        elif data_path.endswith(".parquet"):
            table = pq.read_table(data_path)
            for i in range(len(table)):
                row = {col: table.column(col)[i].as_py() for col in table.column_names}
                self._add_row(row, teacher_prob_key)
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    row = json.loads(line.strip())
                    self._add_row(row, teacher_prob_key)

        logger.info(f"Loaded {len(self.samples)} samples")

    def _add_row(self, row, teacher_prob_key):
        query = row.get("query", "")
        title = row.get("title", "")
        passage = row.get("passage", "")
        label = int(row.get("label", 0))
        if self.num_labels == 2:
            label = min(label, 1)
        doc = (title + " " + passage).strip()

        tp = row.get(teacher_prob_key)
        if isinstance(tp, str):
            tp = json.loads(tp)
        teacher_probs = [float(tp.get(str(i), 0.0)) for i in range(self.num_labels)]

        self.samples.append((query, doc, label, teacher_probs))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query, doc, label, teacher_probs = self.samples[idx]
        encoded = self.tokenizer(
            query, doc,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_token_type_ids=True,
        )
        encoded["labels"] = label
        encoded["teacher_probs"] = teacher_probs
        return encoded


@dataclass
class DistillDataCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        labels = [f.pop("labels") for f in features]
        teacher_probs = [f.pop("teacher_probs") for f in features]
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        batch["teacher_probs"] = torch.tensor(teacher_probs, dtype=torch.float)
        return batch


# ============================================================================
# Trainer
# ============================================================================

class BertDistillTrainer(Trainer):
    def __init__(self, *args, distill_type: str = "jsd", distill_weight: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        valid_types = ("soft", "jsd", "fkl", "rkl", "hard")
        assert distill_type in valid_types, f"invalid distill_type: {distill_type}"
        # 'soft' 视为 'jsd' 的别名
        self.distill_type = "jsd" if distill_type == "soft" else distill_type
        self.distill_weight = distill_weight

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        teacher_probs = inputs.pop("teacher_probs")
        inputs.pop("labels", None)
        outputs = model(**inputs)
        logits = outputs.logits

        if self.distill_type == "hard":
            # 硬标签蒸馏: teacher argmax 作为类别标签, 整个 num_labels 上做 CE
            hard_labels = teacher_probs.argmax(dim=-1)
            loss = F.cross_entropy(logits, hard_labels)
        else:
            epsilon = 1e-8
            p_student = F.softmax(logits, dim=-1).clamp(min=epsilon, max=1 - epsilon)
            q_teacher = teacher_probs.clamp(min=epsilon, max=1 - epsilon)

            # F.kl_div(input=log(Q), target=P, reduction='batchmean') = KL(P || Q)
            if self.distill_type == "jsd":
                m = 0.5 * (p_student + q_teacher)
                kl_pm = F.kl_div(torch.log(m), p_student, reduction='batchmean')
                kl_qm = F.kl_div(torch.log(m), q_teacher, reduction='batchmean')
                loss = 0.5 * (kl_pm + kl_qm)
            elif self.distill_type == "fkl":
                # Forward KL: KL(teacher || student)
                loss = F.kl_div(torch.log(p_student), q_teacher, reduction='batchmean')
            elif self.distill_type == "rkl":
                # Reverse KL: KL(student || teacher)
                loss = F.kl_div(torch.log(q_teacher), p_student, reduction='batchmean')
            else:
                raise ValueError(f"unknown distill_type: {self.distill_type}")

        loss = self.distill_weight * loss
        return (loss, outputs) if return_outputs else loss


# ============================================================================
# Main
# ============================================================================

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, DistillArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Num labels: {model_args.num_labels}")
    logger.info(f"Data: {data_args.train_data_path}")
    logger.info(f"Distill type: {training_args.distill_type}, weight: {training_args.distill_weight}")
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=model_args.num_labels,
    )

    train_dataset = DistillDataset(
        data_args.train_data_path, tokenizer,
        max_length=model_args.max_length,
        num_labels=model_args.num_labels,
        teacher_prob_key=data_args.teacher_prob_key,
    )

    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = DistillDataset(
            data_args.eval_data_path, tokenizer,
            max_length=model_args.max_length,
            num_labels=model_args.num_labels,
            teacher_prob_key=data_args.teacher_prob_key,
        )

    collator = DistillDataCollator(tokenizer)

    trainer = BertDistillTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        distill_type=training_args.distill_type,
        distill_weight=training_args.distill_weight,
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
