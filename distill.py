"""
distill.py
==========
基于 JS 散度的知识蒸馏训练脚本（单维度相关性）。

Teacher 提供 label token 的概率分布，Student 学习复现该分布。
支持 2 标签 (0/1) 和 4 标签 (0/1/2/3) 两种模式。

数据格式 (jsonl):
  {"query": "...", "title": "...", "passage": "...", "teacher_probs": {"0": 0.93, "1": 0.07}}
  或
  {"query": "...", "title": "...", "passage": "...", "teacher_probs": {"0": 0.1, "1": 0.2, "2": 0.5, "3": 0.2}}

用法:
  accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \\
      --num_processes 8 distill.py \\
      --model_name_or_path /path/to/model \\
      --local_data_path /path/to/distill_data.jsonl \\
      --num_labels 2 \\
      --output_dir distill_output \\
      ...
"""

import os
import json
from dataclasses import dataclass, field
from typing import Dict, List, Any

import torch
import torch.nn.functional as F
import transformers
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    Trainer,
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    TrlParser,
    get_kbit_device_map,
    get_quantization_config,
)


@dataclass
class LocalArguments:
    local_data_path: str = field(
        default=None, metadata={"help": "训练数据 jsonl/parquet 路径"}
    )
    num_labels: int = field(
        default=2, metadata={"help": "标签数量: 2 表示 0/1, 4 表示 0/1/2/3"}
    )
    teacher_prob_key: str = field(
        default="teacher_probs",
        metadata={"help": "jsonl 中 teacher 概率分布的字段名"},
    )
    jsd_weight: float = field(
        default=1.0, metadata={"help": "蒸馏 loss 权重"}
    )
    distill_type: str = field(
        default="soft",
        metadata={
            "help": (
                "蒸馏类型: "
                "'soft'/'jsd' 使用 JS 散度; "
                "'fkl' 使用 forward KL = KL(teacher || student); "
                "'rkl' 使用 reverse KL = KL(student || teacher); "
                "'hard' 将 teacher 概率取 argmax 作为硬标签, 用交叉熵训练"
            ),
            "choices": ["soft", "jsd", "fkl", "rkl", "hard"],
        },
    )
    max_seq_length: int = field(
        default=1024, metadata={"help": "最大序列长度"}
    )


# ── 工具函数 ──────────────────────────────────────────────────────────────

def fullwidth_to_halfwidth(s: str) -> str:
    result = []
    for char in s:
        code = ord(char)
        if code == 0x3000:
            code = 0x20
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        result.append(chr(code))
    return ''.join(result)


def get_template(query, title, passage, num_labels):
    doc = title + passage
    if num_labels == 2:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query. "
            "Your task is to score the relevance between the Query and the Doc: "
            "0 means irrelevant, and 1 means relevant."
        )
    else:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query. "
            "Your task is to score the relevance between the Query and the Doc: "
            "0 means irrelevant, 1 means slightly relevant, "
            "2 means partially relevant, and 3 means highly relevant."
        )
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )


# ── Trainer ───────────────────────────────────────────────────────────────

class JSDistillationTrainer(Trainer):
    def __init__(self, *args, target_token_ids: List[int], jsd_weight: float = 1.0,
                 distill_type: str = "soft", **kwargs):
        super().__init__(*args, **kwargs)
        self.target_token_ids = torch.tensor(target_token_ids)
        self.jsd_weight = jsd_weight
        valid_types = ("soft", "jsd", "fkl", "rkl", "hard")
        assert distill_type in valid_types, f"invalid distill_type: {distill_type}"
        # 'soft' 视为 'jsd' 的别名，保持向后兼容
        self.distill_type = "jsd" if distill_type == "soft" else distill_type

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        teacher_probs = inputs.pop("teacher_probs")
        student_outputs = model(**inputs)

        # 取最后一个 token 位置的 logits（即生成位置）
        last_logits = student_outputs.logits[:, -1, :]

        # 提取目标 token 上的 logits / 概率
        target_tokens = self.target_token_ids.to(last_logits.device)
        if target_tokens.dim() == 1:
            target_tokens = target_tokens.unsqueeze(0)
        target_tokens = target_tokens.expand(last_logits.size(0), -1)
        target_logits = torch.gather(last_logits, 1, target_tokens)  # [B, num_labels]

        if self.distill_type == "hard":
            # 硬标签蒸馏: teacher 概率 argmax 作为类别标签, 仅在 target tokens 上做 CE
            hard_labels = teacher_probs.argmax(dim=-1)  # [B]
            loss = F.cross_entropy(target_logits, hard_labels)
        else:
            # 软标签系列: 需要 student 在 target tokens 上的概率分布
            # 在全词表 softmax 后 gather, 保证归一化正确
            student_all_probs = F.softmax(last_logits, dim=-1)
            student_probs = torch.gather(student_all_probs, 1, target_tokens)

            epsilon = 1e-8
            p_student = student_probs.clamp(min=epsilon, max=1 - epsilon)
            q_teacher = teacher_probs.clamp(min=epsilon, max=1 - epsilon)

            # F.kl_div(input=log(Q), target=P, reduction='batchmean') = KL(P || Q)
            if self.distill_type == "jsd":
                # JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = 0.5*(P+Q)
                m = 0.5 * (p_student + q_teacher)
                loss_p_m = F.kl_div(torch.log(m), p_student, reduction='batchmean')
                loss_q_m = F.kl_div(torch.log(m), q_teacher, reduction='batchmean')
                loss = 0.5 * (loss_p_m + loss_q_m)
            elif self.distill_type == "fkl":
                # Forward KL: KL(teacher || student), 让 student 覆盖 teacher 所有 mode
                loss = F.kl_div(torch.log(p_student), q_teacher, reduction='batchmean')
            elif self.distill_type == "rkl":
                # Reverse KL: KL(student || teacher), mode-seeking, student 更"尖锐"
                loss = F.kl_div(torch.log(q_teacher), p_student, reduction='batchmean')
            else:
                raise ValueError(f"unknown distill_type: {self.distill_type}")

        loss = self.jsd_weight * loss
        return (loss, student_outputs) if return_outputs else loss


# ── DataCollator ──────────────────────────────────────────────────────────

@dataclass
class DistillationDataCollator:
    tokenizer: PreTrainedTokenizerBase
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.tokenizer.pad(
            [{k: v for k, v in f.items() if k != "teacher_probs"} for f in features],
            return_tensors=self.return_tensors,
            padding=True,
            padding_side="left",
        )
        teacher_probs = [f["teacher_probs"] for f in features]
        batch["teacher_probs"] = torch.tensor(teacher_probs, dtype=torch.float)
        return batch


# ── 主函数 ────────────────────────────────────────────────────────────────

def main(script_args, training_args, model_config, local_args):
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.label_names = ["teacher_probs"]
    transformers.set_seed(training_args.seed)

    # 模型
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=True,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(model_config.model_name_or_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=True, use_fast=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 目标 token IDs
    label_strs = [str(i) for i in range(local_args.num_labels)]
    target_token_ids = []
    for s in label_strs:
        ids = tokenizer.encode(s, add_special_tokens=False)
        assert len(ids) == 1, f"数字 '{s}' 编码为多个 token: {ids}"
        target_token_ids.append(ids[0])

    if training_args.process_index == 0:
        print(f"[INFO] num_labels={local_args.num_labels}, target_tokens={label_strs} → ids={target_token_ids}")
        print(f"[INFO] distill_type={local_args.distill_type}")

    # 数据集（支持 parquet 和 jsonl）
    data_path = local_args.local_data_path
    if data_path.endswith(".parquet"):
        train_dataset = load_dataset("parquet", data_files=data_path, split="train")
    else:
        train_dataset = load_dataset("json", data_files=data_path)['train']

    def process_dataset(examples):
        all_teacher_probs = []
        texts = []
        for i in range(len(examples["query"])):
            query = fullwidth_to_halfwidth(str(examples["query"][i] or "")).lower()
            title = fullwidth_to_halfwidth(str(examples["title"][i] or "")).lower()
            passage = fullwidth_to_halfwidth(str(examples["passage"][i] or "")).lower()

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": get_template(query, title, passage, local_args.num_labels)},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

            # 提取 teacher 概率
            probs_dict = examples[local_args.teacher_prob_key][i]
            if isinstance(probs_dict, str):
                probs_dict = json.loads(probs_dict)
            teacher_probs = [float(probs_dict[s]) for s in label_strs]
            all_teacher_probs.append(teacher_probs)

        model_inputs = tokenizer(
            texts, return_token_type_ids=False,
            max_length=getattr(local_args, "max_seq_length", 1024),
            truncation=True, padding=False,
        )
        model_inputs["teacher_probs"] = all_teacher_probs
        return model_inputs

    train_dataset = train_dataset.map(
        process_dataset,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=min(64, os.cpu_count() or 1),
    ).shuffle(seed=training_args.seed)

    if training_args.process_index == 0:
        print(f"[INFO] 训练数据量: {len(train_dataset)}")

    # Trainer
    data_collator = DistillationDataCollator(tokenizer=tokenizer)
    trainer = JSDistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        target_token_ids=target_token_ids,
        jsd_weight=local_args.jsd_weight,
        distill_type=local_args.distill_type,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)


# ── 入口 ──────────────────────────────────────────────────────────────────

def make_parser():
    return TrlParser((ScriptArguments, SFTConfig, ModelConfig, LocalArguments))


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_config, local_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_config, local_args)
