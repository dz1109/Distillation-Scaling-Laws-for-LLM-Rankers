"""
infer_resume.py
===============
断点续跑推理：只跑缺失 rank 的数据分片，然后合并所有 rank 结果。

检测已有的 .rank*.tmp 文件，找出缺失的 rank，只对缺失部分进行推理。
最后合并所有 rank 的结果为完整 TSV。

用法:
  torchrun --nproc_per_node=5 infer_resume.py \
      --model_name_or_path /path/to/model \
      --local_data_path /path/to/data.jsonl \
      --infer_save_path /path/to/output.tsv \
      --original_world_size 8 \
      --batch_size 64 \
      --num_labels 4
"""

import os
import json
from dataclasses import dataclass, field

import torch
import numpy as np
import pandas as pd
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, AutoModelForCausalLM
from datasets import load_dataset


@dataclass
class InferArguments:
    model_name_or_path: str = field(metadata={"help": "模型路径"})
    local_data_path: str = field(metadata={"help": "输入 jsonl 数据路径"})
    infer_save_path: str = field(default="./infer_results.tsv", metadata={"help": "输出结果路径"})
    batch_size: int = field(default=64, metadata={"help": "每卡 batch size"})
    max_length: int = field(default=1024, metadata={"help": "最大输入长度"})
    num_labels: int = field(default=4, metadata={"help": "标签数量: 2 或 4"})
    original_world_size: int = field(default=8, metadata={"help": "原始推理时的 GPU 数量"})
    merge_only: bool = field(default=False, metadata={"help": "只合并已有 tmp 文件，不推理"})


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


def get_template_sft(query, title, passage, num_labels=2):
    doc = title + passage
    if num_labels == 2:
        instruction = "Given a web search query, retrieve relevant passages that answer the query. Your task is to score the relevance between the Query and the Doc: 0 means irrelevant, and 1 means relevant."
    else:
        instruction = "Given a web search query, retrieve relevant passages that answer the query. Your task is to score the relevance between the Query and the Doc: 0 means irrelevant, 1 means slightly relevant, 2 means partially relevant, and 3 means highly relevant."
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )


def find_missing_ranks(save_path, original_world_size):
    """找出缺失的 rank 分片。"""
    existing = []
    missing = []
    for r in range(original_world_size):
        tmp = f"{save_path}.rank{r}.tmp"
        if os.path.exists(tmp) and os.path.getsize(tmp) > 100:
            existing.append(r)
        else:
            missing.append(r)
    return existing, missing


def merge_results(save_path, original_world_size):
    """合并所有 rank 的 tmp 文件。"""
    dfs = []
    for r in range(original_world_size):
        tmp = f"{save_path}.rank{r}.tmp"
        if not os.path.exists(tmp):
            print(f"[WARN] 缺少 {tmp}，跳过")
            continue
        df = pd.read_csv(tmp, sep="\t")
        dfs.append(df)
        print(f"  rank{r}: {len(df)} 行")

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values("global_idx").reset_index(drop=True)
    merged.to_csv(save_path, index=False, sep="\t")
    print(f"[INFO] 合并完成: {save_path} ({len(merged)} 行)")


def main():
    parser = HfArgumentParser((InferArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    # 检测缺失 rank
    existing_ranks, missing_ranks = find_missing_ranks(args.infer_save_path, args.original_world_size)
    print(f"[INFO] 原始 world_size={args.original_world_size}")
    print(f"[INFO] 已有 rank: {existing_ranks}")
    print(f"[INFO] 缺失 rank: {missing_ranks}")

    if args.merge_only or not missing_ranks:
        if not missing_ranks:
            print("[INFO] 所有 rank 已完成，直接合并")
        merge_results(args.infer_save_path, args.original_world_size)
        return

    # 分布式初始化（用于补跑缺失 rank）
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = dist.get_rank()
        local_world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        local_world_size = 1

    if local_world_size < len(missing_ranks):
        if local_rank == 0:
            print(f"[ERROR] 需要 {len(missing_ranks)} 张卡补跑缺失 rank，但只有 {local_world_size} 张")
        if dist.is_initialized():
            dist.destroy_process_group()
        return

    # 每张当前卡负责一个缺失的 rank
    if local_rank >= len(missing_ranks):
        print(f"[rank{local_rank}] 无任务，等待")
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    target_rank = missing_ranks[local_rank]
    device = torch.device(f"cuda:{local_rank}")
    print(f"[local_rank={local_rank}] 负责补跑 original_rank={target_rank}")

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据，取对应 shard
    full_dataset = load_dataset("json", data_files=args.local_data_path)['train']
    full_dataset = full_dataset.add_column("__global_idx__", list(range(len(full_dataset))))
    shard_dataset = full_dataset.shard(num_shards=args.original_world_size, index=target_rank)

    print(f"[rank{target_rank}] shard 大小: {len(shard_dataset)}")
    shard_indices = shard_dataset["__global_idx__"]

    def process_both(examples):
        messages = []
        for i in range(len(examples["query"])):
            query = fullwidth_to_halfwidth(examples["query"][i]).lower()
            title = fullwidth_to_halfwidth(examples["title"][i]).lower()
            passage = fullwidth_to_halfwidth(examples["passage"][i]).lower()
            messages.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": get_template_sft(query, title, passage, args.num_labels)},
            ])
        return {"messages": messages}

    shard_dataset = shard_dataset.map(
        process_both, batched=True, remove_columns=shard_dataset.column_names,
    )
    batched_dataset = shard_dataset.batch(args.batch_size)

    # digit token ids
    label_range = list(range(args.num_labels))
    digit_ids = [tokenizer.encode(str(d), add_special_tokens=False) for d in label_range]
    for d_id in digit_ids:
        assert len(d_id) == 1
    digit_ids = [d_id[0] for d_id in digit_ids]
    label_strs = [str(d) for d in label_range]

    # 推理
    all_digit_scores = []
    weighted_scores = []
    output_texts = []

    with torch.no_grad():
        for batch in tqdm(batched_dataset, desc=f"[补跑 rank{target_rank}]"):
            messages = batch['messages']
            texts = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            model_inputs = tokenizer(
                texts, return_tensors="pt", padding=True,
                return_token_type_ids=False, max_length=args.max_length, truncation=True,
            ).to(device)

            outputs = model.generate(
                **model_inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            input_length = model_inputs.input_ids.shape[1]
            generated_sequences = outputs.sequences[:, input_length:]
            generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

            for batch_idx, generated_text in enumerate(generated_texts):
                step_logits = outputs.scores[0][batch_idx]
                probs = torch.softmax(step_logits, dim=-1)
                digit_probs = {}
                for i, d in enumerate(label_strs):
                    digit_probs[d] = probs[digit_ids[i]].item()
                weighted_score = sum(int(d) * digit_probs[d] for d in label_strs) / (args.num_labels - 1)
                all_digit_scores.append(digit_probs)
                weighted_scores.append(weighted_score)
            output_texts.extend(generated_texts)

    # 保存 tmp
    tmp_path = f"{args.infer_save_path}.rank{target_rank}.tmp"
    df = pd.DataFrame({
        'global_idx': shard_indices,
        'all_digit_scores': all_digit_scores,
        'output_texts': output_texts,
        'weighted_scores': weighted_scores,
    })
    df.to_csv(tmp_path, index=False, sep="\t")
    print(f"[rank{target_rank}] 已保存: {tmp_path} ({len(df)} 行)")

    # 等待所有补跑完成
    if dist.is_initialized():
        dist.barrier()

    # rank0 合并
    if local_rank == 0:
        merge_results(args.infer_save_path, args.original_world_size)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
