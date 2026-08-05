"""
infer_model.py
==============
多卡并行推理脚本。每张卡处理数据的一个 shard，最终由 rank0 汇总结果。

用法:
  torchrun --nproc_per_node=8 infer_model.py \
      --model_name_or_path /path/to/model \
      --local_data_path /path/to/data.jsonl \
      --infer_save_path /path/to/output.tsv \
      --batch_size 64
"""

import os
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
    num_labels: int = field(default=2, metadata={"help": "标签数量: 2 表示 0/1, 4 表示 0/1/2/3"})


def setup_distributed():
    """初始化分布式环境，返回 (rank, world_size)。"""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_template_sft(query, title, passage, num_labels=2):
    doc = title + passage
    if num_labels == 2:
        instruction = "Given a web search query, retrieve relevant passages that answer the query. Your task is to score the relevance between the Query and the Doc: 0 means irrelevant, and 1 means relevant."
    else:
        instruction = "Given a web search query, retrieve relevant passages that answer the query. Your task is to score the relevance between the Query and the Doc: 0 means irrelevant, 1 means slightly relevant, 2 means partially relevant, and 3 means highly relevant."
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )


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


def process_both(examples, num_labels=2):
    messages = []
    for i in range(len(examples["query"])):
        query = fullwidth_to_halfwidth(examples["query"][i]).lower()
        title = fullwidth_to_halfwidth(examples["title"][i]).lower()
        passage = fullwidth_to_halfwidth(examples["passage"][i]).lower()
        messages.append([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": get_template_sft(query, title, passage, num_labels)},
        ])
    return {"messages": messages}


def main():
    parser = HfArgumentParser((InferArguments,))
    args = parser.parse_args_into_dataclasses()[0]

    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"[INFO] world_size={world_size}, model={args.model_name_or_path}")
        print(f"[INFO] data={args.local_data_path}, batch_size={args.batch_size}")
        print(f"[INFO] num_labels={args.num_labels} → 标签范围 0~{args.num_labels - 1}")

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

    # 加载数据并分片（保留全局索引以便 join 回原始数据）
    full_dataset = load_dataset("json", data_files=args.local_data_path)['train']
    full_dataset = full_dataset.add_column("__global_idx__", list(range(len(full_dataset))))
    shard_dataset = full_dataset.shard(num_shards=world_size, index=rank)

    if rank == 0:
        print(f"[INFO] 总数据量={len(full_dataset)}, 每卡约={len(shard_dataset)}")

    # 保存 shard 的全局索引
    shard_indices = shard_dataset["__global_idx__"]

    shard_dataset = shard_dataset.map(
        lambda examples: process_both(examples, num_labels=args.num_labels),
        batched=True,
        remove_columns=shard_dataset.column_names,
    )
    batched_dataset = shard_dataset.batch(args.batch_size)

    # digit token ids (根据 num_labels 决定)
    label_range = list(range(args.num_labels))
    digit_ids = [tokenizer.encode(str(d), add_special_tokens=False) for d in label_range]
    for d_id in digit_ids:
        assert len(d_id) == 1, f"数字{d_id}必须编码为单个token"
    digit_ids = [d_id[0] for d_id in digit_ids]
    label_strs = [str(d) for d in label_range]

    # 推理
    all_digit_scores = []
    weighted_scores = []
    output_texts = []

    with torch.no_grad():
        iterator = tqdm(batched_dataset, desc=f"[rank{rank}]", disable=(rank != 0))
        for batch in iterator:
            messages = batch['messages']
            texts = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            model_inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                return_token_type_ids=False,
                max_length=args.max_length,
                truncation=True,
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

                # 加权得分: sum(label_value * prob) / (num_labels - 1) → 归一化到 [0, 1]
                weighted_score = sum(int(d) * digit_probs[d] for d in label_strs) / (args.num_labels - 1)

                all_digit_scores.append(digit_probs)
                weighted_scores.append(weighted_score)
            output_texts.extend(generated_texts)

    # 汇总各卡结果
    if world_size > 1:
        tmp_path = f"{args.infer_save_path}.rank{rank}.tmp"
        df = pd.DataFrame({
            'global_idx': shard_indices,
            'all_digit_scores': all_digit_scores,
            'output_texts': output_texts,
            'weighted_scores': weighted_scores,
        })
        df.to_csv(tmp_path, index=False, sep="\t")
        dist.barrier()

        if rank == 0:
            dfs = []
            for r in range(world_size):
                tmp = f"{args.infer_save_path}.rank{r}.tmp"
                dfs.append(pd.read_csv(tmp, sep="\t"))
                os.remove(tmp)
            merged = pd.concat(dfs, ignore_index=True)
            merged = merged.sort_values("global_idx").reset_index(drop=True)
            merged.to_csv(args.infer_save_path, index=False, sep="\t")
            print(f"[INFO] 结果已保存: {args.infer_save_path} ({len(merged)} 行)")
        dist.barrier()
        dist.destroy_process_group()
    else:
        df = pd.DataFrame({
            'global_idx': shard_indices,
            'all_digit_scores': all_digit_scores,
            'output_texts': output_texts,
            'weighted_scores': weighted_scores,
        })
        df = df.sort_values("global_idx").reset_index(drop=True)
        df.to_csv(args.infer_save_path, index=False, sep="\t")
        print(f"[INFO] 结果已保存: {args.infer_save_path} ({len(df)} 行)")


if __name__ == "__main__":
    main()
