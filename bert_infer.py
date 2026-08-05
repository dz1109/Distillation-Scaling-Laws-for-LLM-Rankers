"""
bert_infer.py
=============
BERT Cross-Encoder 多卡推理脚本。

输出格式与 infer_model.py 对齐 (TSV)，可直接用 eval_ndcg.py 评测。

用法:
  torchrun --nproc_per_node=8 bert_infer.py \
      --model_path ./output/bert_distill \
      --data_path /path/to/test.jsonl \
      --output_path ./bert_infer_results.tsv \
      --batch_size 128 \
      --num_labels 4
"""

import os
import sys
import json
import argparse

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


class InferDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line.strip())
                query = obj.get("query", "")
                title = obj.get("title", "")
                passage = obj.get("passage", "")
                doc = (title + " " + passage).strip()
                self.samples.append((query, doc))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query, doc = self.samples[idx]
        encoded = self.tokenizer(
            query, doc,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_token_type_ids=True,
        )
        encoded["global_idx"] = idx
        return encoded


def collate_fn(batch, tokenizer):
    global_indices = [b.pop("global_idx") for b in batch]
    padded = tokenizer.pad(batch, padding=True, return_tensors="pt")
    return padded, global_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--num_labels", type=int, default=4)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path, num_labels=args.num_labels
    ).to(device)
    model.eval()

    dataset = InferDataset(args.data_path, tokenizer, max_length=args.max_length)

    # 按 rank 分片
    indices = list(range(rank, len(dataset), world_size))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
    )

    local_results = []
    with torch.no_grad():
        for batch, global_indices in tqdm(loader, desc=f"[Rank {rank}]", disable=rank != 0):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1).cpu()

            for i, gidx in enumerate(global_indices):
                prob_list = probs[i].tolist()
                scores_str = json.dumps({str(j): round(p, 6) for j, p in enumerate(prob_list)})
                weighted = sum(j * p for j, p in enumerate(prob_list)) / (args.num_labels - 1)
                local_results.append((gidx, scores_str, round(weighted, 6)))

    # Rank 0 收集所有结果
    if world_size > 1:
        all_results_list = [None] * world_size
        dist.all_gather_object(all_results_list, local_results)
    else:
        all_results_list = [local_results]

    if rank == 0:
        merged = []
        for r in all_results_list:
            merged.extend(r)
        merged.sort(key=lambda x: x[0])

        with open(args.output_path, 'w', encoding='utf-8') as f:
            f.write("global_idx\tall_digit_scores\tweighted_scores\n")
            for gidx, scores_str, weighted in merged:
                f.write(f"{gidx}\t{scores_str}\t{weighted}\n")
        print(f"[INFO] Results saved: {args.output_path} ({len(merged)} rows)")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
