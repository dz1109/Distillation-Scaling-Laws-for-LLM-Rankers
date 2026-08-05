"""
bert_eval.py
============
对多个 BERT Cross-Encoder checkpoint 并行评测（每个 ckpt 分配一张 GPU），
输出 NDCG@5 / NDCG@10 / ECE 以及 polarization 指标
(mean_entropy / mean_maxprob / temp_sensitivity / mean_margin / var_margin)。

用法:
  python bert_eval.py \
      --ckpt_dir ./bert_distill_output \
      --data_path test.jsonl \
      --output bert_eval_results.json \
      --num_labels 2 \
      --eval_label_max 3 \
      --batch_size 128

  # 显式指定多个 ckpt
  python bert_eval.py \
      --ckpt_paths ckpt1 ckpt2 ckpt3 \
      --data_path test.jsonl \
      --output bert_eval_results.json
"""

import os
import json
import glob
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ── 工具函数 ──────────────────────────────────────────────────────────────

def dcg_at_k(scores, k):
    scores = np.array(scores[:k], dtype=np.float64)
    gains = (2 ** scores - 1)
    discounts = np.log2(np.arange(len(scores)) + 2)
    return np.sum(gains / discounts)


def ndcg_at_k(relevances, scores, k):
    order = np.argsort(-np.array(scores))
    sorted_rels = [relevances[i] for i in order]
    actual_dcg = dcg_at_k(sorted_rels, k)
    ideal_order = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal_order, k)
    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def expected_calibration_error(scores, labels, n_bins=10, label_max=1):
    """
    Expected Calibration Error (ECE)
    - scores: 模型输出的加权分数（已归一化到 [0, 1]）
    - labels: 真实标签 (0 ~ label_max)
    - label_max: 数据集 label 的最大值 (用于归一化到 [0, 1])
        二分类数据: label_max=1, 保持原值 (0/1)
        四档数据 (0/1/2/3): label_max=3, labels/3 归一化到 [0,1]
    与模型的 num_labels 解耦, 允许"模型只输出 0/1 但测试集是 0/1/2/3"这种评测。
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if label_max > 1:
        labels = labels / float(label_max)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(scores)
    if total == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_conf = float(scores[mask].mean())
        avg_acc = float(labels[mask].mean())
        ece += (count / total) * abs(avg_conf - avg_acc)
    return ece


def polarization_metrics(rel_probs, margins=None):
    """
    Polarization / sharpness 指标:
      - mean_entropy: 相关性分布的平均熵 H(p), 越小越尖锐
      - mean_maxprob: max softmax 概率的均值, 越大越自信
      - temp_sensitivity: 用 T=2 软化 logits 后熵的增量, 越大代表越易被软化 → 越极化
      - (二分类) mean_margin / var_margin: logit_1 - logit_0 的均值与方差
    """
    eps = 1e-12
    rel_probs = np.asarray(rel_probs, dtype=np.float64)
    p = np.clip(rel_probs, eps, 1.0)
    ent = float(-np.sum(p * np.log(p), axis=-1).mean())
    maxprob = float(rel_probs.max(axis=-1).mean())

    logits_recovered = np.log(np.clip(rel_probs, eps, 1.0))
    soft = logits_recovered / 2.0
    soft = soft - soft.max(axis=-1, keepdims=True)
    soft_p = np.exp(soft)
    soft_p /= soft_p.sum(axis=-1, keepdims=True)
    soft_p_c = np.clip(soft_p, eps, 1.0)
    ent_soft = float(-np.sum(soft_p_c * np.log(soft_p_c), axis=-1).mean())
    temp_sens = ent_soft - ent

    out = {
        "mean_entropy": round(ent, 4),
        "mean_maxprob": round(maxprob, 4),
        "temp_sensitivity": round(temp_sens, 4),
    }
    if margins is not None and len(margins) > 0:
        m = np.asarray(margins, dtype=np.float64)
        out["mean_margin"] = round(float(m.mean()), 4)
        out["var_margin"] = round(float(m.var()), 4)
    return out


# ── Dataset & Collate ─────────────────────────────────────────────────────

class InferDataset(Dataset):
    def __init__(self, data_rows, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        for obj in data_rows:
            query = obj.get("query", "") or ""
            title = obj.get("title", "") or ""
            passage = obj.get("passage", "") or ""
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


# ── 单 ckpt 评测（在指定 GPU 上运行）────────────────────────────────────────

def eval_single_ckpt(ckpt_path, data_path, gpu_id, num_labels, batch_size, max_length,
                     query_key, label_key, eval_label_max):
    """
    在指定 GPU 上对单个 BERT checkpoint 进行推理 + 评测。

    - num_labels: 模型分类头维度, 决定 softmax 与加权得分, 与训练一致。
    - eval_label_max: 数据集 label 的最大值 (仅用于 ECE 归一化), 与 num_labels 解耦。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        ckpt_path, num_labels=num_labels,
    ).to(device)
    model.eval()

    data_rows = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data_rows.append(json.loads(line))

    dataset = InferDataset(data_rows, tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=2,
    )

    # 推理
    weighted_scores = [0.0] * len(dataset)
    rel_probs_all = [None] * len(dataset)   # (num_labels,) per sample
    margins_all = [0.0] * len(dataset) if num_labels == 2 else None
    with torch.no_grad():
        for batch, global_indices in tqdm(
            loader, desc=f"[GPU{gpu_id}] {os.path.basename(ckpt_path)}"
        ):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits.cpu()               # [B, num_labels]
            probs = torch.softmax(logits, dim=-1)       # [B, num_labels]
            for i, gidx in enumerate(global_indices):
                prob_list = probs[i].tolist()
                weighted = sum(j * p for j, p in enumerate(prob_list)) / (num_labels - 1)
                weighted_scores[gidx] = weighted
                # BERT 的 logits 就是相关性 logits, softmax 后即是 rel_probs
                rel_probs_all[gidx] = np.asarray(prob_list, dtype=np.float64)
                if num_labels == 2:
                    margins_all[gidx] = float(logits[i, 1].item() - logits[i, 0].item())

    # 计算 NDCG
    query_groups = defaultdict(lambda: {'relevances': [], 'scores': []})
    for idx, row in enumerate(data_rows):
        qid = str(row[query_key])
        rel = float(row[label_key])
        query_groups[qid]['relevances'].append(rel)
        query_groups[qid]['scores'].append(weighted_scores[idx])

    ndcg5_list = []
    ndcg10_list = []
    for qid, group in query_groups.items():
        ndcg5_list.append(ndcg_at_k(group['relevances'], group['scores'], 5))
        ndcg10_list.append(ndcg_at_k(group['relevances'], group['scores'], 10))

    # ECE: 跨全体样本计算
    all_labels = [float(row[label_key]) for row in data_rows]
    ece = expected_calibration_error(weighted_scores, all_labels, n_bins=10,
                                     label_max=eval_label_max)

    # Polarization
    polar = polarization_metrics(rel_probs_all, margins=margins_all)

    result = {
        "checkpoint": ckpt_path,
        "checkpoint_name": os.path.basename(ckpt_path),
        "gpu_id": gpu_id,
        "num_queries": len(query_groups),
        "ndcg@5": round(float(np.mean(ndcg5_list)), 4),
        "ndcg@10": round(float(np.mean(ndcg10_list)), 4),
        "ece": round(float(ece), 4),
    }
    result.update(polar)
    return result


# ── 主函数 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BERT 多 checkpoint 并行评测")
    parser.add_argument("--ckpt_dir", default=None,
                        help="checkpoint 目录，自动查找 checkpoint-* 子目录")
    parser.add_argument("--ckpt_paths", nargs="+", default=None,
                        help="显式指定多个 checkpoint 路径")
    parser.add_argument("--data_path", required=True, help="测试数据 jsonl 路径")
    parser.add_argument("--output", "-o", required=True, help="评测结果输出 JSON 路径")
    parser.add_argument("--num_labels", type=int, default=2, help="模型分类头维度 (2 或 4)")
    parser.add_argument("--eval_label_max", type=int, default=None,
                        help="测试集 label 最大值 (仅用于 ECE 归一化), 与 num_labels 解耦; "
                             "默认 = num_labels - 1")
    parser.add_argument("--batch_size", type=int, default=128, help="推理 batch size")
    parser.add_argument("--max_length", type=int, default=512, help="最大输入长度")
    parser.add_argument("--query_key", default=None, help="query id 字段名 (自动检测)")
    parser.add_argument("--label_key", default=None, help="label 字段名 (自动检测)")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="使用 GPU 数量 (默认: 可用 GPU 数 或 ckpt 数取小)")
    args = parser.parse_args()

    # 收集 checkpoint 路径
    ckpt_paths = []
    if args.ckpt_paths:
        ckpt_paths = args.ckpt_paths
    elif args.ckpt_dir:
        ckpt_paths = sorted(glob.glob(os.path.join(args.ckpt_dir, "checkpoint-*")))
        if not ckpt_paths:
            ckpt_paths = [args.ckpt_dir]
    else:
        parser.error("必须指定 --ckpt_dir 或 --ckpt_paths")

    print(f"[INFO] 待评测 checkpoint 数: {len(ckpt_paths)}")
    for p in ckpt_paths:
        print(f"  - {p}")

    available_gpus = torch.cuda.device_count()
    num_gpus = args.num_gpus or min(available_gpus, len(ckpt_paths))
    print(f"[INFO] 可用 GPU: {available_gpus}, 使用: {num_gpus}")

    # 自动检测字段名
    with open(args.data_path, 'r') as f:
        sample = json.loads(f.readline())
    query_key = args.query_key
    if query_key is None:
        for candidate in ['query_id', 'qid', 'query']:
            if candidate in sample:
                query_key = candidate
                break
    label_key = args.label_key
    if label_key is None:
        for candidate in ['label', 'relevance', 'rel', 'score', 'rating']:
            if candidate in sample:
                label_key = candidate
                break
    print(f"[INFO] query_key={query_key}, label_key={label_key}")

    eval_label_max = args.eval_label_max if args.eval_label_max is not None else (args.num_labels - 1)
    print(f"[INFO] num_labels={args.num_labels} (model), eval_label_max={eval_label_max} (ECE)")
    print()

    # 并行评测: 每个 ckpt 分配一张 GPU
    all_results = []

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = {}
        for i, ckpt_path in enumerate(ckpt_paths):
            gpu_id = i % num_gpus
            future = executor.submit(
                eval_single_ckpt,
                ckpt_path=ckpt_path,
                data_path=args.data_path,
                gpu_id=gpu_id,
                num_labels=args.num_labels,
                batch_size=args.batch_size,
                max_length=args.max_length,
                query_key=query_key,
                label_key=label_key,
                eval_label_max=eval_label_max,
            )
            futures[future] = ckpt_path

        for future in as_completed(futures):
            ckpt_path = futures[future]
            try:
                result = future.result()
                all_results.append(result)
                extra = ""
                if "mean_margin" in result:
                    extra = f", margin={result['mean_margin']:.3f}, var={result['var_margin']:.3f}"
                print(f"[DONE] {result['checkpoint_name']}: "
                      f"NDCG@5={result['ndcg@5']:.4f}, NDCG@10={result['ndcg@10']:.4f}, "
                      f"ECE={result['ece']:.4f}, H={result['mean_entropy']:.4f}, "
                      f"maxp={result['mean_maxprob']:.4f}, tempSens={result['temp_sensitivity']:.4f}"
                      f"{extra}")
            except Exception as e:
                print(f"[ERROR] {os.path.basename(ckpt_path)}: {e}")
                all_results.append({
                    "checkpoint": ckpt_path,
                    "checkpoint_name": os.path.basename(ckpt_path),
                    "error": str(e),
                })

    # 按 checkpoint 名称排序
    all_results.sort(key=lambda x: x.get("checkpoint_name", ""))

    # 输出结果
    output = {
        "data_path": args.data_path,
        "num_labels": args.num_labels,
        "eval_label_max": eval_label_max,
        "results": all_results,
    }

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"[DONE] 结果已保存: {args.output}")
    print(f"{'='*60}")
    print(f"\n{'checkpoint':<40s} {'NDCG@5':>8s} {'NDCG@10':>8s} {'ECE':>8s} "
          f"{'H':>7s} {'maxp':>7s} {'tempS':>7s} {'margin':>8s} {'var_m':>8s}")
    print("-" * 108)
    for r in all_results:
        if "error" in r:
            print(f"{r['checkpoint_name']:<40s} {'ERROR':>8s}")
        else:
            margin = r.get("mean_margin", None)
            var_m = r.get("var_margin", None)
            margin_s = f"{margin:>8.3f}" if margin is not None else f"{'-':>8s}"
            var_s = f"{var_m:>8.3f}" if var_m is not None else f"{'-':>8s}"
            print(f"{r['checkpoint_name']:<40s} {r['ndcg@5']:>8.4f} {r['ndcg@10']:>8.4f} "
                  f"{r['ece']:>8.4f} {r['mean_entropy']:>7.4f} {r['mean_maxprob']:>7.4f} "
                  f"{r['temp_sensitivity']:>7.4f} {margin_s} {var_s}")


if __name__ == "__main__":
    main()
