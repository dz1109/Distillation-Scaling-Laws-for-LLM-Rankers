"""
eval.py
=======
对多个 LLM checkpoint 并行评测（每个 ckpt 分配一张 GPU），输出汇总结果。

用法:
  python eval.py \
      --ckpt_dir ./distill_output \
      --data_path test.jsonl \
      --output eval_results.json \
      --num_labels 2 \
      --batch_size 64

  # 指定多个 ckpt
  python eval.py \
      --ckpt_paths ckpt1 ckpt2 ckpt3 \
      --data_path test.jsonl \
      --output eval_results.json
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


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


def get_template(query, title, passage, num_labels=2):
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
    与 template 使用的 num_labels 无关, 允许"模型只输出 0/1 但测试集是 0/1/2/3"这种解耦评测。
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
      - mean_entropy: 相关性分布的平均熵 H(p), 越小越"尖锐"
      - mean_maxprob: max softmax 概率的均值, 越大越自信
      - temp_sensitivity: 用 T=2 软化 logits 后熵的增量, 越大代表越易被软化 → 越极化
      - (二分类) mean_margin / var_margin: logit_1 - logit_0 的均值与方差
    - rel_probs: shape (N, C), 对 C 个 label token 做的 softmax
    - margins:   shape (N,), 仅当 C=2 且传入时使用
    """
    eps = 1e-12
    rel_probs = np.asarray(rel_probs, dtype=np.float64)
    p = np.clip(rel_probs, eps, 1.0)
    ent = float(-np.sum(p * np.log(p), axis=-1).mean())
    maxprob = float(rel_probs.max(axis=-1).mean())

    # 温度敏感度: 从概率反推 logits, 除以 T=2 重新 softmax, 看熵上升多少
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


# ── 单 ckpt 评测（在指定 GPU 上运行）────────────────────────────────────────

def eval_single_ckpt(ckpt_path, data_path, gpu_id, num_labels, batch_size, max_length,
                     query_key, label_key, eval_label_max):
    """
    在指定 GPU 上对单个 checkpoint 进行推理 + 评测。
    独立进程中运行，返回评测结果 dict。

    - num_labels: 模型输出空间大小 (决定 template 与 digit token ids), 与训练一致。
    - eval_label_max: 数据集 label 的最大值 (仅用于 ECE 归一化), 与 num_labels 解耦。
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # digit token ids
    label_strs = [str(d) for d in range(num_labels)]
    digit_ids = [tokenizer.encode(s, add_special_tokens=False)[0] for s in label_strs]

    # 加载数据
    data_rows = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data_rows.append(json.loads(line))

    # 预处理
    dataset = load_dataset("json", data_files=data_path)['train']

    def process_messages(examples):
        messages = []
        for i in range(len(examples["query"])):
            query = fullwidth_to_halfwidth(str(examples.get("query", [""])[i] or "")).lower()
            title = fullwidth_to_halfwidth(str(examples.get("title", [""])[i] or "")).lower()
            passage = fullwidth_to_halfwidth(str(examples.get("passage", [""])[i] or "")).lower()
            messages.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": get_template(query, title, passage, num_labels)},
            ])
        return {"messages": messages}

    processed_dataset = dataset.map(
        process_messages, batched=True, remove_columns=dataset.column_names,
    )
    batched_dataset = processed_dataset.batch(batch_size)

    # 推理
    weighted_scores = []
    rel_probs_all = []   # (N, num_labels) 相关性分布 (只对 label token 做 softmax)
    margins_all = []     # (N,) 二分类时的 logit_1 - logit_0
    with torch.no_grad():
        for batch in tqdm(batched_dataset, desc=f"[GPU{gpu_id}] {os.path.basename(ckpt_path)}"):
            messages = batch['messages']
            texts = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            model_inputs = tokenizer(
                texts, return_tensors="pt", padding=True,
                return_token_type_ids=False, max_length=max_length, truncation=True,
            ).to(device)

            outputs = model.generate(
                **model_inputs,
                max_new_tokens=1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

            for batch_idx in range(len(messages)):
                step_logits = outputs.scores[0][batch_idx]
                # 全词表 softmax → 加权 score (原逻辑, 供 NDCG/ECE 使用)
                probs = torch.softmax(step_logits, dim=-1)
                weighted_score = sum(
                    int(d) * probs[digit_ids[i]].item() for i, d in enumerate(label_strs)
                ) / (num_labels - 1)
                weighted_scores.append(weighted_score)

                # 极化度: 只在 label token 上做 softmax
                rel_logits = torch.tensor([step_logits[tid].item() for tid in digit_ids])
                rel_p = torch.softmax(rel_logits, dim=-1).numpy()
                rel_probs_all.append(rel_p)
                if num_labels == 2:
                    margins_all.append(
                        float(step_logits[digit_ids[1]].item() - step_logits[digit_ids[0]].item())
                    )

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

    # ECE: 使用全体样本（跨 query）计算
    all_scores = weighted_scores
    all_labels = [float(row[label_key]) for row in data_rows]
    ece = expected_calibration_error(all_scores, all_labels, n_bins=10, label_max=eval_label_max)

    # Polarization / sharpness 指标
    polar = polarization_metrics(
        rel_probs_all,
        margins=margins_all if num_labels == 2 else None,
    )

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
    parser = argparse.ArgumentParser(description="多 checkpoint 并行评测")
    parser.add_argument("--ckpt_dir", default=None,
                        help="checkpoint 目录，自动查找 checkpoint-* 子目录")
    parser.add_argument("--ckpt_paths", nargs="+", default=None,
                        help="显式指定多个 checkpoint 路径")
    parser.add_argument("--data_path", required=True, help="测试数据 jsonl 路径")
    parser.add_argument("--output", "-o", required=True, help="评测结果输出 JSON 路径")
    parser.add_argument("--num_labels", type=int, default=2, help="模型输出标签数 (2 或 4), 决定 template 与 digit token")
    parser.add_argument("--eval_label_max", type=int, default=None,
                        help="测试集 label 最大值 (仅用于 ECE 归一化), 与 num_labels 解耦; "
                             "默认 = num_labels - 1 (例如 num_labels=2 → 1; num_labels=4 → 3)")
    parser.add_argument("--batch_size", type=int, default=64, help="推理 batch size")
    parser.add_argument("--max_length", type=int, default=1024, help="最大输入长度")
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

    # 确定 GPU 数
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

    # ECE label 归一化上限: 默认 num_labels - 1, 允许通过 --eval_label_max 覆盖
    eval_label_max = args.eval_label_max if args.eval_label_max is not None else (args.num_labels - 1)
    print(f"[INFO] num_labels={args.num_labels} (template), eval_label_max={eval_label_max} (ECE)")
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
