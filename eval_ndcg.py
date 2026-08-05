"""
eval_ndcg.py
============
根据推理结果和测试数据的 label 计算 NDCG@5 和 NDCG@10。

推理结果按行号（global_idx）与原始测试数据对齐。

用法:
  python eval_ndcg.py \
      --infer ./infer_results.tsv \
      --data  /path/to/test.jsonl
"""

import argparse
import json
import unicodedata
import numpy as np
import pandas as pd
from collections import defaultdict


def normalize_text(text):
    """全角→半角，大写→小写"""
    text = unicodedata.normalize("NFKC", text)
    return text.lower().strip()


def dcg_at_k(scores, k):
    """计算 DCG@k。"""
    scores = np.array(scores[:k], dtype=np.float64)
    gains = (2 ** scores - 1)
    discounts = np.log2(np.arange(len(scores)) + 2)
    return np.sum(gains / discounts)


def ndcg_at_k(relevances, scores, k):
    """
    计算单个 query 的 NDCG@k。
    relevances: 真实相关性列表
    scores: 模型预测分数列表（用于排序）
    """
    # 按模型分数降序排列
    order = np.argsort(-np.array(scores))
    sorted_rels = [relevances[i] for i in order]

    # 实际 DCG
    actual_dcg = dcg_at_k(sorted_rels, k)

    # 理想 DCG（按真实相关性降序）
    ideal_order = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal_order, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def main():
    parser = argparse.ArgumentParser(description="计算 NDCG@5 和 NDCG@10")
    parser.add_argument("--infer", "-i", required=True, help="推理结果 TSV 路径")
    parser.add_argument("--data", "-d", required=True, help="原始测试数据 JSONL 路径")
    parser.add_argument("--query_key", "-q", default=None,
                        help="JSONL 中 query id 字段名 (自动检测: query_id/qid/query)")
    parser.add_argument("--label_key", "-l", default=None,
                        help="JSONL 中 label 字段名 (自动检测: label/relevance/score)")
    parser.add_argument("--score_col", "-s", default="weighted_scores",
                        help="推理结果中用于排序的列名 (默认: weighted_scores)")
    parser.add_argument("--output", "-o", default=None,
                        help="结果输出文件路径 (默认: 推理结果同目录下 eval_results.json)")
    args = parser.parse_args()

    # 1. 读取推理结果
    infer_df = pd.read_csv(args.infer, sep="\t")
    print(f"推理结果: {len(infer_df)} 行")
    print(f"  列: {list(infer_df.columns)}")

    # 2. 读取原始数据
    data_rows = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data_rows.append(json.loads(line))
    print(f"原始数据: {len(data_rows)} 行")

    assert len(infer_df) == len(data_rows), \
        f"行数不匹配: 推理结果 {len(infer_df)} vs 原始数据 {len(data_rows)}"

    # 3. 自动检测字段名
    sample = data_rows[0]

    # query id
    query_key = args.query_key
    if query_key is None:
        for candidate in ['query_id', 'qid', 'query']:
            if candidate in sample:
                query_key = candidate
                break
        if query_key is None:
            raise ValueError(f"无法自动检测 query 字段，可用字段: {list(sample.keys())}")

    # label
    label_key = args.label_key
    if label_key is None:
        for candidate in ['label', 'relevance', 'rel', 'score', 'rating']:
            if candidate in sample:
                label_key = candidate
                break
        if label_key is None:
            raise ValueError(f"无法自动检测 label 字段，可用字段: {list(sample.keys())}")

    print(f"  query 字段: {query_key}")
    print(f"  label 字段: {label_key}")
    print()

    # 4. 按 global_idx 对齐，按 query 分组
    scores = infer_df[args.score_col].values

    query_groups = defaultdict(lambda: {'relevances': [], 'scores': []})
    for idx, row in enumerate(data_rows):
        qid = normalize_text(str(row[query_key]))
        rel = float(row[label_key])
        pred_score = float(scores[idx])
        query_groups[qid]['relevances'].append(rel)
        query_groups[qid]['scores'].append(pred_score)

    print(f"Query 数: {len(query_groups)}")
    doc_counts = [len(v['relevances']) for v in query_groups.values()]
    print(f"每 query 文档数: min={min(doc_counts)}, max={max(doc_counts)}, avg={np.mean(doc_counts):.1f}")
    print()

    # 5. 计算 NDCG
    ndcg5_list = []
    ndcg10_list = []

    for qid, group in query_groups.items():
        rels = group['relevances']
        preds = group['scores']

        n5 = ndcg_at_k(rels, preds, 5)
        n10 = ndcg_at_k(rels, preds, 10)
        ndcg5_list.append(n5)
        ndcg10_list.append(n10)

    print("=" * 50)
    print(f"  NDCG@5:  {np.mean(ndcg5_list):.4f}")
    print(f"  NDCG@10: {np.mean(ndcg10_list):.4f}")
    print("=" * 50)

    # 6. 按 query 的 NDCG 分布
    print(f"\nNDCG@10 分布:")
    ndcg10_arr = np.array(ndcg10_list)
    for threshold in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
        count = np.sum(ndcg10_arr <= threshold)
        print(f"  ≤ {threshold:.1f}: {count}/{len(ndcg10_arr)} ({count/len(ndcg10_arr)*100:.1f}%)")

    # 7. 最差的 query
    sorted_indices = np.argsort(ndcg10_list)
    print(f"\nNDCG@10 最低的 5 个 query:")
    worst_queries = []
    for i in sorted_indices[:5]:
        qid = list(query_groups.keys())[i]
        print(f"  {qid}: NDCG@5={ndcg5_list[i]:.4f}, NDCG@10={ndcg10_list[i]:.4f}, "
              f"docs={len(query_groups[qid]['relevances'])}")
        worst_queries.append({
            "qid": qid, "ndcg5": ndcg5_list[i], "ndcg10": ndcg10_list[i],
            "num_docs": len(query_groups[qid]['relevances']),
        })

    # 8. 写出结果文件
    output_path = args.output
    if output_path is None:
        output_path = args.infer.rsplit(".", 1)[0] + "_eval.json"

    results = {
        "infer_file": args.infer,
        "data_file": args.data,
        "num_queries": len(query_groups),
        "num_docs": len(data_rows),
        "ndcg@5": round(float(np.mean(ndcg5_list)), 4),
        "ndcg@10": round(float(np.mean(ndcg10_list)), 4),
        "ndcg@5_std": round(float(np.std(ndcg5_list)), 4),
        "ndcg@10_std": round(float(np.std(ndcg10_list)), 4),
        "per_query": [
            {"qid": qid, "ndcg@5": round(ndcg5_list[i], 4), "ndcg@10": round(ndcg10_list[i], 4)}
            for i, qid in enumerate(query_groups.keys())
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
