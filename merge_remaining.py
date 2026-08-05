"""
merge_remaining.py
==================
按 global_idx 对齐合并 TSV 和 jsonl 为 parquet（支持分片 TSV）。

与 merge_distill_data.py 不同：
  - 支持 TSV 是分片文件（global_idx 不从 0 开始）
  - 支持合并多个 .rank*.tmp 分片为一个完整 TSV 后再 merge
  - 按 global_idx 对齐 jsonl（而不是按行号顺序）

用法:
  # merge 完整的 TSV
  python merge_remaining.py \
      --jsonl /path/to/train.jsonl \
      --tsv /path/to/infer_results.tsv \
      --output /path/to/distill.parquet

  # merge 分片
  python merge_remaining.py \
      --jsonl /path/to/train.jsonl \
      --tsv /path/to/rank2.tmp /path/to/rank3.tmp /path/to/rank5.tmp \
      --output /path/to/distill_partial.parquet
"""

import argparse
import json
import ast
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def load_tsv_by_idx(tsv_paths):
    """加载一个或多个 TSV 文件，按 global_idx 索引存入 dict。"""
    idx_to_probs = {}
    for tsv_path in tsv_paths:
        print(f"[INFO] 读取 TSV: {tsv_path}")
        with open(tsv_path, 'r') as f:
            header = f.readline().strip().split('\t')
            idx_col = header.index('global_idx')
            scores_col = header.index('all_digit_scores')

            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) <= max(idx_col, scores_col):
                    continue
                try:
                    global_idx = int(parts[idx_col])
                    probs = ast.literal_eval(parts[scores_col])
                except (ValueError, SyntaxError):
                    continue
                if isinstance(probs, dict):
                    idx_to_probs[global_idx] = probs

        print(f"  → 已加载 {len(idx_to_probs)} 条记录")
    return idx_to_probs


def main():
    parser = argparse.ArgumentParser(description="按 global_idx 对齐 merge")
    parser.add_argument("--jsonl", required=True, help="原始训练数据 jsonl")
    parser.add_argument("--tsv", nargs='+', required=True, help="TSV 文件路径（可多个分片）")
    parser.add_argument("--output", "-o", required=True, help="输出 parquet")
    parser.add_argument("--filter_invalid", action="store_true",
                        help="过滤 teacher_probs 概率和 < 0.99 的行")
    parser.add_argument("--chunk_size", type=int, default=100000)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载所有 TSV 数据（按 global_idx 索引）
    idx_to_probs = load_tsv_by_idx(args.tsv)
    print(f"[INFO] TSV 总记录数: {len(idx_to_probs)}")
    if not idx_to_probs:
        print("[ERROR] 没有有效的 TSV 记录")
        return

    min_idx = min(idx_to_probs.keys())
    max_idx = max(idx_to_probs.keys())
    print(f"[INFO] global_idx 范围: [{min_idx}, {max_idx}]")

    # 流式读取 jsonl，按 global_idx 匹配
    schema = pa.schema([
        ('query', pa.string()),
        ('title', pa.string()),
        ('passage', pa.string()),
        ('label', pa.int32()),
        ('teacher_probs', pa.string()),
    ])
    writer = pq.ParquetWriter(str(out_path), schema, compression='snappy')

    chunk_queries = []
    chunk_titles = []
    chunk_passages = []
    chunk_labels = []
    chunk_probs = []

    written = 0
    skipped = 0

    print(f"[INFO] 读取 jsonl: {args.jsonl}")
    with open(args.jsonl, 'r') as f:
        for line_idx, line in enumerate(f):
            if line_idx < min_idx:
                continue
            if line_idx > max_idx:
                break

            if line_idx not in idx_to_probs:
                continue

            line = line.strip()
            if not line:
                skipped += 1
                continue

            obj = json.loads(line)
            probs = idx_to_probs[line_idx]

            if args.filter_invalid:
                if sum(probs.values()) < 0.99:
                    skipped += 1
                    continue

            chunk_queries.append(obj.get("query", ""))
            chunk_titles.append(obj.get("title", ""))
            chunk_passages.append(obj.get("passage", ""))
            chunk_labels.append(int(obj.get("label", 0)))
            chunk_probs.append(json.dumps(probs))
            written += 1

            if len(chunk_queries) >= args.chunk_size:
                table = pa.table({
                    'query': pa.array(chunk_queries, type=pa.string()),
                    'title': pa.array(chunk_titles, type=pa.string()),
                    'passage': pa.array(chunk_passages, type=pa.string()),
                    'label': pa.array(chunk_labels, type=pa.int32()),
                    'teacher_probs': pa.array(chunk_probs, type=pa.string()),
                })
                writer.write_table(table)
                chunk_queries.clear()
                chunk_titles.clear()
                chunk_passages.clear()
                chunk_labels.clear()
                chunk_probs.clear()
                print(f"  [PROGRESS] 已写入 {written} 行")

    if chunk_queries:
        table = pa.table({
            'query': pa.array(chunk_queries, type=pa.string()),
            'title': pa.array(chunk_titles, type=pa.string()),
            'passage': pa.array(chunk_passages, type=pa.string()),
            'label': pa.array(chunk_labels, type=pa.int32()),
            'teacher_probs': pa.array(chunk_probs, type=pa.string()),
        })
        writer.write_table(table)

    writer.close()
    print(f"\n[INFO] 完成: 写入={written}, 跳过={skipped}")
    print(f"[INFO] 输出: {out_path}")


if __name__ == "__main__":
    main()
