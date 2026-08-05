"""
merge_distill_data.py
=====================
合并原始训练数据和 teacher 推理结果，生成 distill.py 所需的训练数据（parquet 格式）。

按 global_idx 对齐（而非行号顺序），支持 TSV 中有缺失行。

输入:
  - jsonl: 原始训练数据 (query, title, passage, label, ...)
  - tsv: teacher 推理结果 (global_idx, all_digit_scores, output_texts, weighted_scores)

输出:
  - parquet: 合并后数据 (query, title, passage, label, teacher_probs)
    其中 teacher_probs 存为 JSON 字符串，仅包含 TSV 中存在的行

用法:
  python merge_distill_data.py \
      --jsonl /path/to/train.jsonl \
      --tsv /path/to/infer_results.tsv \
      --output /path/to/train_distill_data.parquet
"""

import argparse
import json
import ast
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


def load_tsv_index(tsv_path):
    """加载 TSV，按 global_idx 建立索引。跳过解析失败的行。"""
    idx_to_probs = {}
    print(f"[INFO] 加载 TSV: {tsv_path}")
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

    print(f"[INFO] TSV 有效记录: {len(idx_to_probs)}")
    if idx_to_probs:
        print(f"[INFO] global_idx 范围: [{min(idx_to_probs)}, {max(idx_to_probs)}]")
    return idx_to_probs


def main():
    parser = argparse.ArgumentParser(description="合并原始数据和 teacher 推理结果（按 global_idx join）")
    parser.add_argument("--jsonl", required=True, help="原始训练数据 jsonl 路径")
    parser.add_argument("--tsv", required=True, help="Teacher 推理结果 tsv 路径")
    parser.add_argument("--output", "-o", required=True, help="输出 parquet 路径")
    parser.add_argument("--filter_invalid", action="store_true",
                        help="过滤掉 teacher_probs 概率和 < 0.99 的异常行")
    parser.add_argument("--chunk_size", type=int, default=1000000,
                        help="每个 chunk 写入的行数 (默认 100000)")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载 TSV 索引
    idx_to_probs = load_tsv_index(args.tsv)
    if not idx_to_probs:
        print("[ERROR] TSV 中没有有效记录")
        return

    schema = pa.schema([
        ('query', pa.string()),
        ('title', pa.string()),
        ('passage', pa.string()),
        ('label', pa.int32()),
        ('teacher_probs', pa.string()),
    ])

    writer = pq.ParquetWriter(str(out_path), schema, compression='snappy')

    written = 0
    skipped = 0
    missing = 0

    chunk_queries = []
    chunk_titles = []
    chunk_passages = []
    chunk_labels = []
    chunk_probs = []

    print(f"[INFO] 读取 jsonl: {args.jsonl}")
    with open(args.jsonl, 'r') as f:
        for line_idx, line in enumerate(f):
            # 跳过 TSV 中不存在的行
            if line_idx not in idx_to_probs:
                missing += 1
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
    print(f"\n[INFO] 完成: 写入={written}, 跳过={skipped}, TSV缺失={missing}")
    print(f"[INFO] 输出: {out_path}")


if __name__ == "__main__":
    main()
