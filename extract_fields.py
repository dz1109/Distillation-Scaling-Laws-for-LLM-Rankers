"""
extract_fields.py
=================
从 parquet 目录中提取 query, title, passage 三个字段，保存为 jsonl。

用法:
  python extract_fields.py --src_dir /path/to/parquet_dir --output /path/to/output.jsonl
"""

import os
import glob
import json
import argparse
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_file(f_path, fields):
    """处理单个 parquet 文件，返回 jsonl 字符串列表。"""
    table = pq.read_table(f_path, columns=fields)
    lines = []
    cols = {col: table.column(col) for col in fields}
    for i in range(len(table)):
        record = {col: str(cols[col][i].as_py() or "") for col in fields}
        lines.append(json.dumps(record, ensure_ascii=False))
    return lines, os.path.basename(f_path)


def main():
    parser = argparse.ArgumentParser(description="从 parquet 提取字段为 jsonl")
    parser.add_argument("--src_dir", required=True, help="parquet 源目录")
    parser.add_argument("--output", "-o", required=True, help="输出 jsonl 路径")
    parser.add_argument("--fields", nargs="+", default=["query", "title", "passage"],
                        help="要提取的字段名 (默认: query title passage)")
    parser.add_argument("--num_workers", type=int, default=16, help="并行 worker 数")
    args = parser.parse_args()

    parquet_files = sorted(glob.glob(os.path.join(args.src_dir, "*.parquet")))
    if not parquet_files:
        parquet_files = sorted(glob.glob(os.path.join(args.src_dir, "**/*.parquet"), recursive=True))
    if not parquet_files:
        print(f"[ERROR] 未找到 parquet 文件: {args.src_dir}")
        return

    print(f"[INFO] 找到 {len(parquet_files)} 个 parquet 文件")
    print(f"[INFO] 提取字段: {args.fields}")
    print(f"[INFO] 并行 workers: {args.num_workers}")

    total = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_file, f, args.fields): f for f in parquet_files}
            for future in as_completed(futures):
                lines, fname = future.result()
                fout.write("\n".join(lines))
                if lines:
                    fout.write("\n")
                total += len(lines)
                print(f"  {fname}: {len(lines)} 行")

    print(f"[INFO] 完成: 总计 {total} 行 → {args.output}")


if __name__ == "__main__":
    main()
