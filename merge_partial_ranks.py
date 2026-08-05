"""
merge_partial_ranks.py
======================
合并已有的 rank tmp 文件为完整 TSV（跳过缺失的 rank）。

用法:
  python merge_partial_ranks.py \
      --tsv_prefix /path/to/infer_results.tsv \
      --world_size 8 \
      --output /path/to/infer_results.tsv
"""

import os
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tsv_prefix", required=True, help="TSV 路径前缀（自动查找 .rank*.tmp）")
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()

    dfs = []
    for r in range(args.world_size):
        tmp = f"{args.tsv_prefix}.rank{r}.tmp"
        if os.path.exists(tmp) and os.path.getsize(tmp) > 100:
            df = pd.read_csv(tmp, sep="\t")
            dfs.append(df)
            print(f"  rank{r}: {len(df)} 行 ✓")
        else:
            print(f"  rank{r}: 缺失，跳过")

    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sort_values("global_idx").reset_index(drop=True)
    merged.to_csv(args.output, index=False, sep="\t")
    print(f"\n[INFO] 合并完成: {args.output} ({len(merged)} 行)")


if __name__ == "__main__":
    main()
