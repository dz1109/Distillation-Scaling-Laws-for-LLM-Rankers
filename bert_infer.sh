#!/bin/bash

# BERT Cross-Encoder 推理脚本
# 用法: bash bert_infer.sh [MODEL_PATH] [DATA_PATH] [OUTPUT_PATH]

MODEL_PATH=${1:-"./output/bert_ce/checkpoint-188"}
DATA_PATH=${2:-"./data/test.jsonl"}
OUTPUT_PATH=${3:-"./bert_infer_results.tsv"}
NUM_LABELS=${NUM_LABELS:-2}

torchrun --nproc_per_node=8 bert_infer.py \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --output_path ${OUTPUT_PATH} \
    --batch_size 128 \
    --max_length 512 \
    --num_labels ${NUM_LABELS}

echo ""
echo "推理完成: ${OUTPUT_PATH}"
echo "评测命令: python eval_ndcg.py --infer ${OUTPUT_PATH} --data ${DATA_PATH}"
