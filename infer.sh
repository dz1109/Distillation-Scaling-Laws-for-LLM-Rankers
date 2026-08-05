#!/bin/bash

# 多卡并行推理脚本
# 用法: bash infer.sh [MODEL_PATH] [DATA_PATH] [SAVE_PATH] [NUM_GPUS] [BATCH_SIZE]
#
# 示例:
#   bash infer.sh ./sft_output/checkpoint-135 ./data/test.jsonl ./results.tsv 8 128

MODEL_PATH=${1:-"./sft_output/checkpoint-135"}
DATA_PATH=${2:-"./data/test.jsonl"}
SAVE_PATH=${3:-"./infer_results.tsv"}
NUM_GPUS=${4:-8}
BATCH_SIZE=${5:-128}
NUM_LABELS=${NUM_LABELS:-4}

torchrun --nproc_per_node=${NUM_GPUS} \
    --master_port 29300 \
    infer_model.py \
    --model_name_or_path ${MODEL_PATH} \
    --local_data_path ${DATA_PATH} \
    --infer_save_path ${SAVE_PATH} \
    --batch_size ${BATCH_SIZE} \
    --max_length 1024 \
    --num_labels ${NUM_LABELS}
