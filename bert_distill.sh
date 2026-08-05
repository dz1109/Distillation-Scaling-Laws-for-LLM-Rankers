#!/bin/bash

# BERT Cross-Encoder 蒸馏脚本 (JS divergence from LLM teacher)
# 用法:
#   设置环境变量后执行:
#     MODEL_PATH=... DATA_PATH=... OUTPUT_DIR=... bash bert_distill.sh

MODEL_PATH=${MODEL_PATH:-"./bert_ce_output/checkpoint-188"}
DATA_PATH=${DATA_PATH:-"./distill_data.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output/bert_distill"}

deepspeed --num_gpus=8 bert_distill.py \
    --model_name_or_path ${MODEL_PATH} \
    --train_data_path ${DATA_PATH} \
    --num_labels 2 \
    --max_length 1024 \
    --output_dir ${OUTPUT_DIR} \
    --deepspeed ds_config_zero1.json \
    --bf16 True \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --warmup_ratio 0.1 \
    --logging_steps 100 \
    --save_steps 1000 \
    --report_to none
