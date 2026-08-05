#!/bin/bash

# BERT Cross-Encoder 训练脚本 (hard label CE loss)
# 用法:
#   设置环境变量后执行:
#     MODEL_PATH=... DATA_PATH=... OUTPUT_DIR=... bash bert_train.sh

MODEL_PATH=${MODEL_PATH:-"./bert-base-multilingual-cased"}
DATA_PATH=${DATA_PATH:-"./data/train.jsonl"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output/bert_ce"}

deepspeed --num_gpus=8 bert_train.py \
    --model_name_or_path ${MODEL_PATH} \
    --train_data_path ${DATA_PATH} \
    --num_labels 4 \
    --max_length 1024 \
    --output_dir ${OUTPUT_DIR} \
    --deepspeed ds_config_zero1.json \
    --bf16 True \
    --per_device_train_batch_size 128 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type linear \
    --learning_rate 1e-5 \
    --num_train_epochs 1 \
    --warmup_ratio 0.01 \
    --weight_decay 0.0001 \
    --logging_steps 100 \
    --save_steps 10000 \
    --save_total_limit 3 \
    --report_to none
