#!/bin/bash

# SFT 训练脚本 (生成式相关性训练)
# 用法:
#   设置环境变量后执行:
#     MODEL_PATH="..." DATA_PATH="..." OUTPUT_DIR="..." bash sft_gen.sh
#
# 环境变量 (可选，有默认值):
#   MODEL_PATH    - 基座模型路径 (默认: Qwen2.5-0.5B-Instruct)
#   DATA_PATH     - 训练数据 jsonl 路径 (默认: ./data/train.jsonl)
#   NUM_LABELS    - 标签数: 2 或 4 (默认: 2)
#   OUTPUT_DIR    - 输出目录 (默认: ./sft_output)
#   NUM_GPUS      - GPU 数量 (默认: 8)

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-0.5B-Instruct"}
DATA_PATH=${DATA_PATH:-"./data/train.jsonl"}
NUM_LABELS=${NUM_LABELS:-2}
OUTPUT_DIR=${OUTPUT_DIR:-"./sft_output"}
NUM_GPUS=${NUM_GPUS:-8}

accelerate launch --config_file=accelerate_configs/deepspeed_zero1.yaml \
    --num_processes ${NUM_GPUS} \
    --main_process_port 29300 \
    sft_gen.py \
    --model_name_or_path ${MODEL_PATH} \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --neftune_noise_alpha 5 \
    --local_data_path ${DATA_PATH} \
    --num_labels ${NUM_LABELS} \
    --output_dir ${OUTPUT_DIR} \
    --bf16 \
    --warmup_ratio 0.01 \
    --weight_decay 0.0001 \
    --use_liger_kernel True \
    --attn_implementation "flash_attention_2" \
    --dataset-name "" \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --torch_dtype bfloat16 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --eval_strategy "no" \
    --num_train_epochs 1 \
    --save_steps 10000 \
    --report_to "none"
