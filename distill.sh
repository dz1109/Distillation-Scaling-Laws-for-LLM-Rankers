#!/bin/bash

# 知识蒸馏训练脚本 (LLM distill)
# 用法: bash distill.sh [NUM_LABELS]
#   NUM_LABELS: 2 (0/1 二分类) 或 4 (0/1/2/3 四分类)，默认 4
#
# 环境变量 (可选):
#   MODEL_PATH    - SFT 初始化模型路径 (默认: ./sft_output/checkpoint-135)
#   DATA_PATH     - 蒸馏数据 parquet 路径 (默认: ./distill_data.parquet)
#   OUTPUT_DIR    - 输出目录
#   NUM_GPUS      - GPU 数量 (默认: 8)
#   DISTILL_TYPE  - 蒸馏类型: soft/jsd/fkl/rkl/hard (默认: soft)

NUM_LABELS=${1:-4}
MODEL_PATH=${MODEL_PATH:-"./sft_output/checkpoint-135"}
DATA_PATH=${DATA_PATH:-"./distill_data.parquet"}
OUTPUT_DIR=${OUTPUT_DIR:-"distill_output"}
NUM_GPUS=${NUM_GPUS:-8}
DISTILL_TYPE=${DISTILL_TYPE:-"soft"}

accelerate launch --config_file=accelerate_configs/deepspeed_zero1.yaml \
    --num_processes ${NUM_GPUS} \
    --main_process_port 29500 \
    distill.py \
    --model_name_or_path ${MODEL_PATH} \
    --local_data_path ${DATA_PATH} \
    --num_labels ${NUM_LABELS} \
    --teacher_prob_key "teacher_probs" \
    --jsd_weight 1.0 \
    --distill_type "${DISTILL_TYPE}" \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --bf16 \
    --warmup_ratio 0.01 \
    --weight_decay 0.0001 \
    --use_liger_kernel True \
    --attn_implementation "flash_attention_2" \
    --dataset_name "" \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --torch_dtype bfloat16 \
    --max_seq_length 1024 \
    --gradient_checkpointing \
    --logging_steps 10 \
    --eval_strategy "no" \
    --num_train_epochs 1 \
    --save_steps 1000 \
    --report_to "none"
