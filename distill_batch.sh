#!/bin/bash

# 批量蒸馏训练脚本 — 顺序执行不同 teacher × 不同 loss 的组合。
#
# 用法:
#   bash distill_batch.sh [NUM_LABELS]
#     NUM_LABELS: 2 或 4, 默认 2
#
# 可通过环境变量覆盖 (空格分隔):
#   DATA_TAGS="1.5b_ms 3b_ms 7b_ms"      # 使用哪些 teacher 数据
#   LOSS_TYPES="jsd fkl rkl hard"        # 使用哪些蒸馏 loss
#   MODEL_PATH=/path/to/student_init     # 学生模型初始化 ckpt
#   DATA_ROOT=/path/to/data              # 数据根目录 (默认: .)
#
# 示例:
#   DATA_TAGS="1.5b_ms 7b_ms" LOSS_TYPES="fkl rkl" bash distill_batch.sh 2

set -e
set -o pipefail

NUM_LABELS=${1:-2}
MODEL_PATH=${MODEL_PATH:-"./sft_output/checkpoint-754"}
DATA_ROOT=${DATA_ROOT:-"."}

# 默认要跑的 teacher 数据标签（对应 train_distill_data_${tag}.parquet）
DATA_TAGS=${DATA_TAGS:-"0.5b_ms"}

# 默认要跑的蒸馏 loss 类型: jsd / fkl / rkl / hard
LOSS_TYPES=${LOSS_TYPES:-"fkl rkl"}

LOG_DIR="distill_logs"
mkdir -p ${LOG_DIR}

echo "======================================================================"
echo "[BATCH] NUM_LABELS=${NUM_LABELS}"
echo "[BATCH] MODEL_PATH=${MODEL_PATH}"
echo "[BATCH] DATA_TAGS=${DATA_TAGS}"
echo "[BATCH] LOSS_TYPES=${LOSS_TYPES}"
echo "======================================================================"

for TAG in ${DATA_TAGS}; do
    DATA_PATH="${DATA_ROOT}/train_distill_data_${TAG}.parquet"

    if [ ! -f "${DATA_PATH}" ]; then
        echo "[SKIP] 数据不存在: ${DATA_PATH}"
        continue
    fi

    # 从 tag 里提取 teacher 规模, 用于输出目录命名 (如 1.5b_ms → 1.5B)
    TEACHER_SIZE=$(echo "${TAG}" | awk -F'_' '{print toupper($1)}')

    for LOSS in ${LOSS_TYPES}; do
        OUTPUT_DIR="distill_0.5B_${NUM_LABELS}label_teacher_${TEACHER_SIZE}_${LOSS}"
        LOG_FILE="${LOG_DIR}/${OUTPUT_DIR}.log"

        echo ""
        echo "----------------------------------------------------------------------"
        echo "[RUN] data=${TAG}  loss=${LOSS}  →  ${OUTPUT_DIR}"
        echo "[RUN] log: ${LOG_FILE}"
        echo "----------------------------------------------------------------------"

        if [ -d "${OUTPUT_DIR}" ] && [ -n "$(ls -A ${OUTPUT_DIR} 2>/dev/null)" ]; then
            echo "[SKIP] 输出目录已存在且非空: ${OUTPUT_DIR}"
            continue
        fi

        accelerate launch --config_file=accelerate_configs/deepspeed_zero1.yaml \
            --num_processes 8 \
            --main_process_port 29600 \
            distill.py \
            --model_name_or_path ${MODEL_PATH} \
            --local_data_path ${DATA_PATH} \
            --num_labels ${NUM_LABELS} \
            --teacher_prob_key "teacher_probs" \
            --jsd_weight 1.0 \
            --distill_type "${LOSS}" \
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
            --max_seq_length 512 \
            --gradient_checkpointing \
            --logging_steps 10 \
            --eval_strategy "no" \
            --num_train_epochs 1 \
            --save_steps 1000 \
            --report_to "none" \
            2>&1 | tee "${LOG_FILE}"

        echo "[DONE] ${OUTPUT_DIR}"
    done
done

echo ""
echo "======================================================================"
echo "[BATCH DONE] 全部组合执行完毕"
echo "======================================================================"
