#!/bin/bash

# 顺序执行 "teacher 数据 × 蒸馏 loss" 的 BERT 蒸馏批处理。
#
# 用法:
#   bash distill_batch_bert.sh [NUM_LABELS]
#     NUM_LABELS: 2 或 4, 默认 2
#
# 可通过环境变量覆盖 (空格分隔):
#   DATA_TAGS="1.5b_ms 3b_ms 7b_ms"       # 使用哪些 teacher 数据
#   LOSS_TYPES="jsd fkl rkl hard"         # 使用哪些蒸馏 loss
#   MODEL_PATH=/path/to/bert_init          # 学生模型初始化 ckpt
#   MODEL_TAG=bert                         # 输出目录前缀标识
#   DATA_ROOT=/path/to/data                # 数据根目录
#
# 示例:
#   DATA_TAGS="3b_ms 7b_ms" LOSS_TYPES="fkl rkl" bash distill_batch_bert.sh 2

set -e
set -o pipefail

NUM_LABELS=${1:-2}
MODEL_PATH=${MODEL_PATH:-"./output/bert_ce/checkpoint-188"}
DATA_ROOT=${DATA_ROOT:-"."}
MODEL_TAG=${MODEL_TAG:-"bert"}
OUTPUT_ROOT=${OUTPUT_ROOT:-"./output"}

# 默认要跑的 teacher 数据标签（对应 train_distill_data_${tag}.parquet）
DATA_TAGS=${DATA_TAGS:-"1.5b_ms 3b_ms 7b_ms 14b_ms 32b_ms"}

# 默认要跑的蒸馏 loss 类型: jsd / fkl / rkl / hard
LOSS_TYPES=${LOSS_TYPES:-"fkl rkl"}

LOG_DIR="bert_distill_logs"
mkdir -p ${LOG_DIR}
mkdir -p ${OUTPUT_ROOT}

echo "======================================================================"
echo "[BATCH] NUM_LABELS=${NUM_LABELS}"
echo "[BATCH] MODEL_PATH=${MODEL_PATH}"
echo "[BATCH] MODEL_TAG=${MODEL_TAG}"
echo "[BATCH] DATA_TAGS=${DATA_TAGS}"
echo "[BATCH] LOSS_TYPES=${LOSS_TYPES}"
echo "======================================================================"

for TAG in ${DATA_TAGS}; do
    DATA_PATH="${DATA_ROOT}/train_distill_data_${TAG}.parquet"

    if [ ! -f "${DATA_PATH}" ]; then
        echo "[SKIP] 数据不存在: ${DATA_PATH}"
        continue
    fi

    for LOSS in ${LOSS_TYPES}; do
        OUTPUT_DIR="${OUTPUT_ROOT}/${MODEL_TAG}_distill_${NUM_LABELS}label_teacher_${TAG}_${LOSS}"
        LOG_FILE="${LOG_DIR}/$(basename ${OUTPUT_DIR}).log"

        echo ""
        echo "----------------------------------------------------------------------"
        echo "[RUN] data=${TAG}  loss=${LOSS}  →  ${OUTPUT_DIR}"
        echo "[RUN] log: ${LOG_FILE}"
        echo "----------------------------------------------------------------------"

        if [ -d "${OUTPUT_DIR}" ] && [ -n "$(ls -A ${OUTPUT_DIR} 2>/dev/null)" ]; then
            echo "[SKIP] 输出目录已存在且非空: ${OUTPUT_DIR}"
            continue
        fi

        deepspeed --num_gpus=8 bert_distill.py \
            --model_name_or_path ${MODEL_PATH} \
            --train_data_path ${DATA_PATH} \
            --num_labels ${NUM_LABELS} \
            --max_length 512 \
            --distill_type "${LOSS}" \
            --distill_weight 1.0 \
            --output_dir ${OUTPUT_DIR} \
            --deepspeed ds_config_zero1.json \
            --bf16 True \
            --per_device_train_batch_size 64 \
            --gradient_accumulation_steps 2 \
            --learning_rate 1e-5 \
            --num_train_epochs 1 \
            --warmup_ratio 0.1 \
            --logging_steps 100 \
            --save_steps 1000 \
            --report_to none \
            2>&1 | tee "${LOG_FILE}"

        echo "[DONE] ${OUTPUT_DIR}"
    done
done

echo ""
echo "======================================================================"
echo "[BATCH DONE] 全部组合执行完毕"
echo "======================================================================"
