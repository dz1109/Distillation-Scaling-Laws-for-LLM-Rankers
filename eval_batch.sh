#!/bin/bash

# 使用 eval.py 批量评测: 多个数据集 × 多个模型目录。
#
# 用法:
#   bash eval_batch.sh [NUM_LABELS]
#     NUM_LABELS: 模型输出标签数 (2 或 4), 默认 2
#
# 可通过环境变量覆盖:
#   CKPT_DIRS="dir1 dir2 ..."         # 空格分隔的模型目录列表, 每个目录内含 checkpoint-*
#   DATASETS="tag:path tag:path ..."   # 空格分隔的 "标签:数据路径" 对
#   OUTPUT_ROOT=./eval_results         # 结果输出根目录
#   BATCH_SIZE=64
#   EVAL_LABEL_MAX=3                   # ECE 归一化上限

set -e
set -o pipefail

NUM_LABELS=${1:-2}
DATA_ROOT=${DATA_ROOT:-"."}
OUTPUT_ROOT=${OUTPUT_ROOT:-"eval_results"}
BATCH_SIZE=${BATCH_SIZE:-64}
MAX_LENGTH=${MAX_LENGTH:-512}
EVAL_LABEL_MAX=${EVAL_LABEL_MAX:-3}

# 默认评测的数据集 (格式: 短标签:文件路径)
DATASETS=${DATASETS:-"\
dl_test:${DATA_ROOT}/data/test.jsonl"}

# 默认要评的模型目录 (设置 CKPT_DIRS 环境变量来覆盖)
CKPT_DIRS=${CKPT_DIRS:-""}

if [ -z "${CKPT_DIRS}" ]; then
    echo "[ERROR] 请设置 CKPT_DIRS 环境变量指定要评测的模型目录"
    echo "  示例: CKPT_DIRS=\"distill_output_1 distill_output_2\" bash eval_batch.sh 2"
    exit 1
fi

LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

echo "======================================================================"
echo "[BATCH] NUM_LABELS=${NUM_LABELS}, EVAL_LABEL_MAX=${EVAL_LABEL_MAX}"
echo "[BATCH] BATCH_SIZE=${BATCH_SIZE}, MAX_LENGTH=${MAX_LENGTH}"
echo "[BATCH] OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "[BATCH] DATASETS:"
for DS in ${DATASETS}; do echo "   - ${DS}"; done
echo "[BATCH] CKPT_DIRS:"
for D in ${CKPT_DIRS}; do echo "   - ${D}"; done
echo "======================================================================"

for CKPT_DIR in ${CKPT_DIRS}; do
    if [ ! -d "${CKPT_DIR}" ]; then
        echo "[SKIP] 模型目录不存在: ${CKPT_DIR}"
        continue
    fi
    MODEL_TAG=$(basename "${CKPT_DIR}")

    for DS in ${DATASETS}; do
        DS_TAG="${DS%%:*}"
        DS_PATH="${DS#*:}"

        if [ ! -f "${DS_PATH}" ]; then
            echo "[SKIP] 数据不存在: ${DS_PATH}"
            continue
        fi

        OUTPUT_JSON="${OUTPUT_ROOT}/eval_${DS_TAG}_${MODEL_TAG}.json"
        LOG_FILE="${LOG_DIR}/eval_${DS_TAG}_${MODEL_TAG}.log"

        if [ -f "${OUTPUT_JSON}" ]; then
            echo "[SKIP] 结果已存在: ${OUTPUT_JSON}"
            continue
        fi

        echo ""
        echo "----------------------------------------------------------------------"
        echo "[RUN] model=${MODEL_TAG}  dataset=${DS_TAG}"
        echo "[RUN] output: ${OUTPUT_JSON}"
        echo "----------------------------------------------------------------------"

        python eval.py \
            --ckpt_dir "${CKPT_DIR}" \
            --data_path "${DS_PATH}" \
            --output "${OUTPUT_JSON}" \
            --num_labels "${NUM_LABELS}" \
            --eval_label_max "${EVAL_LABEL_MAX}" \
            --batch_size "${BATCH_SIZE}" \
            --max_length "${MAX_LENGTH}" \
            2>&1 | tee "${LOG_FILE}"

        echo "[DONE] ${OUTPUT_JSON}"
    done
done

echo ""
echo "======================================================================"
echo "[BATCH DONE] 全部评测执行完毕, 结果目录: ${OUTPUT_ROOT}"
echo "======================================================================"
