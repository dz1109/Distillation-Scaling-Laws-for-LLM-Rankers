# LLM Distillation for Relevance Scoring

基于 LLM 的相关性评分知识蒸馏工具集。

## 功能概述

| 模块 | 文件 | 说明 |
|------|------|------|
| **LLM 蒸馏训练** | `distill.py` | 用 Teacher LLM 的软标签蒸馏 Student LLM（JS/Forward KL/Reverse KL/Hard） |
| **BERT 蒸馏训练** | `bert_distill.py` | 用 LLM Teacher 的软标签蒸馏 BERT Cross-Encoder |
| **BERT 硬标签训练** | `bert_train.py` | 直接使用硬标签训练 BERT Cross-Encoder |
| **SFT 训练** | `sft_gen.py` | 直接用 label 做 SFT（生成式训练） |
| **LLM 推理** | `infer_model.py` | 多卡并行推理 LLM |
| **BERT 推理** | `bert_infer.py` | 多卡并行推理 BERT |
| **断点续跑** | `infer_resume.py` | 断点续跑推理 |
| **LLM 评测** | `eval.py` | LLM checkpoint 评测（NDCG/ECE/Polarization） |
| **BERT 评测** | `bert_eval.py` | BERT checkpoint 评测 |
| **NDCG 计算** | `eval_ndcg.py` | 从 TSV 推理结果计算 NDCG |
| **Polarization 分析** | `polarization.py` | 分析 Teacher 的极化程度 |
| **数据合并** | `merge_distill_data.py` | 合并训练数据 + Teacher 推理结果 |
| **字段提取** | `extract_fields.py` | 从 parquet 提取字段 |
| **工具脚本** | `*.sh` | 各模块的运行脚本 |

## 数据格式

训练/测试数据为 JSONL 格式：
```json
{"qid": "...", "query": "...", "docid": "...", "title": "...", "passage": "...", "label": 0}
```

蒸馏数据需要额外包含 Teacher 推理后的 `teacher_probs` 字段（通过 `merge_distill_data.py` 合并）。

## 安装

```bash
pip install torch transformers datasets trl accelerate deepspeed pyarrow
```

## 快速开始

```bash
# 1. SFT 训练
sh sft_gen.sh

# 2. Teacher 推理
sh infer.sh

# 3. 合并蒸馏数据
sh distill_batch.sh

# 4. 蒸馏训练
sh distill.sh

# 5. 评测
sh eval_batch.sh
```

## 配置文件

DeepSpeed 配置示例见 `ds_config_zero1.json`，可根据需要调整 ZeRO stage。
