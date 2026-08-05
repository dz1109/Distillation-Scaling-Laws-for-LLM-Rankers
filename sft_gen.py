"""
sft_gen.py
==========
基于 TRL SFTTrainer 的生成式相关性训练（SFT）。

用 label 直接作为 assistant 回复，做 next-token prediction 训练。

数据格式 (jsonl):
  {"query": "...", "title": "...", "passage": "...", "label": 1}

用法:
  accelerate launch --config_file=accelerate_configs/deepspeed_zero2.yaml \\
      --num_processes 8 sft_gen.py \\
      --model_name_or_path /path/to/model \\
      --local_data_path /path/to/train.jsonl \\
      --num_labels 2 \\
      --output_dir ./output/sft
"""

import argparse
import os
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_quantization_config,
    setup_chat_format,
)


@dataclass
class LocalArguments:
    local_data_path: str = field(default=None, metadata={"help": "训练数据 jsonl 路径"})
    num_labels: int = field(default=2, metadata={"help": "标签数量: 2 表示 0/1, 4 表示 0/1/2/3"})


def fullwidth_to_halfwidth(s: str) -> str:
    """将字符串中的全角字符转换为半角字符"""
    result = []
    for char in s:
        code = ord(char)
        # 全角空格特殊处理
        if code == 0x3000:
            code = 0x20
        # 其他全角字符（除空格）转换
        elif 0xFF01 <= code <= 0xFF5E:
            code -= 0xFEE0
        result.append(chr(code))
    return ''.join(result)


def get_template_sft(query, title, passage, num_labels=2):
    doc = title + passage
    if num_labels == 2:
        instruction = "Given a web search query, retrieve relevant passages that answer the query. Your task is to score the relevance between the Query and the Doc: 0 means irrelevant, and 1 means relevant."
    else:
        instruction = "Given a web search query, retrieve relevant passages that answer the query. Your task is to score the relevance between the Query and the Doc: 0 means irrelevant, 1 means slightly relevant, 2 means partially relevant, and 3 means highly relevant."
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc
    )


def main(script_args, training_args, model_args, local_args):
    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # Create model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, use_fast=True
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    # Set default chat template if needed
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    ################
    # Dataset
    ################
    train_dataset = load_dataset("json", data_files=local_args.local_data_path)['train']

    def process_dataset(examples):
        messages = []
        for i in range(len(examples["query"])):
            query = fullwidth_to_halfwidth(examples["query"][i]).lower()
            title = fullwidth_to_halfwidth(examples["title"][i]).lower()
            passage = fullwidth_to_halfwidth(examples["passage"][i]).lower()
            final_label = examples["label"][i]
            messages.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": get_template_sft(query, title, passage, local_args.num_labels)},
                {"role": "assistant", "content": f"{int(final_label)}"}
            ])
        return {"messages": messages}

    train_dataset = train_dataset.map(
        process_dataset,
        batched=True,
        remove_columns=train_dataset.column_names,
    ).shuffle(seed=training_args.seed)

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


def make_parser(subparsers: argparse._SubParsersAction = None):
    dataclass_types = (ScriptArguments, SFTConfig, ModelConfig, LocalArguments)
    if subparsers is not None:
        parser = subparsers.add_parser("sft", help="Run the SFT training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, local_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(script_args, training_args, model_args, local_args)
