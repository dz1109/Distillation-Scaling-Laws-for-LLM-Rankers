"""
polarization.py
===============
分析 Teacher LLM 相关性输出的极化程度。

随着 Teacher 规模增大，相关性分布是否变得更尖锐（更低熵、更大margin）。
这对理解蒸馏行为至关重要：如果大模型 teacher 过于极化，小模型 student
可能无法模仿。

用法:
  python polarization.py \
      --ckpt_paths checkpoint_1.5B checkpoint_3B checkpoint_7B \
      --data_path test.jsonl --output polarization.json --num_labels 2
"""

import os
import json
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset


def fullwidth_to_halfwidth(s: str) -> str:
    out = []
    for ch in s:
        c = ord(ch)
        if c == 0x3000:
            c = 0x20
        elif 0xFF01 <= c <= 0xFF5E:
            c -= 0xFEE0
        out.append(chr(c))
    return "".join(out)


def get_template(query, title, passage, num_labels=2):
    doc = title + passage
    if num_labels == 2:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query. "
            "Your task is to score the relevance between the Query and the Doc: "
            "0 means irrelevant, and 1 means relevant."
        )
    else:
        instruction = (
            "Given a web search query, retrieve relevant passages that answer the query. "
            "Your task is to score the relevance between the Query and the Doc: "
            "0 means irrelevant, 1 means slightly relevant, "
            "2 means partially relevant, and 3 means highly relevant."
        )
    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction, query=query, doc=doc)


def _entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return float(-np.sum(p * np.log(p), axis=-1).mean())


def eval_single_teacher(ckpt_path, data_path, gpu_id, num_labels, batch_size, max_length):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    label_strs = [str(d) for d in range(num_labels)]
    digit_ids = [tokenizer.encode(s, add_special_tokens=False)[0] for s in label_strs]

    dataset = load_dataset("json", data_files=data_path)["train"]

    def process(examples):
        msgs = []
        for i in range(len(examples["query"])):
            q = fullwidth_to_halfwidth(str(examples.get("query", [""])[i] or "")).lower()
            t = fullwidth_to_halfwidth(str(examples.get("title", [""])[i] or "")).lower()
            p = fullwidth_to_halfwidth(str(examples.get("passage", [""])[i] or "")).lower()
            msgs.append([
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": get_template(q, t, p, num_labels)},
            ])
        return {"messages": msgs}

    ds = dataset.map(process, batched=True, remove_columns=dataset.column_names)
    batched = ds.batch(batch_size)

    rel_probs = []   # (N, num_labels) softmax over the relevance tokens
    margins = []     # (N,) logit_yes - logit_no (binary only)
    with torch.no_grad():
        for batch in tqdm(batched, desc=f"[GPU{gpu_id}] {os.path.basename(ckpt_path)}"):
            texts = tokenizer.apply_chat_template(
                batch["messages"], tokenize=False, add_generation_prompt=True)
            inp = tokenizer(texts, return_tensors="pt", padding=True,
                           return_token_type_ids=False, max_length=max_length,
                           truncation=True).to(device)
            out = model.generate(**inp, max_new_tokens=1, do_sample=False,
                                output_scores=True, return_dict_in_generate=True,
                                pad_token_id=tokenizer.eos_token_id)
            for b in range(len(batch["messages"])):
                logits = out.scores[0][b]
                rel_logits = torch.tensor([logits[i].item() for i in digit_ids])
                p = torch.softmax(rel_logits, dim=-1).numpy()
                rel_probs.append(p)
                if num_labels == 2:
                    margins.append(float(logits[digit_ids[1]].item() - logits[digit_ids[0]].item()))

    rel_probs = np.array(rel_probs)                       # (N, C)
    ent = _entropy(rel_probs)                             # mean entropy over the relevance dist
    maxprob = float(rel_probs.max(axis=-1).mean())
    # temperature sensitivity: soften logits by T=2 (in logit space) and see how much entropy rises
    logits_recovered = np.log(np.clip(rel_probs, 1e-12, 1.0))
    soft = logits_recovered / 2.0
    soft = soft - soft.max(axis=-1, keepdims=True)
    soft_p = np.exp(soft); soft_p /= soft_p.sum(axis=-1, keepdims=True)
    temp_sens = _entropy(soft_p) - ent                    # >0, bigger = easier to soften = more polarized

    res = {
        "checkpoint_name": os.path.basename(os.path.dirname(ckpt_path)) or os.path.basename(ckpt_path),
        "num_samples": int(rel_probs.shape[0]),
        "mean_entropy": round(ent, 4),
        "mean_maxprob": round(maxprob, 4),
        "temp_sensitivity": round(temp_sens, 4),
    }
    if num_labels == 2 and margins:
        m = np.array(margins)
        res["mean_margin"] = round(float(m.mean()), 4)
        res["var_margin"] = round(float(m.var()), 4)
    return res


def main():
    ap = argparse.ArgumentParser(description="Measure teacher logit polarization vs size.")
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--ckpt_paths", nargs="+", default=None)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--num_labels", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--num_gpus", type=int, default=None)
    args = ap.parse_args()

    if args.ckpt_paths:
        ckpts = args.ckpt_paths
    elif args.ckpt_dir:
        ckpts = sorted(glob.glob(os.path.join(args.ckpt_dir, "checkpoint-*"))) or [args.ckpt_dir]
    else:
        ap.error("need --ckpt_dir or --ckpt_paths")

    ngpu = args.num_gpus or min(torch.cuda.device_count(), len(ckpts))
    results = []
    with ProcessPoolExecutor(max_workers=ngpu) as ex:
        futs = {}
        for i, c in enumerate(ckpts):
            futs[ex.submit(eval_single_teacher, c, args.data_path, i % ngpu,
                           args.num_labels, args.batch_size, args.max_length)] = c
        for fu in as_completed(futs):
            try:
                r = fu.result(); results.append(r)
                extra = f", margin={r.get('mean_margin','-')}, var={r.get('var_margin','-')}"
                print(f"[DONE] {r['checkpoint_name']}: H={r['mean_entropy']}, "
                      f"maxp={r['mean_maxprob']}, tempSens={r['temp_sensitivity']}{extra}")
            except Exception as e:
                print(f"[ERROR] {os.path.basename(futs[fu])}: {e}")

    results.sort(key=lambda x: x.get("checkpoint_name", ""))
    json.dump({"data_path": args.data_path, "results": results},
              open(args.output, "w"), ensure_ascii=False, indent=2)
    print(f"\n[SAVED] {args.output}")
    print("\nTREND check: polarization INCREASES with teacher size iff "
          "mean_entropy ↓ and mean_margin / var_margin ↑ across 1.5B->32B.")


if __name__ == "__main__":
    main()
