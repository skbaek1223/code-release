"""
Step 5: Generate val predictions with the SFT model

Val data: hard_selected items that didn't get goals generated (precomputed in 01_val_loader)

Uses the same prompt and output format as 04_sft_data_and_train.py.
- system prompt: shared (numbered steps)
- parsing: regex `\\d+. (.+)`

Usage:
    python 05_inference_val.py --model-dir checkpoints/exp_full/final --output data/eval/predictions.jsonl --dataset hotpotqa
    python 05_inference_val.py --model-dir checkpoints/exp_full/final --output data/eval/predictions.jsonl --dataset nq --sample 500
"""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from importlib import import_module
_val_loader = import_module("01_val_loader")
load_val_questions = _val_loader.load_val_questions

MODEL_ID = "Qwen/Qwen3-8B"

# Same as 04_sft_data_and_train.py
SYSTEM_PROMPT = """You are an information retrieval planning expert. Given a question, generate an ordered sequence of concrete retrieval steps required to find the answer. Each step represents one retrieval action.

Write one step per line, numbered. No other text."""

_STEP_RE = re.compile(r"^\s*\d+[\.\)]\s*(.+)", re.MULTILINE)


def get_free_gpus(n: int = 1, min_free_mib: int = 10_000) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    free = []
    for line in result.stdout.strip().splitlines():
        idx, free_mib = line.split(", ")
        if int(free_mib) >= min_free_mib:
            free.append(idx.strip())
    return free[:n]


def run_inference(model_dir: Path, output_path: Path, dataset: str, sample: int | None = None, batch_size: int = 8,
                  id_file: Path | None = None):
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        free_gpus = get_free_gpus(n=1)
        if not free_gpus:
            raise RuntimeError("No available GPUs.")
        os.environ["CUDA_VISIBLE_DEVICES"] = free_gpus[0]
        print(f"Using GPU (auto-detected): {free_gpus[0]}")
    else:
        print(f"Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

    items = load_val_questions(dataset, sample)
    if id_file is not None:
        target_ids = set(id_file.read_text().strip().splitlines())
        items = [item for item in items if item["id"] in target_ids]
        print(f"ID file filter: {len(target_ids)} requested → {len(items)} matched")
    print(f"Running inference on: {len(items)} items | dataset: {dataset} | model: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, str(model_dir))
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    n_empty = 0

    # Check for already-inferred IDs → resume
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f_existing:
            for line in f_existing:
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        if done_ids:
            print(f"{len(done_ids)} already complete, resuming the rest")

    with open(output_path, "a", encoding="utf-8") as f_out:
        for i in tqdm(range(0, len(items), batch_size), desc="inference"):
            batch_raw = items[i: i + batch_size]
            batch = [item for item in batch_raw if item["id"] not in done_ids]
            if not batch:
                continue

            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Question: {item['question']}"},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for item in batch
            ]

            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            input_lengths = inputs["input_ids"].shape[1]
            generated = outputs[:, input_lengths:]
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)

            for item, text in zip(batch, decoded):
                text = text.strip()
                steps = [m.group(1).strip() for m in _STEP_RE.finditer(text)]
                if not steps:
                    # fallback: if output has no numbering, treat the whole thing as 1 step
                    steps = [text] if text else []

                r = {"id": item["id"], "predicted_steps": steps}
                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += 1
                if not steps:
                    n_empty += 1

    print(f"Saved: {total} → {output_path} (parse failures: {n_empty})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path, help="Path to the LoRA checkpoint (final/)")
    parser.add_argument("--output", required=True, type=Path, help="Path to save predictions")
    parser.add_argument("--dataset", required=True, choices=["nq", "hotpotqa"], help="Dataset to evaluate")
    parser.add_argument("--sample", type=int, default=None, help="Number to sample from val (default: all)")
    parser.add_argument("--batch-size", type=int, default=8, help="Inference batch size (default: 8)")
    parser.add_argument("--id-file", type=Path, default=None, help="File listing IDs to process (one per line)")
    args = parser.parse_args()
    run_inference(args.model_dir, args.output, args.dataset, args.sample, args.batch_size,
                  args.id_file)
