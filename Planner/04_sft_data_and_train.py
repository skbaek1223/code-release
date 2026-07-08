"""
Step 4: SFT 데이터 생성 + QLoRA SFT 학습

02_generate_goals.py 로 필터링된 goals (*_goals_filtered.jsonl) 을 사용합니다.

Subcommands:
    python 04_sft_data_and_train.py convert            # filtered goals → train/val JSONL 변환
    python 04_sft_data_and_train.py train --train-data data/sft/train.jsonl --output-dir checkpoints/exp_full
"""
import argparse
import json
import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

GOALS_DIR = Path(__file__).parent.parent / "data" / "goals"
SFT_DIR = Path(__file__).parent.parent / "data" / "sft"

SYSTEM_PROMPT = """You are an information retrieval planning expert. Given a question, generate an ordered sequence of concrete retrieval steps required to find the answer. Each step represents one retrieval action.

Write one step per line, numbered. No other text."""

MODEL_ID = "Qwen/Qwen3-8B"
VAL_DATA = Path(__file__).parent.parent / "data" / "sft" / "val.jsonl"


# ── convert (기존 03_make_sft_data.py) ─────────────────────────

def item_to_sft(item: dict) -> dict:
    if "plan" in item:
        steps = [item["plan"]]
    else:
        steps = [s if isinstance(s, str) else s["goal"] for s in item["steps"]]

    output = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))

    return {
        "id": item["id"],
        "dataset": item["dataset"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {item['question']}"},
            {"role": "assistant", "content": output},
        ],
    }


def convert(val_ratio: float = 0.05):
    SFT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    TRAIN_DATASETS = {"nq", "hotpotqa"}
    goal_files = sorted(
        p for p in GOALS_DIR.glob("*_goals_filtered.jsonl")
        if p.stem.replace("_goals_filtered", "") in TRAIN_DATASETS
    )
    if not goal_files:
        raise FileNotFoundError(f"goals 파일 없음: {GOALS_DIR}")

    all_items: list[dict] = []
    for path in goal_files:
        items = [json.loads(line) for line in open(path)]
        dataset_name = path.stem.replace("_goals_filtered", "")
        print(f"{dataset_name}: {len(items)}개 로드")
        all_items.extend(items)

    random.shuffle(all_items)
    print(f"전체: {len(all_items)}개")

    val_count = max(1, int(len(all_items) * val_ratio))
    val = all_items[:val_count]
    train = all_items[val_count:]

    val_path = SFT_DIR / "val.jsonl"
    with open(val_path, "w") as f:
        for item in val:
            f.write(json.dumps(item_to_sft(item), ensure_ascii=False) + "\n")
    print(f"val 저장: {len(val)}개 → {val_path}")

    train_path = SFT_DIR / "train.jsonl"
    with open(train_path, "w") as f:
        for item in train:
            f.write(json.dumps(item_to_sft(item), ensure_ascii=False) + "\n")
    print(f"train 저장: {len(train)}개 → {train_path}")


# ── train (기존 04_sft_train.py) ───────────────────────────────

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from trl import SFTConfig, SFTTrainer


@dataclass
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """assistant 응답 토큰에만 loss 계산."""

    response_template: Union[str, List[int]] = None
    ignore_index: int = -100
    mlm: bool = False

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.response_template, str):
            self.response_token_ids = self.tokenizer.encode(
                self.response_template, add_special_tokens=False
            )
        else:
            self.response_token_ids = list(self.response_template)

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(batch["labels"])):
            seq = batch["input_ids"][i].tolist()
            original_labels = batch["labels"][i].clone()
            batch["labels"][i] = torch.full_like(batch["labels"][i], self.ignore_index)
            response_starts = []
            for idx in range(len(seq) - len(self.response_token_ids) + 1):
                if seq[idx: idx + len(self.response_token_ids)] == self.response_token_ids:
                    response_starts.append(idx + len(self.response_token_ids))
            if not response_starts:
                warnings.warn("No response template found in an example; all labels masked.")
                continue
            for j, start in enumerate(response_starts):
                end = (
                    response_starts[j + 1] - len(self.response_token_ids)
                    if j + 1 < len(response_starts)
                    else len(seq)
                )
                batch["labels"][i, start:end] = original_labels[start:end]
        return batch


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def get_free_gpus(min_free_mib: int = 10000) -> list[int]:
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    free_gpus = []
    for line in result.stdout.strip().splitlines():
        idx, free_mib = line.split(", ")
        if int(free_mib) >= min_free_mib:
            free_gpus.append(int(idx))
    return free_gpus


def train(train_data: Path, output_dir: Path):
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        free_gpus = get_free_gpus()
        if not free_gpus:
            raise RuntimeError("사용 가능한 GPU가 없습니다.")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(free_gpus[0])
        print(f"사용 GPU (자동 감지): {free_gpus}")
    else:
        print(f"사용 GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

    print(f"train data: {train_data} ({sum(1 for _ in open(train_data))}개)")
    print(f"output dir: {output_dir}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Model (4-bit QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    train_raw = load_jsonl(train_data)
    val_raw = load_jsonl(VAL_DATA)

    def preprocess(examples):
        return {
            "text": [
                tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                for msgs in examples["messages"]
            ]
        }

    remove_cols = ["messages", "id", "dataset"]
    train_ds = Dataset.from_list(train_raw).map(preprocess, batched=True, remove_columns=remove_cols)
    val_ds = Dataset.from_list(val_raw).map(preprocess, batched=True, remove_columns=remove_cols)

    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n",
        tokenizer=tokenizer,
    )

    # Training
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        dataloader_num_workers=8,
        dataset_text_field="text",
        max_length=512,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )

    ckpt = None
    ckpts = sorted(output_dir.glob("checkpoint-*"))
    if ckpts:
        ckpt = str(ckpts[-1])
        print(f"체크포인트에서 재개: {ckpt}")
    trainer.train(resume_from_checkpoint=ckpt)
    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    print(f"모델 저장 → {final_dir}")


# ── CLI ────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("convert", help="goals → train/val JSONL 변환")

    train_parser = sub.add_parser("train", help="QLoRA SFT 학습")
    train_parser.add_argument("--train-data", required=True, type=Path, help="train JSONL 경로")
    train_parser.add_argument("--output-dir", required=True, type=Path, help="체크포인트 저장 디렉토리")

    args = parser.parse_args()
    if args.command == "convert":
        convert()
    elif args.command == "train":
        train(args.train_data, args.output_dir)
