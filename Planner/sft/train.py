"""
Qwen3-8B QLoRA SFT

CUDA_VISIBLE_DEVICES=1 python train.py
"""
import json
from pathlib import Path

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import os
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from trl import SFTTrainer, SFTConfig


@dataclass
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """Only compute loss on assistant response tokens (removed from trl 0.24.0)."""

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

MODEL_ID = "Qwen/Qwen3-8B"
DATA_DIR = Path(__file__).parent.parent / "data" / "sft"
OUTPUT_DIR = Path(__file__).parent.parent / "checkpoints" / "qwen3-8b-planner"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def format_messages(tokenizer, example: dict) -> str:
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )


def get_free_gpus(min_free_mib: int = 10000) -> list[int]:
    """Return indices of GPUs with at least min_free_mib free memory.
    Must be called before CUDA is initialized."""
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    free_gpus = []
    for line in result.stdout.strip().splitlines():
        idx, free_mib = line.split(", ")
        if int(free_mib) >= min_free_mib:
            free_gpus.append(int(idx))
    return free_gpus


def main():
    # Auto-detect only if CUDA_VISIBLE_DEVICES wasn't set externally
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        free_gpus = get_free_gpus()
        if not free_gpus:
            raise RuntimeError("No available GPUs.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, free_gpus))
        print(f"Using GPUs (auto-detected): {free_gpus}")
    else:
        print(f"Using GPUs: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Model (4-bit QLoRA)
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

    # 3. LoRA
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

    # 4. Dataset
    train_raw = load_jsonl(DATA_DIR / "train.jsonl")
    val_raw = load_jsonl(DATA_DIR / "val.jsonl")

    def preprocess(examples):
        texts = []
        for msgs in examples["messages"]:
            text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            texts.append(text)
        return {"text": texts}

    train_ds = Dataset.from_list(train_raw).map(preprocess, batched=True, remove_columns=["messages", "id", "dataset"])
    val_ds = Dataset.from_list(val_raw).map(preprocess, batched=True, remove_columns=["messages", "id", "dataset"])

    # Compute loss only on assistant responses
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # 5. Training
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch = 16
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
        report_to="none",  # switch to "wandb" to enable W&B logging
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

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "final"))
    print(f"Model saved → {OUTPUT_DIR / 'final'}")


if __name__ == "__main__":
    main()
