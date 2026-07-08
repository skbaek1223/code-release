"""
Step 5: SFT 모델로 val 예측 생성

val 데이터: hard_selected 중 goals 미생성 항목 (01_val_loader 에서 precompute)

04_sft_data_and_train.py 와 동일한 프롬프트·출력 형식을 사용합니다.
- system prompt: 통일 (번호 매긴 steps)
- 파싱: regex `\\d+. (.+)`

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

# 04_sft_data_and_train.py 와 동일
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
            raise RuntimeError("사용 가능한 GPU가 없습니다.")
        os.environ["CUDA_VISIBLE_DEVICES"] = free_gpus[0]
        print(f"사용 GPU (자동 감지): {free_gpus[0]}")
    else:
        print(f"사용 GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")

    items = load_val_questions(dataset, sample)
    if id_file is not None:
        target_ids = set(id_file.read_text().strip().splitlines())
        items = [item for item in items if item["id"] in target_ids]
        print(f"ID 파일 필터: {len(target_ids)}개 요청 → {len(items)}개 매칭")
    print(f"추론 대상: {len(items)}개 | 데이터셋: {dataset} | 모델: {model_dir}")

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

    # 이미 추론된 ID 확인 → 이어쓰기
    done_ids: set[str] = set()
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f_existing:
            for line in f_existing:
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        if done_ids:
            print(f"기존 {len(done_ids)}개 완료, 나머지 이어쓰기")

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
                    # fallback: 번호 없이 출력된 경우 전체를 1개 step으로
                    steps = [text] if text else []

                r = {"id": item["id"], "predicted_steps": steps}
                f_out.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += 1
                if not steps:
                    n_empty += 1

    print(f"저장: {total}개 → {output_path} (파싱 실패: {n_empty}개)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, type=Path, help="LoRA 체크포인트 경로 (final/)")
    parser.add_argument("--output", required=True, type=Path, help="예측 결과 저장 경로")
    parser.add_argument("--dataset", required=True, choices=["nq", "hotpotqa"], help="평가 데이터셋")
    parser.add_argument("--sample", type=int, default=None, help="val에서 샘플링할 개수 (기본: 전체)")
    parser.add_argument("--batch-size", type=int, default=8, help="추론 배치 크기 (기본: 8)")
    parser.add_argument("--id-file", type=Path, default=None, help="처리할 ID 목록 파일 (한 줄에 하나)")
    args = parser.parse_args()
    run_inference(args.model_dir, args.output, args.dataset, args.sample, args.batch_size,
                  args.id_file)
