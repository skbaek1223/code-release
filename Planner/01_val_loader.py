"""
Validation 데이터 로더 (hard_selected − goals 방식)

hard_selected 에서 goals 로 이미 생성된 항목을 제외하고,
데이터셋별로 N 개를 샘플링하여 val 데이터로 사용한다.

1) python 01_val_loader.py              → nq, hotpotqa 전부 precompute
2) python 01_val_loader.py nq           → 특정 데이터셋만 precompute

precompute 결과: data/val/{dataset}_val.jsonl

다른 스크립트에서 사용:
    from 01_val_loader import load_val_items, load_val_questions
"""
from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
VAL_DIR = DATA_DIR / "val"
HARD_SELECTED_DIR = DATA_DIR / "hard_selected"
GOALS_DIR = DATA_DIR / "goals"

VAL_SAMPLE_SIZE = {"nq": 5000, "hotpotqa": 5000}
SEED = 42

DATASETS = list(VAL_SAMPLE_SIZE.keys())


# ── precompute ──────────────────────────────────────────────────

def precompute(dataset: str) -> Path:
    """hard_selected 에서 goals ID 를 제외하고 샘플링하여 val jsonl 저장."""
    if dataset not in VAL_SAMPLE_SIZE:
        raise ValueError(f"지원하지 않는 데이터셋: {dataset} (가능: {DATASETS})")

    VAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VAL_DIR / f"{dataset}_val.jsonl"

    print(f"[{dataset}] precompute 시작 (hard_selected − goals)...")

    # goals 에 있는 id 수집
    goals_path = GOALS_DIR / f"{dataset}_goals.jsonl"
    goals_ids: set[str] = set()
    if goals_path.exists():
        with open(goals_path, encoding="utf-8") as f:
            for line in f:
                goals_ids.add(json.loads(line)["id"])
    print(f"  goals: {len(goals_ids)}개")

    # hard_selected 에서 goals 에 없는 항목 추출
    hard_path = HARD_SELECTED_DIR / f"{dataset}_hard_selected.jsonl"
    if not hard_path.exists():
        raise FileNotFoundError(f"{hard_path} 없음. precompute/04_extract_hard.py 를 먼저 실행하세요.")

    candidates: list[dict] = []
    with open(hard_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["id"] not in goals_ids:
                candidates.append(item)
    print(f"  hard_selected − goals: {len(candidates)}개")

    # 샘플링
    n_sample = VAL_SAMPLE_SIZE[dataset]
    rng = random.Random(SEED)
    if len(candidates) < n_sample:
        print(f"  WARNING: 후보 {len(candidates)}개 < {n_sample}개, 전부 사용")
        n_sample = len(candidates)

    sampled = rng.sample(candidates, n_sample)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[{dataset}] {n_sample}개 저장 → {out_path}")
    return out_path


# ── 로드 인터페이스 (precompute 된 파일에서 읽기) ────────────────

def load_val_items(dataset: str) -> dict[str, dict]:
    """precompute 된 jsonl 파일에서 val 데이터를 로드."""
    path = VAL_DIR / f"{dataset}_val.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음. 먼저 'python 01_val_loader.py {dataset}' 를 실행하세요."
        )
    items: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            items[item["id"]] = item
    print(f"{dataset} val: {len(items)}개 로드 (from {path.name})")
    return items


def load_val_questions(dataset: str, sample: int | None = None, seed: int = 42) -> list[dict]:
    items = load_val_items(dataset)
    questions = [{"id": v["id"], "question": v["question"]} for v in items.values()]
    if sample is not None and sample < len(questions):
        random.seed(seed)
        questions = random.sample(questions, sample)
        print(f"샘플링: {sample}개")
    return questions


# ── main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validation 데이터 precompute (hard_selected − goals)")
    parser.add_argument(
        "datasets", nargs="*", default=DATASETS,
        choices=[*VAL_SAMPLE_SIZE.keys()], help="precompute 할 데이터셋 (기본: 전부)",
    )
    args = parser.parse_args()

    for ds in args.datasets:
        precompute(ds)

    print("\nDone.")
