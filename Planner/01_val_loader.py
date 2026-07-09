"""
Validation data loader (hard_selected minus goals)

Excludes items from hard_selected that already have goals generated, then
samples N per dataset to use as val data.

1) python 01_val_loader.py              → precompute both nq and hotpotqa
2) python 01_val_loader.py nq           → precompute a specific dataset only

Precompute output: data/val/{dataset}_val.jsonl

Used by other scripts as:
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
    """Exclude goals IDs from hard_selected, sample, and save the val jsonl."""
    if dataset not in VAL_SAMPLE_SIZE:
        raise ValueError(f"Unsupported dataset: {dataset} (available: {DATASETS})")

    VAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VAL_DIR / f"{dataset}_val.jsonl"

    print(f"[{dataset}] starting precompute (hard_selected − goals)...")

    # Collect IDs present in goals
    goals_path = GOALS_DIR / f"{dataset}_goals.jsonl"
    goals_ids: set[str] = set()
    if goals_path.exists():
        with open(goals_path, encoding="utf-8") as f:
            for line in f:
                goals_ids.add(json.loads(line)["id"])
    print(f"  goals: {len(goals_ids)}")

    # Extract items from hard_selected that aren't in goals
    hard_path = HARD_SELECTED_DIR / f"{dataset}_hard_selected.jsonl"
    if not hard_path.exists():
        raise FileNotFoundError(f"{hard_path} not found. Run precompute/04_extract_hard.py first.")

    candidates: list[dict] = []
    with open(hard_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["id"] not in goals_ids:
                candidates.append(item)
    print(f"  hard_selected − goals: {len(candidates)}")

    # Sample
    n_sample = VAL_SAMPLE_SIZE[dataset]
    rng = random.Random(SEED)
    if len(candidates) < n_sample:
        print(f"  WARNING: {len(candidates)} candidates < {n_sample}, using all")
        n_sample = len(candidates)

    sampled = rng.sample(candidates, n_sample)

    with open(out_path, "w", encoding="utf-8") as f:
        for item in sampled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[{dataset}] saved {n_sample} → {out_path}")
    return out_path


# ── load interface (reads from precomputed files) ────────────────

def load_val_items(dataset: str) -> dict[str, dict]:
    """Load val data from the precomputed jsonl file."""
    path = VAL_DIR / f"{dataset}_val.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run 'python 01_val_loader.py {dataset}' first."
        )
    items: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            items[item["id"]] = item
    print(f"{dataset} val: loaded {len(items)} (from {path.name})")
    return items


def load_val_questions(dataset: str, sample: int | None = None, seed: int = 42) -> list[dict]:
    items = load_val_items(dataset)
    questions = [{"id": v["id"], "question": v["question"]} for v in items.values()]
    if sample is not None and sample < len(questions):
        random.seed(seed)
        questions = random.sample(questions, sample)
        print(f"Sampled: {sample}")
    return questions


# ── main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Precompute validation data (hard_selected − goals)")
    parser.add_argument(
        "datasets", nargs="*", default=DATASETS,
        choices=[*VAL_SAMPLE_SIZE.keys()], help="Datasets to precompute (default: all)",
    )
    args = parser.parse_args()

    for ds in args.datasets:
        precompute(ds)

    print("\nDone.")
