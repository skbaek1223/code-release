"""
Step 4: Extract only judge_pass=False items into hard_selected

- judge_pass=True  → Qwen produced a correct plan → dropped (easy question)
- judge_pass=False → plan failed → included in hard_selected

Output: data/hard_selected/{dataset}_hard_selected.jsonl
Usage: python 04_extract_hard.py [nq|hotpotqa ...]  (default: all)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUT_DIR = DATA_DIR / "hard_selected"

# Meta fields to strip from the judge results
DROP_FIELDS = {"predicted_steps", "judge_pass", "judge_reason"}


def run(dataset: str):
    in_path = DATA_DIR / "precompute" / f"{dataset}_all_judged.jsonl"
    out_path = OUT_DIR / f"{dataset}_hard_selected.jsonl"

    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found. Run 03_judge_retrieval_steps_{dataset}.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    hard = 0
    easy = 0
    no_verdict = 0

    with open(in_path, encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            total += 1
            if "judge_pass" not in it:
                no_verdict += 1
            elif it["judge_pass"]:
                easy += 1
            else:
                hard += 1
                clean = {k: v for k, v in it.items() if k not in DROP_FIELDS}
                f_out.write(json.dumps(clean, ensure_ascii=False) + "\n")

    print(f"[{dataset}] Total judged:  {total}")
    print(f"  PASS (easy): {easy} ({100*easy/max(total,1):.1f}%) → excluded")
    print(f"  FAIL (hard): {hard} ({100*hard/max(total,1):.1f}%) → hard_selected")
    if no_verdict:
        print(f"  No verdict:  {no_verdict} → excluded")
    print(f"\nSaved {hard} hard_selected items → {out_path}")


ALL_DATASETS = ("nq", "hotpotqa")

if __name__ == "__main__":
    datasets = sys.argv[1:] if len(sys.argv) > 1 else list(ALL_DATASETS)
    for ds in datasets:
        assert ds in ALL_DATASETS, f"Unknown dataset: {ds}. Choose from {ALL_DATASETS}"
        run(ds)
