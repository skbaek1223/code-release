"""
Step 1: HotpotQA train 데이터 로드 및 전처리

- supporting_facts 기준으로 context snippet 추출
- 출력: data/precompute/hotpotqa_all.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path("/mnt/raid6/skbaek1223/project/Data/hotpotqa")
OUT_DIR = Path(__file__).parent.parent.parent / "data" / "precompute"

TRAIN_FILES = [
    DATA_DIR / "train-00000-of-00002.parquet",
    DATA_DIR / "train-00001-of-00002.parquet",
]

CONTEXT_WINDOW = 1  # supporting sent_id 앞뒤로 몇 문장 포함할지


def extract_supporting_snippets(row) -> list[dict]:
    sf = row["supporting_facts"]
    ctx = row["context"]
    if sf is None or ctx is None:
        return []

    sf_titles = list(sf["title"])
    sf_sent_ids = list(sf["sent_id"])

    context_map: dict[str, list[str]] = {
        t: list(sents)
        for t, sents in zip(ctx["title"], ctx["sentences"])
    }

    sf_indices: dict[str, set[int]] = {}
    for t, sid in zip(sf_titles, sf_sent_ids):
        sf_indices.setdefault(t, set()).add(int(sid))

    titles = list(dict.fromkeys(sf_titles))
    snippets = []
    for t in titles:
        if t not in context_map:
            continue
        all_sents = context_map[t]
        selected = sorted({
            neighbor
            for sf_i in sf_indices.get(t, set())
            for neighbor in range(sf_i - CONTEXT_WINDOW, sf_i + CONTEXT_WINDOW + 1)
            if 0 <= neighbor < len(all_sents)
        })
        if not selected:
            continue
        snippets.append({
            "title": t,
            "text": " ".join(all_sents[i] for i in selected),
        })
    return snippets


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "hotpotqa_all.jsonl"

    skipped = 0
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for p in TRAIN_FILES:
            df = pd.read_parquet(p)
            print(f"Loaded {p.name}: {len(df)} rows")
            for _, row in df.iterrows():
                snippets = extract_supporting_snippets(row)
                if not snippets:
                    skipped += 1
                    continue
                item = {
                    "id": row["id"],
                    "dataset": "hotpotqa",
                    "question": row["question"],
                    "answer": str(row["answer"]),
                    "question_type": row.get("type", ""),
                    "supporting_context": snippets,
                }
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1
            del df

    print(f"Written: {written}, Skipped: {skipped}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
