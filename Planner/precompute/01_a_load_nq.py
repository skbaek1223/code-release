"""
Step 1a: Load and preprocess Natural Questions train data

- Only processes records that have both a long_answer and a short_answer
- Extracts the context sentence directly using the short_answer's start_token (no LLM needed)

- Output: data/precompute/nq_all.jsonl

"""
from __future__ import annotations

import json
import re
from pathlib import Path

DATA_PATH = Path("/mnt/raid6/skbaek1223/project/Data/natural_questions/original/train.jsonl")
OUT_DIR = Path(__file__).parent.parent.parent / "data" / "precompute"
OUT_PATH = OUT_DIR / "nq_all.jsonl"

CONTEXT_WINDOW = 1  # how many sentences to include before/after the matched sentence


# ──────────────────────────────────────────────
# Text processing
# ──────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip()]


def extract_long_answer_sentences(
    row: dict,
) -> tuple[list[str], list[tuple[int, int]]] | tuple[None, None]:
    """
    Split the long_answer text into sentences, and also return the absolute
    token range (first_tok, last_tok) corresponding to each sentence.
    """
    ann = row["annotations"]
    la = ann["long_answer"][0]
    if la["candidate_index"] == -1:
        return None, None

    tokens = row["document"]["tokens"]
    tok_list = tokens["token"]
    is_html = tokens["is_html"]
    la_start, la_end = la["start_token"], la["end_token"]

    # Collect non-HTML tokens along with their absolute indices
    parts: list[str] = []
    abs_indices: list[int] = []
    for i in range(la_start, min(la_end + 1, len(tok_list))):
        if not is_html[i]:
            parts.append(tok_list[i])
            abs_indices.append(i)

    if not parts:
        return None, None

    # Starting character offset of each token within full_text
    char_offsets: list[int] = []
    pos = 0
    for tok in parts:
        char_offsets.append(pos)
        pos += len(tok) + 1  # +1 for the space

    full_text = " ".join(parts)
    sentences = split_sentences(full_text)
    if not sentences:
        return None, None

    # Map each sentence's char range → absolute token range
    token_ranges: list[tuple[int, int]] = []
    search_start = 0
    for sent in sentences:
        sent_char_start = full_text.index(sent, search_start)
        sent_char_end = sent_char_start + len(sent)
        search_start = sent_char_end

        sent_abs_toks = [
            abs_indices[j]
            for j, off in enumerate(char_offsets)
            if sent_char_start <= off < sent_char_end
        ]
        if sent_abs_toks:
            token_ranges.append((sent_abs_toks[0], sent_abs_toks[-1]))
        else:
            token_ranges.append((-1, -1))

    return sentences, token_ranges


# ──────────────────────────────────────────────
# Locating the answer sentence
# ──────────────────────────────────────────────

def find_answer_sentence_idx(
    row: dict,
    token_ranges: list[tuple[int, int]],
) -> int | None:
    """Return the index of the sentence containing the short_answer's start_token."""
    short_answers = row["annotations"]["short_answers"]
    if not short_answers:
        return None

    sa_start_tokens = {
        tok
        for sa in short_answers
        for tok in sa.get("start_token", [])
        if tok != -1
    }
    if not sa_start_tokens:
        return None

    for i, (tok_start, tok_end) in enumerate(token_ranges):
        if tok_start == -1:
            continue
        if any(tok_start <= sa_tok <= tok_end for sa_tok in sa_start_tokens):
            return i

    return None


# ──────────────────────────────────────────────
# Record processing
# ──────────────────────────────────────────────

def process_row(row: dict) -> dict | None:
    ann = row["annotations"]

    # Filter on short answer
    short_ans = ann["short_answers"][0]
    if not short_ans["text"]:
        return None
    answer = short_ans["text"][0]

    question = row["question"]["text"]

    # Split the long answer into sentences (with token ranges)
    sentences, token_ranges = extract_long_answer_sentences(row)
    if not sentences:
        return None

    # Locate the answer sentence's index
    idx = find_answer_sentence_idx(row, token_ranges)
    if idx is None:
        return None

    # Build the snippet within ±CONTEXT_WINDOW
    start = max(0, idx - CONTEXT_WINDOW)
    end = min(len(sentences), idx + CONTEXT_WINDOW + 1)
    context_text = " ".join(sentences[start:end])

    title = row["document"].get("title", "")
    return {
        "id": str(row["id"]),
        "dataset": "nq",
        "question": question,
        "answer": answer,
        "supporting_context": [{"title": title, "text": context_text}],
    }


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done_ids: set[str] = set()
    if OUT_PATH.exists():
        with open(OUT_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except Exception:
                    pass
        print(f"Resume: {len(done_ids)} records already written")

    written = 0
    skipped = 0

    with open(DATA_PATH, encoding="utf-8") as in_f, \
         open(OUT_PATH, "a", encoding="utf-8") as out_f:
        for i, line in enumerate(in_f):
            row = json.loads(line)
            if str(row["id"]) in done_ids:
                continue

            result = process_row(row)
            if result is None:
                skipped += 1
            else:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                written += 1

            if (i + 1) % 10_000 == 0:
                out_f.flush()
                print(f"  [{i+1:,}] written={written:,} skipped={skipped:,}")

    print(f"Done. Written: {written:,}, Skipped: {skipped:,}")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
