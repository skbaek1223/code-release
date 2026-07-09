"""
Step 6: Evaluate the SFT model's val outputs with LLM-as-Judge

Val data: hard_selected items that didn't get goals generated (precomputed in 01_val_loader)

Uses the same prompts, models, and parsing logic as precompute/03_a,b,c.
- NQ (single-hop): Qwen2.5-14B, single-hop-specific prompt
- HotpotQA (multi-hop): Qwen2.5-32B, multi-hop-specific prompt

Usage:
    python 06_evaluate_val.py --predictions pred.jsonl --dataset hotpotqa
    python 06_evaluate_val.py --predictions pred.jsonl --dataset nq --output results.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from openai import OpenAI

from importlib import import_module
_val_loader = import_module("01_val_loader")
load_val_items = _val_loader.load_val_items

EVAL_DIR = Path(__file__).parent.parent / "data" / "eval"

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "12"))
MAX_RETRIES = 3
MAX_TOKENS = 256
MAX_CONTEXT_CHARS = 24_000
MIN_FREE_MB = int(os.environ.get("MIN_FREE_MB", "15000"))

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_VERDICT_RE = re.compile(r"^\s*(PASS|FAIL)\s*$", re.MULTILINE | re.IGNORECASE)


# ── per-dataset config ─────────────────────────────────────────────

DATASET_CONFIGS = {
    "nq": {
        "model_name": "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
        "model_path": "/mnt/raid6/skbaek1223/models/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
        "port": 8001,
    },
    "hotpotqa": {
        "model_name": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        "model_path": "/mnt/raid6/skbaek1223/models/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        "port": 8002,
    },
}

# ── NQ single-hop prompt (same as 03_a) ──────────────────────────

NQ_JUDGE_SYSTEM = """You are a quality judge for a single-hop retrieval plan.

You will be given a question, a predicted retrieval plan, the supporting context, and the answer.

Pass if the retrieval plan is targeted enough that a retrieval system would reliably return the supporting context and the answer can be confidently derived from it.

Respond in exactly two lines:
Line 1: PASS or FAIL
Line 2: One sentence — explain why the step passes or fails.

Example (PASS):
Question: where was donovan mitchell picked in the draft
Predicted retrieval plan: Find Donovan Mitchell's draft pick number.
Supporting context:
Mitchell was drafted by the Denver Nuggets with the 13th overall pick in the 2017 NBA draft only to be traded to the Utah Jazz for the 24th pick ( Tyler Lydon ) and Trey Lyles . On July 5 , 2017 , Mitchell signed a four - year rookie scale contract with the Jazz .
Answer: 13th

PASS
The step directly asks for Donovan Mitchell's draft pick number, which reliably retrieves the supporting context about his 13th overall selection from which the answer can be confidently derived.

Example (FAIL):
Question: what does april's baby have on grey's anatomy
Predicted retrieval plan: Find the plot or specific details about April's baby in the TV show Grey's Anatomy.
Supporting context:
Not long after their fight , April realizes she is pregnant . April and Jackson 's baby is diagnosed during pregnancy with Osteogenesis Imperfecta type 2 , and learn that the baby will not survive long after birth . Jackson believes that termination is the best option , however April would rather give birth to the baby knowing it will not live very long .
Answer: Osteogenesis Imperfecta type 2

FAIL
The step asks for general plot details about April's baby rather than specifically asking what medical condition the baby was diagnosed with, so it would retrieve many unrelated plot points instead of reliably targeting the supporting context about the Osteogenesis Imperfecta diagnosis."""

# ── Multi-hop prompt (same as 03_b, 03_c) ────────────────────────

MULTIHOP_JUDGE_SYSTEM = """You are a quality judge for a multi-hop retrieval plan.

You will be given a question, predicted retrieval steps, numbered context sources, and the answer.

Pass if:
1. The retrieval steps are targeted enough that a retrieval system would reliably return all context sources needed to derive the answer.
2. The answer can be confidently derived from them.

Respond in exactly two lines:
Line 1: PASS or FAIL
Line 2: One sentence — explain why the plan passes or fails.

Example (PASS):
Question: The Oberoi family is part of a hotel company that has a head office in what city?
Step 1: Find which hotel company the Oberoi family is part of.
Step 2: Find the city where that hotel company has its head office.
[Source 1] The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.
[Source 2] The Oberoi Group is a hotel company with its head office in Delhi.  Founded in 1934, the company owns and/or operates 30+ luxury hotels and two river cruise ships in six countries, primarily under its Oberoi Hotels & Resorts and Trident Hotels brands.
Answer: Delhi

PASS
Step 1 targets [Source 1] to retrieve the hotel company name (The Oberoi Group), and Step 2 uses that name to target [Source 2] for the head office city — both sources are reachable in the correct order, and each step's goal is specific enough to retrieve exactly the bridging fact needed.

Example (FAIL):
Question: Of the film directors Kenji Mizoguchi and Andrzej Żuławski, which one often went against mainstream commercialism in his films?
Step 1: Find information about Kenji Mizoguchi's approach to mainstream commercialism in his films.
Step 2: Find information about Andrzej Żuławski's approach to mainstream commercialism in his films.
[Source 1] Kenji Mizoguchi (溝口 健二 , Mizoguchi Kenji , May 16, 1898 – August 24, 1956) was a Japanese film director and screenwriter.
[Source 2] Andrzej Żuławski (22 November 1940 – 17 February 2016) was a Polish film director and writer.  He was born in Lwów, Poland (now Ukraine).  Żuławski often went against mainstream commercialism in his films, and enjoyed success mostly with European art-house audiences.
Answer: Andrzej Żuławski

FAIL
Both steps vaguely ask to "find information about" each director's approach to mainstream commercialism rather than specifically asking whether each director went against it — the overly broad queries would retrieve general biographical information instead of reliably targeting the specific fact in [Source 2] about Żuławski's anti-commercial stance."""


# ── NQ context truncation (same as 03_a) ────────────────────────────

def _truncate_context(ctx_text: str, answer: str) -> str:
    sents = _SENT_SPLIT.split(ctx_text)

    if len(sents) >= 2:
        sup_idx = None
        for i, s in enumerate(sents):
            if answer in s:
                sup_idx = i
                break
        if sup_idx is None:
            sup_idx = len(sents) // 2

        sup_sent = sents[sup_idx]
        budget = MAX_CONTEXT_CHARS - len(sup_sent)

        before = " ".join(sents[:sup_idx])
        after = " ".join(sents[sup_idx + 1:])
        half = budget // 2

        if before and len(before) > half:
            before = "[...] " + before[-(half - 6):]
        if after and len(after) > half:
            after = after[:half - 6] + " [...]"

        parts = [p for p in (before, sup_sent, after) if p]
        return " ".join(parts)

    words = ctx_text.split()
    ans_idx = None
    for i, w in enumerate(words):
        if answer in w:
            ans_idx = i
            break
    if ans_idx is None:
        ans_idx = 0

    half = MAX_CONTEXT_CHARS // 2
    before_words = []
    used = 0
    for w in reversed(words[:ans_idx]):
        cost = len(w) + 1
        if used + cost > half:
            break
        before_words.append(w)
        used += cost
    before_words.reverse()
    after_words = []
    used = 0
    for w in words[ans_idx:]:
        cost = len(w) + 1
        if used + cost > half:
            break
        after_words.append(w)
        used += cost

    prefix = "[...] " if len(before_words) < ans_idx else ""
    suffix = " [...]" if len(after_words) < len(words) - ans_idx else ""
    return prefix + " ".join(before_words + after_words) + suffix


# ── prompt construction ────────────────────────────────────────────────

def make_judge_prompt_nq(item: dict) -> str:
    """Same as 03_a: single-hop prompt"""
    lines = [f"Question: {item['question']}"]

    steps = item.get("predicted_steps", [])
    if steps:
        lines.append(f"Predicted retrieval plan: {steps[0]}")

    ctx_parts = [snippet["text"] for snippet in item.get("supporting_context", [])]
    if ctx_parts:
        ctx_text = "\n\n".join(ctx_parts)
        if len(ctx_text) > MAX_CONTEXT_CHARS:
            ctx_text = _truncate_context(ctx_text, item.get("answer", ""))
        lines.append("Supporting context:\n" + ctx_text)

    lines.append(f"Answer: {item['answer']}")
    lines.append("Now evaluate the predicted retrieval plan above.")
    return "\n\n".join(lines)


def make_judge_prompt_multihop(item: dict) -> str:
    """Same as 03_b, 03_c: multi-hop prompt"""
    lines = [f"Question: {item['question']}"]

    steps = item.get("predicted_steps", [])
    lines.append(
        "Predicted retrieval plan:\n"
        + "\n".join(f"  Step {i+1}: {s}" for i, s in enumerate(steps))
    )

    ctx_parts = [f"[Source {i}] {snippet['text']}" for i, snippet in enumerate(item.get("supporting_context", []), 1)]
    if ctx_parts:
        lines.append("Context sources:\n" + "\n\n".join(ctx_parts))

    lines.append(f"Answer: {item['answer']}")
    lines.append("Now evaluate the predicted retrieval plan above.")
    return "\n\n".join(lines)


# ── GPU / vLLM management ──────────────────────────────────────────────

def find_free_gpus(n: int = 1, min_free_mb: int = MIN_FREE_MB) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    gpus: list[str] = []
    for line in result.stdout.strip().splitlines():
        idx, free_mb = line.split(", ")
        if int(free_mb.strip()) >= min_free_mb:
            gpus.append(idx.strip())
        if len(gpus) == n:
            break
    if len(gpus) < n:
        raise RuntimeError(
            f"Not enough free GPUs: need {n} (>= {min_free_mb} MB free), found {len(gpus)}"
        )
    return gpus


def _ping(base_url: str) -> bool:
    try:
        OpenAI(api_key="EMPTY", base_url=base_url).models.list()
        return True
    except Exception:
        return False


def start_vllm(model_path: str, model_name: str, port: int, gpu_id: str, timeout: int = 300) -> subprocess.Popen | None:
    base_url = f"http://localhost:{port}/v1"
    if _ping(base_url):
        print(f"[port {port}] vLLM server already running — reusing it.")
        return None

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "1200")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")

    log_path = f"/tmp/vllm_{port}_stderr.log"
    print(f"[port {port}] Starting vLLM... (model={Path(model_path).name}, GPU={gpu_id})")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--served-model-name", model_name,
            "--tensor-parallel-size", "1",
            "--port", str(port),
            "--dtype", "float16",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=open(log_path, "w"),
    )

    for _ in range(timeout):
        time.sleep(1)
        if proc.poll() is not None:
            raise RuntimeError(f"[port {port}] vLLM process exited unexpectedly. Log: {log_path}")
        if _ping(base_url):
            print(f"[port {port}] vLLM ready. (GPU {gpu_id})")
            return proc

    proc.terminate()
    raise RuntimeError(f"[port {port}] vLLM startup timed out ({timeout}s). Log: {log_path}")


def stop_vllm(proc: subprocess.Popen | None, port: int):
    if proc is None:
        return
    print(f"[port {port}] Stopping vLLM...")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


# ── judging ─────────────────────────────────────────────────────────

def parse_predicted_steps(raw: dict) -> list[str] | None:
    if "predicted_steps" in raw:
        return raw["predicted_steps"]
    if "pred_steps" in raw:
        return raw["pred_steps"]
    if "output" in raw:
        try:
            return json.loads(raw["output"]).get("steps", [])
        except (json.JSONDecodeError, AttributeError):
            return None
    return None


def load_predictions(pred_path: Path) -> dict[str, list[str]]:
    preds: dict[str, list[str]] = {}
    with open(pred_path, encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            steps = parse_predicted_steps(raw)
            if steps is not None:
                preds[raw["id"]] = steps
            else:
                print(f"WARNING: {raw['id']} parse failed — predicted_steps field required")
    return preds


def judge_single(
    item: dict,
    dataset: str,
    oai_client: OpenAI,
    model_name: str,
) -> dict:
    """Called with item already containing predicted_steps and supporting_context. Same as 03_abc."""
    if dataset == "nq":
        system = NQ_JUDGE_SYSTEM
        prompt = make_judge_prompt_nq(item)
    else:
        system = MULTIHOP_JUDGE_SYSTEM
        prompt = make_judge_prompt_multihop(item)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = oai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            )
            content = response.choices[0].message.content.strip()
            verdict_match = _VERDICT_RE.search(content)
            if not verdict_match:
                print(f"  SKIP {item['id']}: no PASS/FAIL found")
                return {
                    "id": item["id"],
                    "pass": False,
                    "reason": "parse_error: no PASS/FAIL found",
                    "predicted_steps": item.get("predicted_steps", []),
                    "dataset": dataset,
                }
            judge_pass = verdict_match.group(1).upper() == "PASS"
            reason = content[verdict_match.end():].strip()
            return {
                "id": item["id"],
                "pass": judge_pass,
                "reason": reason,
                "predicted_steps": item.get("predicted_steps", []),
                "dataset": dataset,
            }
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"  RETRY {item['id']} ({attempt}/{MAX_RETRIES}): {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"  ERROR {item['id']}: {e} (failed after {MAX_RETRIES} retries)")
                return {
                    "id": item["id"],
                    "pass": False,
                    "reason": f"error: {e}",
                    "predicted_steps": item.get("predicted_steps", []),
                    "dataset": dataset,
                }


def print_summary(results: list[dict]) -> None:
    n_total = len(results)
    if n_total == 0:
        print("No evaluation results.")
        return

    n_pass = sum(1 for r in results if r["pass"])
    print("\n=== Evaluation summary ===")
    print(f"Total: {n_total}")
    print(f"PASS: {n_pass} ({100 * n_pass / n_total:.1f}%)")
    print(f"FAIL: {n_total - n_pass} ({100 * (n_total - n_pass) / n_total:.1f}%)")


# ── main ─────────────────────────────────────────────────────────

def _evaluate_dataset(
    pred_path: Path,
    output_path: Path,
    dataset: str,
    oai_client: OpenAI,
    model_name: str,
    max_workers: int,
) -> None:
    """Evaluate a single dataset, assuming vLLM is already running."""
    val_items = load_val_items(dataset)
    preds = load_predictions(pred_path)

    eval_items: list[dict] = []
    for id_, steps in preds.items():
        if id_ in val_items:
            eval_items.append({**val_items[id_], "predicted_steps": steps})

    unknown_ids = [id_ for id_ in preds if id_ not in val_items]
    if unknown_ids:
        print(f"WARNING: {len(unknown_ids)} prediction ids not found in val (skipping): {unknown_ids[:3]}")

    if not eval_items:
        print(f"[{dataset}] No items to evaluate.")
        return

    # Check for already-evaluated IDs → resume
    done_ids: set[str] = set()
    existing_results: list[dict] = []
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f_existing:
            for line in f_existing:
                try:
                    r = json.loads(line)
                    done_ids.add(r["id"])
                    existing_results.append(r)
                except (json.JSONDecodeError, KeyError):
                    pass
        if done_ids:
            print(f"{len(done_ids)} already complete, resuming the rest")

    remaining = [item for item in eval_items if item["id"] not in done_ids]
    if not remaining:
        print(f"[{dataset}] All items already evaluated, skipping")
        return

    print(f"Evaluating: {len(remaining)} items (dataset: {dataset}, model: {model_name}, workers: {max_workers})")

    results: list[dict] = list(existing_results)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(judge_single, item, dataset, oai_client, model_name): item["id"]
            for item in remaining
        }
        with tqdm(total=len(remaining), desc=f"[{dataset}] eval", unit="item") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nResults saved: {output_path}")
    print_summary(results)


def run(pred_path: Path, output_path: Path, dataset: str, max_workers: int = NUM_WORKERS) -> None:
    """Evaluate a single dataset (for CLI use)."""
    cfg = DATASET_CONFIGS[dataset]
    model_name = cfg["model_name"]
    model_path = cfg["model_path"]
    port = cfg["port"]

    gpus = find_free_gpus(1)
    proc = start_vllm(model_path, model_name, port, gpus[0])
    oai_client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")

    try:
        _evaluate_dataset(pred_path, output_path, dataset, oai_client, model_name, max_workers)
    finally:
        stop_vllm(proc, port)


def run_multi(
    tasks: list[tuple[Path, Path, str]],
    max_workers: int = NUM_WORKERS,
) -> None:
    """Evaluate multiple datasets. Groups datasets by model so vLLM is only started once per model.

    tasks: [(pred_path, output_path, dataset), ...]
    """
    from collections import defaultdict

    # Group datasets by model
    by_model: dict[tuple[str, str, int], list[tuple[Path, Path, str]]] = defaultdict(list)
    for pred_path, output_path, dataset in tasks:
        cfg = DATASET_CONFIGS[dataset]
        key = (cfg["model_name"], cfg["model_path"], cfg["port"])
        by_model[key].append((pred_path, output_path, dataset))

    gpus = find_free_gpus(1)

    for (model_name, model_path, port), group in by_model.items():
        ds_names = [ds for _, _, ds in group]
        print(f"\n=== Starting vLLM: {model_name} (datasets: {ds_names}) ===")
        proc = start_vllm(model_path, model_name, port, gpus[0])
        oai_client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")

        try:
            for pred_path, output_path, dataset in group:
                print(f"\n--- Evaluating {dataset} ---")
                _evaluate_dataset(pred_path, output_path, dataset, oai_client, model_name, max_workers)
        finally:
            stop_vllm(proc, port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions", required=True, type=Path,
        help="SFT model prediction JSONL (id + predicted_steps fields)",
    )
    parser.add_argument(
        "--dataset", required=True, choices=["nq", "hotpotqa"],
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to save judge results (default: data/eval/{dataset}_val_judge.jsonl)",
    )
    parser.add_argument(
        "--workers", type=int, default=NUM_WORKERS,
        help="Number of parallel requests (default: 12)",
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = EVAL_DIR / f"{args.dataset}_val_judge.jsonl"
    run(args.predictions, args.output, args.dataset, args.workers)
