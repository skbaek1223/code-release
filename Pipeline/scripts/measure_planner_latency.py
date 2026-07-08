"""Measure planner-only wall-clock latency per question, per dataset.

Mirrors the planner block in run_re_guide.py but times only the vLLM
`generate()` call so model-load and tokenization costs are excluded. The
planner is invoked as a single batched call over all questions in the
dataset, matching production. Per-question latency is reported as
wall_time / N.

Usage:
    # Single GPU
    CUDA_VISIBLE_DEVICES=4 python measure_planner_latency.py \
        --datasets nq,ambigqa,hotpotqa,musique \
        --output planner_latency.json

    # 2 GPUs in parallel (TP=1 each): nq and ambigqa start first, one per
    # GPU; whichever GPU frees up next pulls hotpotqa, the other musique.
    python measure_planner_latency.py --gpus 6,7 \
        --datasets nq,ambigqa,hotpotqa,musique \
        --output planner_latency.json

To match production conditions for the 14B Re-Guide runs, use TP=1, since
`run_all_datasets_model.py --preset r1_qwen14b` splits --gpus into
per-dataset TP=1 workers.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _HERE
DATA_DIR = os.path.normpath(os.path.join(_PROJECT_ROOT, "..", "..", "data", "eval"))
DEFAULT_STEPS_MODEL = os.path.normpath(os.path.join(
    _PROJECT_ROOT, "..", "..", "data", "experiments", "S7000",
    "checkpoints", "final_merged"
))

DATASET_FILES = {
    "nq":       "nq_test.json",
    "ambigqa":  "ambigqa_dev.json",
    "hotpotqa": "hotpotqa_dev.json",
    "musique":  "musique_dev.json",
    "triviaqa": "triviaqa_test.json",
    "2wiki":    "2wiki_dev.json",
}

STEPS_SYSTEM = ("You are an information retrieval planning expert. Given a question, "
                "generate an ordered sequence of concrete retrieval steps required to "
                "find the answer. Each step represents one retrieval action.\n\n"
                "Write one step per line, numbered. No other text.")


def load_questions(dataset_name, subset_num=-1):
    path = os.path.join(DATA_DIR, DATASET_FILES[dataset_name])
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(stripped)
    else:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if subset_num > 0:
        data = data[:subset_num]
    return [item["Question"] for item in data]


def build_prompts(tok, questions):
    return [
        tok.apply_chat_template(
            [{"role": "system", "content": STEPS_SYSTEM},
             {"role": "user",   "content": f"Question: {q}"}],
            tokenize=False, add_generation_prompt=True)
        for q in questions
    ]


def _run_one_dataset(d, tok, llm, sp, subset_num, gpu_label):
    qs = load_questions(d, subset_num=subset_num)
    prompts = build_prompts(tok, qs)
    n = len(prompts)
    print(f"[{gpu_label}][{d}] N={n}, starting timed batch", flush=True)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, sampling_params=sp)
    wall = time.perf_counter() - t0

    out_tok_total = sum(len(o.outputs[0].token_ids) for o in outs)
    per_q_ms = wall / n * 1000.0
    avg_out_tok = out_tok_total / n

    print(f"[{gpu_label}][{d}] total {wall:.2f}s | per-question {per_q_ms:.1f} ms "
          f"| avg out toks {avg_out_tok:.1f}", flush=True)

    return {
        "n_questions": n,
        "total_wall_s": round(wall, 4),
        "per_question_ms": round(per_q_ms, 1),
        "avg_output_tokens": round(avg_out_tok, 1),
        "tp": 1,
    }


def _worker(gpu_id, task_queue, result_queue, args):
    """Per-GPU worker: load LLM once, pull datasets from queue until drained."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    label = f"gpu{gpu_id}"

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"[{label}] loading planner model: {args.steps_model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.steps_model_path, trust_remote_code=True)
    llm = LLM(
        model=args.steps_model_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.90,
        dtype="half",
    )
    sp = SamplingParams(max_tokens=1024, temperature=0.0, top_p=1.0,
                        repetition_penalty=1.1)

    warmed = False
    while True:
        d = task_queue.get()
        if d is None:
            break

        if not warmed and args.warmup > 0:
            warm_qs = load_questions(d, subset_num=args.warmup)
            warm_prompts = build_prompts(tok, warm_qs)
            print(f"[{label}][warmup] generating {len(warm_prompts)} prompts (untimed)",
                  flush=True)
            _ = llm.generate(warm_prompts, sampling_params=sp)
            warmed = True

        payload = _run_one_dataset(d, tok, llm, sp, args.subset_num, label)
        payload["gpu"] = gpu_id
        result_queue.put((d, payload))


def _run_parallel(gpu_list, datasets, args):
    # Strip parent CUDA_VISIBLE_DEVICES so each spawned worker binds cleanly
    # to the GPU id it was assigned.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    for d in datasets:
        task_queue.put(d)
    for _ in gpu_list:
        task_queue.put(None)

    workers = []
    for gpu_id in gpu_list:
        p = ctx.Process(target=_worker,
                        args=(gpu_id, task_queue, result_queue, args))
        p.start()
        workers.append(p)

    collected = {}
    while len(collected) < len(datasets):
        d, payload = result_queue.get()
        collected[d] = payload

    for p in workers:
        p.join()

    return {d: collected[d] for d in datasets if d in collected}


def _run_sequential(datasets, args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    visible = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
    tp = max(1, len(visible))
    print(f"[setup] CUDA_VISIBLE_DEVICES={visible or '<all>'} -> TP={tp}")
    print(f"[setup] planner model: {args.steps_model_path}")

    tok = AutoTokenizer.from_pretrained(args.steps_model_path, trust_remote_code=True)
    llm = LLM(
        model=args.steps_model_path,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.90,
        dtype="half",
    )
    sp = SamplingParams(max_tokens=1024, temperature=0.0, top_p=1.0,
                        repetition_penalty=1.1)

    if args.warmup > 0 and datasets:
        warm_qs = load_questions(datasets[0], subset_num=args.warmup)
        warm_prompts = build_prompts(tok, warm_qs)
        print(f"[warmup] generating {len(warm_prompts)} prompts (untimed)")
        _ = llm.generate(warm_prompts, sampling_params=sp)

    results = {}
    for d in datasets:
        payload = _run_one_dataset(d, tok, llm, sp, args.subset_num, "seq")
        payload["tp"] = tp
        results[d] = payload
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps_model_path", type=str, default=DEFAULT_STEPS_MODEL)
    ap.add_argument("--datasets", type=str, default="nq,ambigqa,hotpotqa,musique")
    ap.add_argument("--gpus", type=str, default=None,
                    help="Comma-separated GPU IDs to run in parallel, one "
                         "vLLM worker per GPU (TP=1). Datasets are pulled "
                         "from a shared queue. If omitted, runs sequentially "
                         "using CUDA_VISIBLE_DEVICES.")
    ap.add_argument("--subset_num", type=int, default=-1,
                    help="Limit per-dataset question count (debug). -1 = all.")
    ap.add_argument("--warmup", type=int, default=8,
                    help="Number of warmup questions to run before timed batch.")
    ap.add_argument("--output", type=str, default="planner_latency.json")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    if args.gpus:
        gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
        print(f"[setup] parallel mode, GPUs={gpu_list}, datasets={datasets}")
        results = _run_parallel(gpu_list, datasets, args)
    else:
        results = _run_sequential(datasets, args)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[done] wrote {args.output}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
