"""
Phase 1: ratio sweep (total fixed at 1000)

1) generate  — generate all goals needed across every experiment at once + SFT conversion
2) sample + train — allocate one GPU per condition, run in parallel
3) infer + eval — unified worker handles infer first, runs eval as slots free up
"""
import json
import math
import os
import queue
import random
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
ROOT = PIPELINE_DIR.parent
GOALS_DIR = ROOT / "data" / "goals"
SFT_DIR = ROOT / "data" / "sft"
EXP_DIR = ROOT / "data" / "experiments"

TRAIN_JSONL = SFT_DIR / "train.jsonl"
VAL_RATIO = 0.05

TRAIN_DATASETS = ["nq", "hotpotqa"]

EXPERIMENTS = [
    {"tag": "R1", "nq": 300, "hotpotqa": 700},
    {"tag": "R2", "nq": 400, "hotpotqa": 600},
    {"tag": "R3", "nq": 500, "hotpotqa": 500}
]

EVAL_DATASETS = ["nq", "hotpotqa"]
VAL_COUNTS = {"nq": 5000, "hotpotqa": 5000}
SEED = 42


# ── GPU detection ────────────────────────────────────────────────────

def detect_free_gpus(n: int, min_free_mib: int = 10_000) -> list[int]:
    """Find and return at least n free GPUs."""
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    gpus: list[int] = []
    for line in result.stdout.strip().splitlines():
        idx, free = line.split(", ")
        if int(free.strip()) >= min_free_mib:
            gpus.append(int(idx.strip()))
        if len(gpus) == n:
            break
    if len(gpus) < n:
        raise RuntimeError(
            f"Needed {n} free GPUs, only found {len(gpus)} (threshold: {min_free_mib} MiB)")
    return gpus


# ── step helpers ────────────────────────────────────────────────

def count_goals(dataset: str) -> int:
    path = GOALS_DIR / f"{dataset}_goals.jsonl"
    if not path.exists():
        return 0
    return sum(1 for _ in open(path))


def step_generate(counts: dict[str, int]):
    """Generate any goals shortfall per dataset and convert to SFT."""
    print("\n=== [generate] checking and generating goals ===")
    need_rebuild = False
    for ds, n_train in counts.items():
        if n_train == 0:
            continue
        n_goals_needed = math.ceil(n_train / (1 - VAL_RATIO))
        n_goals_exist = count_goals(ds)
        deficit = n_goals_needed - n_goals_exist
        if deficit > 0:
            print(f"  {ds}: have {n_goals_exist} goals → need {n_goals_needed}, generating +{deficit}")
            cmd = [
                sys.executable, str(PIPELINE_DIR / "02_generate_goals.py"),
                ds, "--limit", str(deficit),
            ]
            subprocess.run(cmd, check=True)
            need_rebuild = True
        else:
            print(f"  {ds}: have {n_goals_exist} goals ≥ {n_goals_needed} needed, sufficient")

    if need_rebuild or not TRAIN_JSONL.exists():
        print("\n  Running 04_sft_data_and_train.py convert...")
        cmd = [sys.executable, str(PIPELINE_DIR / "04_sft_data_and_train.py"), "convert"]
        subprocess.run(cmd, check=True)
    else:
        print("\n  No change in goals, skipping SFT conversion")


def load_train_by_dataset(path: Path) -> dict[str, list[dict]]:
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            by_dataset[item["dataset"]].append(item)
    return by_dataset


def step_sample(tag: str, counts: dict[str, int], seed: int):
    exp_dir = EXP_DIR / tag
    train_path = exp_dir / "train.jsonl"
    config_path = exp_dir / "config.json"

    # Skip if a file already exists sampled with the same config
    if train_path.exists() and config_path.exists():
        with open(config_path) as f:
            existing = json.load(f)
        if existing.get("counts") == counts and existing.get("seed") == seed:
            print(f"[{tag}] sample already exists ({existing['total']}), skipping")
            return

    print(f"[{tag}] starting sample")
    by_dataset = load_train_by_dataset(TRAIN_JSONL)

    rng = random.Random(seed)
    subset = []
    for ds, n in counts.items():
        if n == 0:
            continue
        pool = by_dataset.get(ds, [])
        if len(pool) < n:
            print(f"  [{tag}] WARNING: {ds} requested {n} > available {len(pool)}, using all")
            n = len(pool)
        sampled = rng.sample(pool, n)
        subset.extend(sampled)
    rng.shuffle(subset)

    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(train_path, "w") as f:
        for item in subset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    config = {"tag": tag, "counts": counts, "seed": seed, "total": len(subset)}
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[{tag}] sample complete: {len(subset)} → {train_path}")


# ── sample + train → submit tasks to infer_queue/eval_queue ─────

def run_experiment_on_gpu(tag: str, counts: dict[str, int], gpu_id: int,
                          gpu_pool: queue.Queue, infer_queue: queue.Queue,
                          eval_queue: queue.Queue):
    """Run sample → train on the given GPU, then submit an infer task to the queue."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    total = sum(counts.values())
    print(f"\n{'='*60}")
    print(f"[{tag}] GPU {gpu_id} | nq={counts['nq']}, hotpotqa={counts['hotpotqa']} (total {total})")
    print(f"{'='*60}")

    # sample (CPU)
    step_sample(tag, counts, SEED)

    # train — skip if a checkpoint already exists
    exp_dir = EXP_DIR / tag
    ckpt_dir = exp_dir / "checkpoints" / "final"
    if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
        print(f"[{tag}] train already complete ({ckpt_dir}), skipping")
    else:
        print(f"[{tag}] starting train (GPU {gpu_id})")
        subprocess.run([
            sys.executable, str(PIPELINE_DIR / "04_sft_data_and_train.py"), "train",
            "--train-data", str(exp_dir / "train.jsonl"),
            "--output-dir", str(exp_dir / "checkpoints"),
        ], env=env, check=True)
        print(f"[{tag}] train complete")

    # train complete → return this experiment's GPU to the pool
    gpu_pool.put(gpu_id)

    # Submit infer tasks to the queue
    for ds in EVAL_DATASETS:
        pred_path = exp_dir / f"predictions_{ds}.jsonl"
        if pred_path.exists():
            n_done = sum(1 for _ in open(pred_path))
            if n_done >= VAL_COUNTS[ds]:
                print(f"[{tag}] infer {ds} complete ({n_done}), skipping")
                eval_queue.put((tag, ds))
                continue
            print(f"[{tag}] infer {ds} incomplete ({n_done}/{VAL_COUNTS[ds]}), will resume")
        infer_queue.put((tag, ds, ckpt_dir, pred_path))


# ── eval helper ───────────────────────────────────────────────

def _do_one_eval(eval_mod, tag: str, ds: str, client, model_name: str,
                 gpu_id: int):
    """Evaluate a single (tag, ds). Skips if already complete."""
    exp_dir = EXP_DIR / tag
    pred_path = exp_dir / f"predictions_{ds}.jsonl"
    judge_path = exp_dir / f"judge_{ds}.jsonl"

    if judge_path.exists():
        n_judged = sum(1 for _ in open(judge_path))
        if n_judged >= VAL_COUNTS[ds]:
            print(f"[GPU {gpu_id}] eval {tag}/{ds} already complete ({n_judged}), skipping")
            return
    print(f"[GPU {gpu_id}] starting eval {tag}/{ds}")
    eval_mod._evaluate_dataset(
        pred_path, judge_path, ds, client, model_name, eval_mod.NUM_WORKERS,
    )
    print(f"[GPU {gpu_id}] eval {tag}/{ds} complete")


# ── unified worker (infer first + eval concurrently) ──────────────────────

def _start_workers(
    gpu_pool: queue.Queue,
    infer_queue: queue.Queue,
    eval_queue: queue.Queue,
    all_submitted: threading.Event,
    n_total_eval: int,
    n_workers: int,
) -> list[threading.Thread]:
    """Unified infer/eval worker.

    Processes unassigned tasks from infer_queue first when present; once all
    have been assigned, processes eval tasks from eval_queue. Eval runs
    concurrently rather than waiting for all infer to finish.
    """
    from openai import OpenAI
    sys.path.insert(0, str(PIPELINE_DIR))
    from importlib import import_module
    eval_mod = import_module("06_evaluate_val")

    eval_completed = [0]
    counter_lock = threading.Lock()
    all_eval_done = threading.Event()

    def worker(worker_id: int):
        port = 8010 + worker_id

        while not all_eval_done.is_set():
            # ── 1) infer takes priority ──────────────────────────────
            infer_task = None
            try:
                infer_task = infer_queue.get_nowait()
            except queue.Empty:
                pass

            if infer_task is not None:
                tag, ds, model_dir, pred_path = infer_task
                gpu_id = gpu_pool.get()
                try:
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    cmd = [
                        sys.executable, str(PIPELINE_DIR / "05_inference_val.py"),
                        "--model-dir", str(model_dir),
                        "--output", str(pred_path),
                        "--dataset", ds,
                    ]
                    print(f"[{tag}] starting infer {ds} (GPU {gpu_id})")
                    subprocess.run(cmd, env=env, check=True)
                    print(f"[{tag}] infer {ds} complete (GPU {gpu_id})")
                finally:
                    gpu_pool.put(gpu_id)
                eval_queue.put((tag, ds))
                continue

            # ── 2) eval processing (non-blocking, infer keeps priority) ──
            try:
                tag, ds = eval_queue.get_nowait()
            except queue.Empty:
                # Both empty: check for completion, or wait
                if all_submitted.is_set() and infer_queue.empty():
                    with counter_lock:
                        if eval_completed[0] >= n_total_eval:
                            all_eval_done.set()
                            return
                time.sleep(1)
                continue

            gpu_id = gpu_pool.get()
            current_model_key = None
            proc = None
            client = None
            try:
                while True:
                    # Yield to any unassigned infer task even mid-eval
                    if not infer_queue.empty():
                        eval_queue.put((tag, ds))
                        break

                    cfg = eval_mod.DATASET_CONFIGS[ds]
                    mk = (cfg["model_name"], cfg["model_path"])
                    if mk != current_model_key:
                        if proc:
                            eval_mod.stop_vllm(proc, port)
                        proc = eval_mod.start_vllm(
                            cfg["model_path"], cfg["model_name"],
                            port, str(gpu_id),
                        )
                        client = OpenAI(
                            api_key="EMPTY",
                            base_url=f"http://localhost:{port}/v1",
                        )
                        current_model_key = mk
                    _do_one_eval(eval_mod, tag, ds, client,
                                 cfg["model_name"], gpu_id)

                    with counter_lock:
                        eval_completed[0] += 1
                        if eval_completed[0] >= n_total_eval:
                            all_eval_done.set()

                    if all_eval_done.is_set():
                        break

                    # Yield and tear down vLLM if an unassigned infer task appears
                    if not infer_queue.empty():
                        break

                    # Next eval task (non-blocking)
                    try:
                        tag, ds = eval_queue.get_nowait()
                    except queue.Empty:
                        break
            finally:
                if proc:
                    eval_mod.stop_vllm(proc, port)
                gpu_pool.put(gpu_id)

    threads = []
    for i in range(n_workers):
        t = threading.Thread(target=worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)
    return threads


# ── main ────────────────────────────────────────────────────────

def step_precompute_val():
    """Precompute val data (skip if it already exists)."""
    print("\n=== [precompute] checking val data ===")
    from importlib import import_module
    val_loader = import_module("01_val_loader")
    for ds in EVAL_DATASETS:
        val_path = val_loader.VAL_DIR / f"{ds}_val.jsonl"
        if val_path.exists() and val_path.stat().st_size > 0:
            print(f"  {ds}: already exists ({val_path}), skipping")
        else:
            val_loader.precompute(ds)


def main():
    n_exp = len(EXPERIMENTS)

    # 0. Acquire GPUs
    gpu_ids = detect_free_gpus(n_exp)
    print(f"Using GPUs: {gpu_ids} ({n_exp} conditions)")

    # 0.5. Precompute val data (used by 05, 06)
    step_precompute_val()

    # 1. Generate: generate the max count needed across all experiments at once
    max_counts: dict[str, int] = {}
    for exp in EXPERIMENTS:
        for ds in TRAIN_DATASETS:
            max_counts[ds] = max(max_counts.get(ds, 0), exp[ds])
    print(f"Max needed per dataset: {max_counts}")
    step_generate(max_counts)

    # 2. Create queues
    gpu_pool: queue.Queue[int] = queue.Queue()
    infer_queue: queue.Queue[tuple] = queue.Queue()
    eval_queue: queue.Queue[tuple[str, str]] = queue.Queue()
    all_submitted = threading.Event()

    # 3. Start unified workers (infer first, eval concurrently)
    n_total_eval = n_exp * len(EVAL_DATASETS)
    worker_threads = _start_workers(
        gpu_pool, infer_queue, eval_queue, all_submitted, n_total_eval, n_exp,
    )

    # 4. Sample + Train → submit tasks to infer_queue / eval_queue
    print(f"\n{'='*60}")
    print(f"Running in parallel: {n_exp} conditions × {n_exp} GPUs")
    print(f"{'='*60}")

    train_futures = {}
    with ThreadPoolExecutor(max_workers=n_exp) as train_pool:
        for exp, gpu_id in zip(EXPERIMENTS, gpu_ids):
            counts = {ds: exp[ds] for ds in TRAIN_DATASETS}
            f = train_pool.submit(
                run_experiment_on_gpu, exp["tag"], counts, gpu_id,
                gpu_pool, infer_queue, eval_queue,
            )
            train_futures[f] = exp["tag"]

        for f in as_completed(train_futures):
            tag = train_futures[f]
            try:
                f.result()
                print(f"\n✓ [{tag}] sample+train complete, infer/eval submitted")
            except Exception as e:
                print(f"\n✗ [{tag}] failed: {e}")
                raise

    # All train complete → no more new tasks
    all_submitted.set()
    print(f"\n=== All tasks submitted, waiting for workers to finish ===")

    # Wait for workers to finish
    for t in worker_threads:
        t.join()

    print("\n=== Phase 1 complete ===")


if __name__ == "__main__":
    main()
