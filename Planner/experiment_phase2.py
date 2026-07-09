"""
Phase 2: size sweep

Set the Phase 1 optimal ratio in RATIO below, then run.

Usage:
    python experiment_phase2.py              # run everything from step 1
    HF_HUB_OFFLINE=1 python experiment_phase2.py --start 4
    # start from step 3 (sample+train)
    python experiment_phase2.py --train-only  # train only, save model, then exit

Steps:
    1) generate        — generate goals once, sized for the largest experiment
    2) convert+sample  — convert goals → SFT, then sample per-size train data
    3) train           — allocate one GPU per size, train in parallel
    4) infer+eval      — unified worker handles infer first, runs eval as slots free up
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
from importlib import import_module
from pathlib import Path

PIPELINE_DIR = Path(__file__).parent
ROOT = PIPELINE_DIR.parent
GOALS_DIR = ROOT / "data" / "goals"
SFT_DIR = ROOT / "data" / "sft"
EXP_DIR = ROOT / "data" / "experiments"

TRAIN_JSONL = SFT_DIR / "train.jsonl"
VAL_RATIO = 0.05

# Phase 1 optimal ratio (sums to 1000) — adjust based on Phase 1 results
RATIO = {"nq": 300, "hotpotqa": 700}

SIZES = [9000, 10000]

EVAL_DATASETS = ["nq", "hotpotqa"]
SEED = 42
ALLOWED_GPUS = [1, 3, 5, 7]
N_TRAIN_GPUS = 3
N_INFER_GPUS = 4
N_INFER_SPLITS = 2  # how many GPUs to split each (tag, ds) infer job across
N_EVAL_SPLITS = 2   # how many GPUs to split each (tag, ds) eval job across


# ── GPU detection ────────────────────────────────────────────────────

def detect_free_gpus(n: int, min_free_mib: int = 10_000,
                     strict: bool = False) -> list[int]:
    """Find and return up to n free GPUs.

    Only GPUs in ALLOWED_GPUS are considered candidates.
    strict=True raises RuntimeError if fewer than n are found;
    strict=False returns as many as are available (at least 1 required).
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    gpus: list[int] = []
    for line in result.stdout.strip().splitlines():
        idx, free = line.split(", ")
        gpu_idx = int(idx.strip())
        if gpu_idx not in ALLOWED_GPUS:
            continue
        if int(free.strip()) >= min_free_mib:
            gpus.append(gpu_idx)
        if len(gpus) == n:
            break
    if len(gpus) < n:
        if strict:
            raise RuntimeError(
                f"Needed {n} free GPUs, only found {len(gpus)} (threshold: {min_free_mib} MiB)")
        if len(gpus) == 0:
            raise RuntimeError(
                f"0 free GPUs — at least 1 required (threshold: {min_free_mib} MiB)")
        print(f"⚠ Requested {n} GPUs but only {len(gpus)} available, proceeding with {len(gpus)}")
    return gpus


# ── step helpers ────────────────────────────────────────────────

def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in open(path))


def count_goals(dataset: str) -> int:
    path = GOALS_DIR / f"{dataset}_goals.jsonl"
    if not path.exists():
        return 0
    return sum(1 for _ in open(path))


def step_generate(counts: dict[str, int]) -> bool:
    """Check and generate goals. Returns True if any were newly generated."""
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
                ds, "--limit", str(deficit), "--skip-judge",
            ]
            subprocess.run(cmd, check=True)
            need_rebuild = True
        else:
            print(f"  {ds}: have {n_goals_exist} goals ≥ {n_goals_needed} needed, sufficient")
    return need_rebuild


def step_convert():
    """Convert raw goals → SFT train/val JSONL."""
    print("\n=== [convert] raw goals → SFT conversion ===")
    sys.path.insert(0, str(PIPELINE_DIR))
    sft_mod = import_module("04_sft_data_and_train")

    SFT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    TRAIN_DATASETS = {"nq", "hotpotqa"}
    goal_files = sorted(
        p for p in GOALS_DIR.glob("*_goals.jsonl")
        if p.stem.replace("_goals", "") in TRAIN_DATASETS
    )
    if not goal_files:
        raise FileNotFoundError(f"No goals files found: {GOALS_DIR}")

    all_items: list[dict] = []
    for path in goal_files:
        items = [json.loads(line) for line in open(path)]
        dataset_name = path.stem.replace("_goals", "")
        print(f"  {dataset_name}: loaded {len(items)} (raw)")
        all_items.extend(items)

    rng.shuffle(all_items)
    print(f"  total: {len(all_items)}")

    val_count = max(1, int(len(all_items) * VAL_RATIO))
    val = all_items[:val_count]
    train = all_items[val_count:]

    val_path = SFT_DIR / "val.jsonl"
    with open(val_path, "w") as f:
        for item in val:
            f.write(json.dumps(sft_mod.item_to_sft(item), ensure_ascii=False) + "\n")
    print(f"  saved val: {len(val)} → {val_path}")

    train_path = SFT_DIR / "train.jsonl"
    with open(train_path, "w") as f:
        for item in train:
            f.write(json.dumps(sft_mod.item_to_sft(item), ensure_ascii=False) + "\n")
    print(f"  saved train: {len(train)} → {train_path}")


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
        print(f"  [{tag}] {ds}: sampled {n}")
    rng.shuffle(subset)

    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(train_path, "w") as f:
        for item in subset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    config = {"tag": tag, "counts": counts, "seed": seed, "total": len(subset)}
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[{tag}] sample complete: {len(subset)} → {train_path}")


# ── infer split/merge ──────────────────────────────────────────

def _prepare_split_infer(tag: str, ds: str, model_dir: Path,
                         n_splits: int = N_INFER_SPLITS,
                         ) -> list[tuple]:
    """Split the remaining inference targets into n_splits, writing an ID file for each.

    Returns: [(tag, ds, model_dir, part_pred, id_file, part_idx), ...]
    Returns an empty list if already complete.
    """
    val_path = ROOT / "data" / "val" / f"{ds}_val.jsonl"
    pred_path = EXP_DIR / tag / f"predictions_{ds}.jsonl"

    all_ids = []
    with open(val_path) as f:
        for line in f:
            all_ids.append(json.loads(line)["id"])

    done_ids: set[str] = set()
    if pred_path.exists():
        with open(pred_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    remaining = [id_ for id_ in all_ids if id_ not in done_ids]
    if not remaining:
        return []

    exp_dir = EXP_DIR / tag
    chunk_size = math.ceil(len(remaining) / n_splits)
    tasks = []
    for i in range(n_splits):
        chunk = remaining[i * chunk_size : (i + 1) * chunk_size]
        if not chunk:
            continue
        id_file = exp_dir / f"_ids_{ds}_part{i}.txt"
        with open(id_file, "w") as f:
            f.write("\n".join(chunk) + "\n")
        part_pred = exp_dir / f"predictions_{ds}_part{i}.jsonl"
        if part_pred.exists():
            part_pred.unlink()
        tasks.append((tag, ds, model_dir, part_pred, id_file, i))

    print(f"[{tag}] infer {ds}: {len(remaining)} remaining → split into {len(tasks)}")
    return tasks


def _merge_part_predictions(tag: str, ds: str):
    """Append part files into the main predictions file, then clean up."""
    exp_dir = EXP_DIR / tag
    pred_path = exp_dir / f"predictions_{ds}.jsonl"
    parts = sorted(exp_dir.glob(f"predictions_{ds}_part*.jsonl"))
    if not parts:
        return
    with open(pred_path, "a") as f_out:
        for part in parts:
            with open(part) as f_in:
                for line in f_in:
                    f_out.write(line)
            part.unlink()
    for id_file in exp_dir.glob(f"_ids_{ds}_part*.txt"):
        id_file.unlink()
    print(f"[{tag}] infer {ds}: merged {len(parts)} parts")


def _enqueue_infer_splits(tag: str, ds: str, model_dir: Path,
                          infer_queue: queue.Queue,
                          eval_queue: queue.Queue,
                          infer_parts_remaining: dict,
                          infer_parts_lock: threading.Lock,
                          eval_parts_remaining: dict,
                          eval_parts_lock: threading.Lock):
    """Push infer split tasks onto the queue, or push straight to eval if already done."""
    sub_tasks = _prepare_split_infer(tag, ds, model_dir)
    if not sub_tasks:
        print(f"[{tag}] infer {ds} already complete, skipping")
        _enqueue_eval_splits(tag, ds, eval_queue,
                             eval_parts_remaining, eval_parts_lock)
        return
    with infer_parts_lock:
        infer_parts_remaining[(tag, ds)] = len(sub_tasks)
    for st in sub_tasks:
        infer_queue.put(st)


# ── eval split ─────────────────────────────────────────────────

def _prepare_split_eval(tag: str, ds: str,
                        n_splits: int = N_EVAL_SPLITS,
                        ) -> list[tuple]:
    """Split the remaining eval targets into n_splits, writing a part prediction file for each.

    Returns: [(tag, ds, part_pred, part_judge, part_idx), ...]
    Returns an empty list if already complete.
    """
    exp_dir = EXP_DIR / tag
    pred_path = exp_dir / f"predictions_{ds}.jsonl"
    judge_path = exp_dir / f"judge_{ds}.jsonl"

    # Collect all prediction IDs
    all_ids = []
    with open(pred_path) as f:
        for line in f:
            try:
                all_ids.append(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass

    # Exclude IDs already judged
    done_ids: set[str] = set()
    if judge_path.exists():
        with open(judge_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    remaining = [id_ for id_ in all_ids if id_ not in done_ids]
    if not remaining:
        return []

    # Read the prediction rows corresponding to the remaining IDs
    pred_by_id: dict[str, str] = {}
    with open(pred_path) as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj["id"] in remaining:
                    pred_by_id[obj["id"]] = line
            except (json.JSONDecodeError, KeyError):
                pass

    chunk_size = math.ceil(len(remaining) / n_splits)
    tasks = []
    for i in range(n_splits):
        chunk = remaining[i * chunk_size : (i + 1) * chunk_size]
        if not chunk:
            continue
        part_pred = exp_dir / f"predictions_{ds}_eval_part{i}.jsonl"
        with open(part_pred, "w") as f:
            for id_ in chunk:
                if id_ in pred_by_id:
                    f.write(pred_by_id[id_])
        part_judge = exp_dir / f"judge_{ds}_part{i}.jsonl"
        if part_judge.exists():
            part_judge.unlink()
        tasks.append((tag, ds, part_pred, part_judge, i))

    print(f"[{tag}] eval {ds}: {len(remaining)} remaining → split into {len(tasks)}")
    return tasks


def _merge_part_judges(tag: str, ds: str):
    """Append part judge files into the main judge file, then clean up."""
    exp_dir = EXP_DIR / tag
    judge_path = exp_dir / f"judge_{ds}.jsonl"
    parts = sorted(exp_dir.glob(f"judge_{ds}_part*.jsonl"))
    if not parts:
        return
    with open(judge_path, "a") as f_out:
        for part in parts:
            with open(part) as f_in:
                for line in f_in:
                    f_out.write(line)
            part.unlink()
    # Also clean up part prediction files
    for pp in exp_dir.glob(f"predictions_{ds}_eval_part*.jsonl"):
        pp.unlink()
    print(f"[{tag}] eval {ds}: merged {len(parts)} parts")


def _enqueue_eval_splits(tag: str, ds: str,
                         eval_queue: queue.Queue,
                         eval_parts_remaining: dict,
                         eval_parts_lock: threading.Lock):
    """Push eval split tasks onto the queue, or mark as done if already complete."""
    sub_tasks = _prepare_split_eval(tag, ds)
    if not sub_tasks:
        print(f"[{tag}] eval {ds} already complete, skipping")
        eval_queue.put(("__done__", tag, ds))
        return
    with eval_parts_lock:
        eval_parts_remaining[(tag, ds)] = len(sub_tasks)
    for st in sub_tasks:
        eval_queue.put(st)


# ── train ─────────────────────────────────────────────────────

def run_train_on_gpu(tag: str, counts: dict[str, int], gpu_id: int):
    """Run train on the given GPU."""
    exp_dir = EXP_DIR / tag
    final_dir = exp_dir / "checkpoints" / "final"
    if final_dir.exists():
        print(f"[{tag}] train already complete ({final_dir}), skipping")
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    total = sum(counts.values())
    print(f"\n{'='*60}")
    print(f"[{tag}] GPU {gpu_id} | {counts} (total {total})")
    print(f"{'='*60}")

    print(f"[{tag}] starting train (GPU {gpu_id})")
    subprocess.run([
        sys.executable, str(PIPELINE_DIR / "04_sft_data_and_train.py"), "train",
        "--train-data", str(exp_dir / "train.jsonl"),
        "--output-dir", str(exp_dir / "checkpoints"),
    ], env=env, check=True)
    print(f"[{tag}] train complete")

    # After train completes, remove stale infer/eval results to force a rerun
    for ds in EVAL_DATASETS:
        for pattern in [f"predictions_{ds}.jsonl", f"judge_{ds}.jsonl"]:
            old = exp_dir / pattern
            if old.exists():
                old.unlink()
                print(f"[{tag}] removed stale {pattern} (will rerun)")


# ── eval helper ───────────────────────────────────────────────

def _do_one_eval(eval_mod, tag: str, ds: str, client, model_name: str,
                 gpu_id: int, pred_path: Path = None, judge_path: Path = None,
                 part_idx: int = None):
    """Evaluate a single eval task (full or part)."""
    exp_dir = EXP_DIR / tag
    if pred_path is None:
        pred_path = exp_dir / f"predictions_{ds}.jsonl"
    if judge_path is None:
        judge_path = exp_dir / f"judge_{ds}.jsonl"

    label = f"{tag}/{ds}" if part_idx is None else f"{tag}/{ds} part{part_idx}"

    # Skip eval that's already complete
    if judge_path.exists() and _count_lines(judge_path) >= _count_lines(pred_path):
        print(f"[GPU {gpu_id}] eval {label} already complete, skipping")
        return
    if judge_path.exists():
        judge_path.unlink()
    print(f"[GPU {gpu_id}] starting eval {label}")
    eval_mod._evaluate_dataset(
        pred_path, judge_path, ds, client, model_name, eval_mod.NUM_WORKERS,
    )
    print(f"[GPU {gpu_id}] eval {label} complete")


# ── unified worker (infer first + eval concurrently) ──────────────────────

def _start_workers(
    gpu_pool: queue.Queue,
    infer_queue: queue.Queue,
    eval_queue: queue.Queue,
    all_submitted: threading.Event,
    n_total_eval: int,
    n_workers: int,
    infer_parts_remaining: dict,
    infer_parts_lock: threading.Lock,
    eval_parts_remaining: dict,
    eval_parts_lock: threading.Lock,
) -> list[threading.Thread]:
    """Unified infer/eval worker.

    Processes split infer subtasks from infer_queue first when present; once all
    have been assigned, processes eval split tasks from eval_queue. Eval runs
    concurrently rather than waiting for all infer to finish.
    """
    from openai import OpenAI
    sys.path.insert(0, str(PIPELINE_DIR))
    eval_mod = import_module("06_evaluate_val")

    eval_completed = [0]
    counter_lock = threading.Lock()
    all_eval_done = threading.Event()

    def worker(worker_id: int):
        port = 8010 + worker_id

        while not all_eval_done.is_set():
            # ── 1) infer takes priority (split subtasks) ────────────
            infer_task = None
            try:
                infer_task = infer_queue.get_nowait()
            except queue.Empty:
                pass

            if infer_task is not None:
                tag, ds, model_dir, part_pred, id_file, part_idx = infer_task
                gpu_id = gpu_pool.get()
                try:
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                    cmd = [
                        sys.executable, str(PIPELINE_DIR / "05_inference_val.py"),
                        "--model-dir", str(model_dir),
                        "--output", str(part_pred),
                        "--dataset", ds,
                        "--id-file", str(id_file),
                    ]
                    print(f"[{tag}] starting infer {ds} part{part_idx} (GPU {gpu_id})")
                    subprocess.run(cmd, env=env, check=True)
                    print(f"[{tag}] infer {ds} part{part_idx} complete (GPU {gpu_id})")
                finally:
                    gpu_pool.put(gpu_id)
                # Once all parts are done, merge → submit eval split tasks
                with infer_parts_lock:
                    infer_parts_remaining[(tag, ds)] -= 1
                    if infer_parts_remaining[(tag, ds)] == 0:
                        _merge_part_predictions(tag, ds)
                        _enqueue_eval_splits(tag, ds, eval_queue,
                                             eval_parts_remaining,
                                             eval_parts_lock)
                continue

            # ── 2) eval processing (split subtasks, infer keeps priority) ──
            eval_task = None
            try:
                eval_task = eval_queue.get_nowait()
            except queue.Empty:
                # Both empty: check for completion, or wait
                if all_submitted.is_set() and infer_queue.empty():
                    with counter_lock:
                        if eval_completed[0] >= n_total_eval:
                            all_eval_done.set()
                            return
                time.sleep(1)
                continue

            # For (tag, ds) already complete, just bump the counter and move on
            if eval_task[0] == "__done__":
                _, tag, ds = eval_task
                with counter_lock:
                    eval_completed[0] += 1
                    if eval_completed[0] >= n_total_eval:
                        all_eval_done.set()
                continue

            tag, ds, part_pred, part_judge, part_idx = eval_task

            gpu_id = gpu_pool.get()
            current_model_key = None
            proc = None
            client = None
            try:
                while True:
                    # Yield to any unassigned infer task even mid-eval
                    if not infer_queue.empty():
                        eval_queue.put((tag, ds, part_pred, part_judge, part_idx))
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
                                 cfg["model_name"], gpu_id,
                                 pred_path=part_pred, judge_path=part_judge,
                                 part_idx=part_idx)

                    # Part done → merge once all parts are done
                    with eval_parts_lock:
                        eval_parts_remaining[(tag, ds)] -= 1
                        if eval_parts_remaining[(tag, ds)] == 0:
                            _merge_part_judges(tag, ds)
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
                        eval_task = eval_queue.get_nowait()
                    except queue.Empty:
                        break

                    if eval_task[0] == "__done__":
                        _, tag, ds = eval_task
                        with counter_lock:
                            eval_completed[0] += 1
                            if eval_completed[0] >= n_total_eval:
                                all_eval_done.set()
                        if all_eval_done.is_set():
                            break
                        try:
                            eval_task = eval_queue.get_nowait()
                        except queue.Empty:
                            break

                    tag, ds, part_pred, part_judge, part_idx = eval_task
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
    val_loader = import_module("01_val_loader")
    for ds in EVAL_DATASETS:
        val_path = val_loader.VAL_DIR / f"{ds}_val.jsonl"
        if val_path.exists() and val_path.stat().st_size > 0:
            print(f"  {ds}: already exists ({val_path}), skipping")
        else:
            val_loader.precompute(ds)


def main(start: int = 1, train_only: bool = False):
    n_exp = len(SIZES)
    ratio_total = sum(RATIO.values())

    # 0. Precompute val data
    step_precompute_val()

    # 1. Generate: generate goals sized for the largest experiment
    if start <= 1:
        max_size = max(SIZES)
        max_counts = {ds: n * max_size // ratio_total for ds, n in RATIO.items()}
        print(f"=== generating goals sized for the largest experiment ({max_size}) ===")
        step_generate(max_counts)

    # 2. Convert + Sample: convert goals → SFT, then sample per size
    tags: list[str] = [f"S{size}" for size in SIZES]
    if start <= 2:
        step_convert()
        for size in SIZES:
            tag = f"S{size}"
            counts = {ds: n * size // ratio_total for ds, n in RATIO.items()}
            step_sample(tag, counts, SEED)

    # 3. Train only (--train-only)
    if train_only and start <= 3:
        train_gpus = detect_free_gpus(N_TRAIN_GPUS, strict=True)
        print(f"\n{'='*60}")
        print(f"Parallel training (train only): {n_exp} size experiments (GPUs: {train_gpus})")
        print(f"{'='*60}")

        train_futures = {}
        with ThreadPoolExecutor(max_workers=N_TRAIN_GPUS) as train_pool:
            for size, gpu_id in zip(SIZES, train_gpus):
                tag = f"S{size}"
                counts = {ds: n * size // ratio_total for ds, n in RATIO.items()}
                f = train_pool.submit(run_train_on_gpu, tag, counts, gpu_id)
                train_futures[f] = tag

            for f in as_completed(train_futures):
                tag = train_futures[f]
                try:
                    f.result()
                    print(f"\n✓ [{tag}] train complete")
                except Exception as e:
                    print(f"\n✗ [{tag}] failed: {e}")
                    raise

        print("\n=== Train only complete ===")
        return

    # 3+4. Train → Infer+Eval pipeline (start infer as soon as train finishes)
    gpu_pool: queue.Queue[int] = queue.Queue()
    infer_queue: queue.Queue[tuple] = queue.Queue()
    eval_queue: queue.Queue[tuple] = queue.Queue()
    all_submitted = threading.Event()
    n_total_eval = n_exp * len(EVAL_DATASETS)
    infer_parts_remaining: dict[tuple[str, str], int] = {}
    infer_parts_lock = threading.Lock()
    eval_parts_remaining: dict[tuple[str, str], int] = {}
    eval_parts_lock = threading.Lock()

    if start <= 3:
        train_gpus = detect_free_gpus(N_TRAIN_GPUS, strict=True)
        print(f"\n{'='*60}")
        print(f"Parallel training + immediate infer: {n_exp} size experiments (GPUs: {train_gpus})")
        print(f"{'='*60}")

        # Start infer+eval workers first (GPU pool is empty, so they'll wait)
        worker_threads = _start_workers(
            gpu_pool, infer_queue, eval_queue, all_submitted,
            n_total_eval, N_INFER_GPUS,
            infer_parts_remaining, infer_parts_lock,
            eval_parts_remaining, eval_parts_lock,
        )

        def train_then_enqueue(tag, counts, gpu_id):
            """After train completes, hand its GPU to the infer pool and submit infer tasks."""
            run_train_on_gpu(tag, counts, gpu_id)
            # Hand the freed GPU over to the infer pool
            gpu_pool.put(gpu_id)
            print(f"[{tag}] GPU {gpu_id} → handed to infer pool")
            # Submit infer split tasks for this tag
            exp_dir = EXP_DIR / tag
            ckpt_dir = exp_dir / "checkpoints" / "final"
            for ds in EVAL_DATASETS:
                _enqueue_infer_splits(
                    tag, ds, ckpt_dir, infer_queue, eval_queue,
                    infer_parts_remaining, infer_parts_lock,
                    eval_parts_remaining, eval_parts_lock,
                )

        train_futures = {}
        with ThreadPoolExecutor(max_workers=N_TRAIN_GPUS) as train_pool:
            for size, gpu_id in zip(SIZES, train_gpus):
                tag = f"S{size}"
                counts = {ds: n * size // ratio_total for ds, n in RATIO.items()}
                f = train_pool.submit(train_then_enqueue, tag, counts, gpu_id)
                train_futures[f] = tag

            for f in as_completed(train_futures):
                tag = train_futures[f]
                try:
                    f.result()
                    print(f"\n✓ [{tag}] train+infer submission complete")
                except Exception as e:
                    print(f"\n✗ [{tag}] failed: {e}")
                    raise

        # All train complete → no more infer tasks will be added
        all_submitted.set()

    else:
        # start == 4: skip train, run infer+eval only
        infer_gpus = detect_free_gpus(N_INFER_GPUS, strict=True)
        print(f"\n{'='*60}")
        print(f"Starting Infer + Eval (GPUs: {infer_gpus})")
        print(f"{'='*60}")

        for g in infer_gpus:
            gpu_pool.put(g)

        for tag in tags:
            exp_dir = EXP_DIR / tag
            ckpt_dir = exp_dir / "checkpoints" / "final"
            for ds in EVAL_DATASETS:
                _enqueue_infer_splits(
                    tag, ds, ckpt_dir, infer_queue, eval_queue,
                    infer_parts_remaining, infer_parts_lock,
                    eval_parts_remaining, eval_parts_lock,
                )
        all_submitted.set()

        worker_threads = _start_workers(
            gpu_pool, infer_queue, eval_queue, all_submitted,
            n_total_eval, N_INFER_GPUS,
            infer_parts_remaining, infer_parts_lock,
            eval_parts_remaining, eval_parts_lock,
        )

    for t in worker_threads:
        t.join()

    print("\n=== Phase 2 complete ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2: size sweep")
    parser.add_argument("--start", type=int, default=1, choices=[1, 2, 3, 4],
                        help="starting step (1=generate, 2=convert+sample, 3=train, 4=infer+eval)")
    parser.add_argument("--train-only", action="store_true",
                        help="run train step only, save model, then exit (skip infer+eval)")
    args = parser.parse_args()
    main(start=args.start, train_only=args.train_only)
