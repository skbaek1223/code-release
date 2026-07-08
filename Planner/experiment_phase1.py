"""
Phase 1: 비율 탐색 (총량 1000 고정)

1) generate  — 모든 실험에 필요한 goals 를 한 번에 생성 + SFT 변환
2) sample + train — 대조군별 GPU 1개씩 할당, 병렬 실행
3) infer + eval — 통합 worker 가 infer 우선 처리, 배정 완료 시 eval 병행
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


# ── GPU 감지 ────────────────────────────────────────────────────

def detect_free_gpus(n: int, min_free_mib: int = 10_000) -> list[int]:
    """여유 GPU 를 n 개 이상 찾아 반환한다."""
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
            f"여유 GPU {n}장 필요, {len(gpus)}장만 감지됨 (기준: {min_free_mib} MiB)")
    return gpus


# ── step helpers ────────────────────────────────────────────────

def count_goals(dataset: str) -> int:
    path = GOALS_DIR / f"{dataset}_goals.jsonl"
    if not path.exists():
        return 0
    return sum(1 for _ in open(path))


def step_generate(counts: dict[str, int]):
    """각 데이터셋별로 goals 부족분을 생성하고 SFT 변환."""
    print("\n=== [generate] goals 확인 및 생성 ===")
    need_rebuild = False
    for ds, n_train in counts.items():
        if n_train == 0:
            continue
        n_goals_needed = math.ceil(n_train / (1 - VAL_RATIO))
        n_goals_exist = count_goals(ds)
        deficit = n_goals_needed - n_goals_exist
        if deficit > 0:
            print(f"  {ds}: goals {n_goals_exist}개 → {n_goals_needed}개 필요, +{deficit}개 생성")
            cmd = [
                sys.executable, str(PIPELINE_DIR / "02_generate_goals.py"),
                ds, "--limit", str(deficit),
            ]
            subprocess.run(cmd, check=True)
            need_rebuild = True
        else:
            print(f"  {ds}: goals {n_goals_exist}개 ≥ {n_goals_needed}개 필요, 충분")

    if need_rebuild or not TRAIN_JSONL.exists():
        print("\n  04_sft_data_and_train.py convert 실행...")
        cmd = [sys.executable, str(PIPELINE_DIR / "04_sft_data_and_train.py"), "convert"]
        subprocess.run(cmd, check=True)
    else:
        print("\n  goals 변동 없음, SFT 변환 생략")


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

    # 이미 동일 설정으로 샘플링된 파일이 있으면 건너뛰기
    if train_path.exists() and config_path.exists():
        with open(config_path) as f:
            existing = json.load(f)
        if existing.get("counts") == counts and existing.get("seed") == seed:
            print(f"[{tag}] sample 이미 존재 ({existing['total']}개), 건너뛰기")
            return

    print(f"[{tag}] sample 시작")
    by_dataset = load_train_by_dataset(TRAIN_JSONL)

    rng = random.Random(seed)
    subset = []
    for ds, n in counts.items():
        if n == 0:
            continue
        pool = by_dataset.get(ds, [])
        if len(pool) < n:
            print(f"  [{tag}] WARNING: {ds} 요청 {n}개 > 가용 {len(pool)}개, 전부 사용")
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

    print(f"[{tag}] sample 완료: {len(subset)}개 → {train_path}")


# ── sample + train → infer_queue/eval_queue 에 태스크 제출 ─────

def run_experiment_on_gpu(tag: str, counts: dict[str, int], gpu_id: int,
                          gpu_pool: queue.Queue, infer_queue: queue.Queue,
                          eval_queue: queue.Queue):
    """sample → train 을 지정 GPU 에서 실행 후, infer 태스크를 큐에 제출."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    total = sum(counts.values())
    print(f"\n{'='*60}")
    print(f"[{tag}] GPU {gpu_id} | nq={counts['nq']}, hotpotqa={counts['hotpotqa']} (총 {total})")
    print(f"{'='*60}")

    # sample (CPU)
    step_sample(tag, counts, SEED)

    # train — checkpoint 이 이미 있으면 건너뛰기
    exp_dir = EXP_DIR / tag
    ckpt_dir = exp_dir / "checkpoints" / "final"
    if ckpt_dir.exists() and any(ckpt_dir.iterdir()):
        print(f"[{tag}] train 이미 완료 ({ckpt_dir}), 건너뛰기")
    else:
        print(f"[{tag}] train 시작 (GPU {gpu_id})")
        subprocess.run([
            sys.executable, str(PIPELINE_DIR / "04_sft_data_and_train.py"), "train",
            "--train-data", str(exp_dir / "train.jsonl"),
            "--output-dir", str(exp_dir / "checkpoints"),
        ], env=env, check=True)
        print(f"[{tag}] train 완료")

    # train 완료 → 이 실험의 GPU 를 풀에 반납
    gpu_pool.put(gpu_id)

    # infer 태스크를 큐에 제출
    for ds in EVAL_DATASETS:
        pred_path = exp_dir / f"predictions_{ds}.jsonl"
        if pred_path.exists():
            n_done = sum(1 for _ in open(pred_path))
            if n_done >= VAL_COUNTS[ds]:
                print(f"[{tag}] infer {ds} 완료 ({n_done}개), 건너뛰기")
                eval_queue.put((tag, ds))
                continue
            print(f"[{tag}] infer {ds} 불완전 ({n_done}/{VAL_COUNTS[ds]}개), 이어쓰기 예정")
        infer_queue.put((tag, ds, ckpt_dir, pred_path))


# ── eval helper ───────────────────────────────────────────────

def _do_one_eval(eval_mod, tag: str, ds: str, client, model_name: str,
                 gpu_id: int):
    """단일 (tag, ds) 평가. 이미 완료면 건너뛴다."""
    exp_dir = EXP_DIR / tag
    pred_path = exp_dir / f"predictions_{ds}.jsonl"
    judge_path = exp_dir / f"judge_{ds}.jsonl"

    if judge_path.exists():
        n_judged = sum(1 for _ in open(judge_path))
        if n_judged >= VAL_COUNTS[ds]:
            print(f"[GPU {gpu_id}] eval {tag}/{ds} 이미 완료 ({n_judged}개), 건너뛰기")
            return
    print(f"[GPU {gpu_id}] eval {tag}/{ds} 시작")
    eval_mod._evaluate_dataset(
        pred_path, judge_path, ds, client, model_name, eval_mod.NUM_WORKERS,
    )
    print(f"[GPU {gpu_id}] eval {tag}/{ds} 완료")


# ── 통합 worker (infer 우선 + eval 병행) ──────────────────────

def _start_workers(
    gpu_pool: queue.Queue,
    infer_queue: queue.Queue,
    eval_queue: queue.Queue,
    all_submitted: threading.Event,
    n_total_eval: int,
    n_workers: int,
) -> list[threading.Thread]:
    """infer/eval 통합 worker.

    infer_queue 에 미배정 태스크가 있으면 우선 처리하고,
    모두 배정되었으면 eval_queue 에서 eval 태스크를 처리한다.
    모든 infer 완료를 기다리지 않고 eval 을 병행한다.
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
            # ── 1) infer 우선 처리 ──────────────────────────────
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
                    print(f"[{tag}] infer {ds} 시작 (GPU {gpu_id})")
                    subprocess.run(cmd, env=env, check=True)
                    print(f"[{tag}] infer {ds} 완료 (GPU {gpu_id})")
                finally:
                    gpu_pool.put(gpu_id)
                eval_queue.put((tag, ds))
                continue

            # ── 2) eval 처리 (non-blocking, infer 우선권 유지) ──
            try:
                tag, ds = eval_queue.get_nowait()
            except queue.Empty:
                # 양쪽 다 비어 있으면 종료 확인 또는 대기
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
                    # eval 중에도 미배정 infer 가 있으면 양보
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

                    # 미배정 infer 가 있으면 vLLM 정리 후 양보
                    if not infer_queue.empty():
                        break

                    # 다음 eval 태스크 (non-blocking)
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
    """val 데이터를 precompute (이미 존재하면 건너뛰기)."""
    print("\n=== [precompute] val 데이터 확인 ===")
    from importlib import import_module
    val_loader = import_module("01_val_loader")
    for ds in EVAL_DATASETS:
        val_path = val_loader.VAL_DIR / f"{ds}_val.jsonl"
        if val_path.exists() and val_path.stat().st_size > 0:
            print(f"  {ds}: 이미 존재 ({val_path}), 건너뛰기")
        else:
            val_loader.precompute(ds)


def main():
    n_exp = len(EXPERIMENTS)

    # 0. GPU 확보
    gpu_ids = detect_free_gpus(n_exp)
    print(f"사용할 GPU: {gpu_ids} ({n_exp}개 대조군)")

    # 0.5. Val 데이터 precompute (05, 06 에서 사용)
    step_precompute_val()

    # 1. Generate: 모든 실험에 필요한 최대 개수만큼 한 번에 생성
    max_counts: dict[str, int] = {}
    for exp in EXPERIMENTS:
        for ds in TRAIN_DATASETS:
            max_counts[ds] = max(max_counts.get(ds, 0), exp[ds])
    print(f"데이터셋별 최대 필요량: {max_counts}")
    step_generate(max_counts)

    # 2. 큐 생성
    gpu_pool: queue.Queue[int] = queue.Queue()
    infer_queue: queue.Queue[tuple] = queue.Queue()
    eval_queue: queue.Queue[tuple[str, str]] = queue.Queue()
    all_submitted = threading.Event()

    # 3. 통합 worker 시작 (infer 우선 처리, eval 병행)
    n_total_eval = n_exp * len(EVAL_DATASETS)
    worker_threads = _start_workers(
        gpu_pool, infer_queue, eval_queue, all_submitted, n_total_eval, n_exp,
    )

    # 4. Sample + Train → infer_queue / eval_queue 에 태스크 제출
    print(f"\n{'='*60}")
    print(f"병렬 실행: {n_exp}개 대조군 × {n_exp}개 GPU")
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
                print(f"\n✓ [{tag}] sample+train 완료, infer/eval 제출됨")
            except Exception as e:
                print(f"\n✗ [{tag}] 실패: {e}")
                raise

    # 모든 train 완료 → 더 이상 새 태스크 없음
    all_submitted.set()
    print(f"\n=== 모든 태스크 제출 완료, worker 완료 대기 ===")

    # worker 완료 대기
    for t in worker_threads:
        t.join()

    print("\n=== Phase 1 완료 ===")


if __name__ == "__main__":
    main()
