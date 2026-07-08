"""
Phase 2: 크기 탐색

Phase 1 최적 비율을 아래 RATIO 에 설정한 뒤 실행합니다.

Usage:
    python experiment_phase2.py              # 1단계부터 전체 실행
    HF_HUB_OFFLINE=1 python experiment_phase2.py --start 4
    # 3단계(sample+train)부터 시작
    python experiment_phase2.py --train-only  # train만 실행, 모델 저장 후 종료

단계:
    1) generate        — 최대 크기에 맞게 goals 를 한 번에 생성
    2) convert+sample  — goals → SFT 변환 후, 크기별 train 데이터 샘플링
    3) train           — 크기별 GPU 1개씩 할당, 병렬 학습
    4) infer+eval      — 통합 worker 가 infer 우선 처리, 배정 완료 시 eval 병행
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

# Phase 1 최적 비율 (합=1000 기준) — Phase 1 결과에 따라 수정
RATIO = {"nq": 300, "hotpotqa": 700}

SIZES = [9000, 10000]

EVAL_DATASETS = ["nq", "hotpotqa"]
SEED = 42
ALLOWED_GPUS = [1, 3, 5, 7]
N_TRAIN_GPUS = 3
N_INFER_GPUS = 4
N_INFER_SPLITS = 2  # 각 (tag, ds) infer 를 몇 개 GPU 로 분할
N_EVAL_SPLITS = 2   # 각 (tag, ds) eval 을 몇 개 GPU 로 분할


# ── GPU 감지 ────────────────────────────────────────────────────

def detect_free_gpus(n: int, min_free_mib: int = 10_000,
                     strict: bool = False) -> list[int]:
    """여유 GPU 를 최대 n 개 찾아 반환한다.

    ALLOWED_GPUS 에 포함된 GPU 만 후보로 사용한다.
    strict=True 이면 n 개 미만일 때 RuntimeError,
    strict=False 이면 가용한 만큼만 반환 (최소 1개 필요).
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
                f"여유 GPU {n}장 필요, {len(gpus)}장만 감지됨 (기준: {min_free_mib} MiB)")
        if len(gpus) == 0:
            raise RuntimeError(
                f"여유 GPU 0장 — 최소 1장 필요 (기준: {min_free_mib} MiB)")
        print(f"⚠ GPU {n}장 요청했으나 {len(gpus)}장만 가용, {len(gpus)}장으로 진행")
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
    """goals 확인 및 생성. 새로 생성했으면 True 반환."""
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
                ds, "--limit", str(deficit), "--skip-judge",
            ]
            subprocess.run(cmd, check=True)
            need_rebuild = True
        else:
            print(f"  {ds}: goals {n_goals_exist}개 ≥ {n_goals_needed}개 필요, 충분")
    return need_rebuild


def step_convert():
    """raw goals → SFT train/val JSONL 변환."""
    print("\n=== [convert] raw goals → SFT 변환 ===")
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
        raise FileNotFoundError(f"goals 파일 없음: {GOALS_DIR}")

    all_items: list[dict] = []
    for path in goal_files:
        items = [json.loads(line) for line in open(path)]
        dataset_name = path.stem.replace("_goals", "")
        print(f"  {dataset_name}: {len(items)}개 로드 (raw)")
        all_items.extend(items)

    rng.shuffle(all_items)
    print(f"  전체: {len(all_items)}개")

    val_count = max(1, int(len(all_items) * VAL_RATIO))
    val = all_items[:val_count]
    train = all_items[val_count:]

    val_path = SFT_DIR / "val.jsonl"
    with open(val_path, "w") as f:
        for item in val:
            f.write(json.dumps(sft_mod.item_to_sft(item), ensure_ascii=False) + "\n")
    print(f"  val 저장: {len(val)}개 → {val_path}")

    train_path = SFT_DIR / "train.jsonl"
    with open(train_path, "w") as f:
        for item in train:
            f.write(json.dumps(sft_mod.item_to_sft(item), ensure_ascii=False) + "\n")
    print(f"  train 저장: {len(train)}개 → {train_path}")


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
        print(f"  [{tag}] {ds}: {n}개 샘플링")
    rng.shuffle(subset)

    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(train_path, "w") as f:
        for item in subset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    config = {"tag": tag, "counts": counts, "seed": seed, "total": len(subset)}
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[{tag}] sample 완료: {len(subset)}개 → {train_path}")


# ── infer 분할/병합 ──────────────────────────────────────────

def _prepare_split_infer(tag: str, ds: str, model_dir: Path,
                         n_splits: int = N_INFER_SPLITS,
                         ) -> list[tuple]:
    """남은 추론 대상을 n_splits 개로 분할, ID 파일 생성.

    반환: [(tag, ds, model_dir, part_pred, id_file, part_idx), ...]
    이미 완료된 경우 빈 리스트 반환.
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

    print(f"[{tag}] infer {ds}: 남은 {len(remaining)}개 → {len(tasks)}개 분할")
    return tasks


def _merge_part_predictions(tag: str, ds: str):
    """part 파일들을 메인 predictions 파일에 append 후 정리."""
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
    print(f"[{tag}] infer {ds}: {len(parts)}개 part 병합 완료")


def _enqueue_infer_splits(tag: str, ds: str, model_dir: Path,
                          infer_queue: queue.Queue,
                          eval_queue: queue.Queue,
                          infer_parts_remaining: dict,
                          infer_parts_lock: threading.Lock,
                          eval_parts_remaining: dict,
                          eval_parts_lock: threading.Lock):
    """infer 분할 태스크를 큐에 넣거나, 이미 완료면 eval 큐에 넣는다."""
    sub_tasks = _prepare_split_infer(tag, ds, model_dir)
    if not sub_tasks:
        print(f"[{tag}] infer {ds} 이미 완료, 건너뛰기")
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
    """남은 평가 대상을 n_splits 개로 분할, part prediction 파일 생성.

    반환: [(tag, ds, part_pred, part_judge, part_idx), ...]
    이미 완료된 경우 빈 리스트 반환.
    """
    exp_dir = EXP_DIR / tag
    pred_path = exp_dir / f"predictions_{ds}.jsonl"
    judge_path = exp_dir / f"judge_{ds}.jsonl"

    # 전체 prediction ID 수집
    all_ids = []
    with open(pred_path) as f:
        for line in f:
            try:
                all_ids.append(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass

    # 이미 judge 완료된 ID 제외
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

    # remaining ID 에 해당하는 prediction 행을 읽어둠
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

    print(f"[{tag}] eval {ds}: 남은 {len(remaining)}개 → {len(tasks)}개 분할")
    return tasks


def _merge_part_judges(tag: str, ds: str):
    """part judge 파일들을 메인 judge 파일에 append 후 정리."""
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
    # part prediction 파일도 정리
    for pp in exp_dir.glob(f"predictions_{ds}_eval_part*.jsonl"):
        pp.unlink()
    print(f"[{tag}] eval {ds}: {len(parts)}개 part 병합 완료")


def _enqueue_eval_splits(tag: str, ds: str,
                         eval_queue: queue.Queue,
                         eval_parts_remaining: dict,
                         eval_parts_lock: threading.Lock):
    """eval 분할 태스크를 큐에 넣거나, 이미 완료면 완료 처리."""
    sub_tasks = _prepare_split_eval(tag, ds)
    if not sub_tasks:
        print(f"[{tag}] eval {ds} 이미 완료, 건너뛰기")
        eval_queue.put(("__done__", tag, ds))
        return
    with eval_parts_lock:
        eval_parts_remaining[(tag, ds)] = len(sub_tasks)
    for st in sub_tasks:
        eval_queue.put(st)


# ── train ─────────────────────────────────────────────────────

def run_train_on_gpu(tag: str, counts: dict[str, int], gpu_id: int):
    """train 을 지정 GPU 에서 실행."""
    exp_dir = EXP_DIR / tag
    final_dir = exp_dir / "checkpoints" / "final"
    if final_dir.exists():
        print(f"[{tag}] train 이미 완료 ({final_dir}), 건너뛰기")
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    total = sum(counts.values())
    print(f"\n{'='*60}")
    print(f"[{tag}] GPU {gpu_id} | {counts} (총 {total})")
    print(f"{'='*60}")

    print(f"[{tag}] train 시작 (GPU {gpu_id})")
    subprocess.run([
        sys.executable, str(PIPELINE_DIR / "04_sft_data_and_train.py"), "train",
        "--train-data", str(exp_dir / "train.jsonl"),
        "--output-dir", str(exp_dir / "checkpoints"),
    ], env=env, check=True)
    print(f"[{tag}] train 완료")

    # train 완료 후 기존 infer/eval 결과 삭제 (재실행 강제)
    for ds in EVAL_DATASETS:
        for pattern in [f"predictions_{ds}.jsonl", f"judge_{ds}.jsonl"]:
            old = exp_dir / pattern
            if old.exists():
                old.unlink()
                print(f"[{tag}] 기존 {pattern} 삭제 (재실행 예정)")


# ── eval helper ───────────────────────────────────────────────

def _do_one_eval(eval_mod, tag: str, ds: str, client, model_name: str,
                 gpu_id: int, pred_path: Path = None, judge_path: Path = None,
                 part_idx: int = None):
    """단일 eval 태스크 (전체 또는 part) 평가."""
    exp_dir = EXP_DIR / tag
    if pred_path is None:
        pred_path = exp_dir / f"predictions_{ds}.jsonl"
    if judge_path is None:
        judge_path = exp_dir / f"judge_{ds}.jsonl"

    label = f"{tag}/{ds}" if part_idx is None else f"{tag}/{ds} part{part_idx}"

    # 이미 완료된 eval 건너뛰기
    if judge_path.exists() and _count_lines(judge_path) >= _count_lines(pred_path):
        print(f"[GPU {gpu_id}] eval {label} 이미 완료, 건너뛰기")
        return
    if judge_path.exists():
        judge_path.unlink()
    print(f"[GPU {gpu_id}] eval {label} 시작")
    eval_mod._evaluate_dataset(
        pred_path, judge_path, ds, client, model_name, eval_mod.NUM_WORKERS,
    )
    print(f"[GPU {gpu_id}] eval {label} 완료")


# ── 통합 worker (infer 우선 + eval 병행) ──────────────────────

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
    """infer/eval 통합 worker.

    infer_queue 에 분할된 infer 서브태스크가 있으면 우선 처리하고,
    모두 배정되었으면 eval_queue 에서 eval 분할 태스크를 처리한다.
    모든 infer 완료를 기다리지 않고 eval 을 병행한다.
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
            # ── 1) infer 우선 처리 (분할 서브태스크) ────────────
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
                    print(f"[{tag}] infer {ds} part{part_idx} 시작 (GPU {gpu_id})")
                    subprocess.run(cmd, env=env, check=True)
                    print(f"[{tag}] infer {ds} part{part_idx} 완료 (GPU {gpu_id})")
                finally:
                    gpu_pool.put(gpu_id)
                # 모든 part 완료 시 병합 → eval 분할 태스크 제출
                with infer_parts_lock:
                    infer_parts_remaining[(tag, ds)] -= 1
                    if infer_parts_remaining[(tag, ds)] == 0:
                        _merge_part_predictions(tag, ds)
                        _enqueue_eval_splits(tag, ds, eval_queue,
                                             eval_parts_remaining,
                                             eval_parts_lock)
                continue

            # ── 2) eval 처리 (분할 서브태스크, infer 우선권 유지) ──
            eval_task = None
            try:
                eval_task = eval_queue.get_nowait()
            except queue.Empty:
                # 양쪽 다 비어 있으면 종료 확인 또는 대기
                if all_submitted.is_set() and infer_queue.empty():
                    with counter_lock:
                        if eval_completed[0] >= n_total_eval:
                            all_eval_done.set()
                            return
                time.sleep(1)
                continue

            # 이미 완료된 (tag, ds) 는 카운트만 올리고 넘어감
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
                    # eval 중에도 미배정 infer 가 있으면 양보
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

                    # part 완료 → 모든 part 완료 시 병합
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

                    # 미배정 infer 가 있으면 vLLM 정리 후 양보
                    if not infer_queue.empty():
                        break

                    # 다음 eval 태스크 (non-blocking)
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
    """val 데이터를 precompute (이미 존재하면 건너뛰기)."""
    print("\n=== [precompute] val 데이터 확인 ===")
    val_loader = import_module("01_val_loader")
    for ds in EVAL_DATASETS:
        val_path = val_loader.VAL_DIR / f"{ds}_val.jsonl"
        if val_path.exists() and val_path.stat().st_size > 0:
            print(f"  {ds}: 이미 존재 ({val_path}), 건너뛰기")
        else:
            val_loader.precompute(ds)


def main(start: int = 1, train_only: bool = False):
    n_exp = len(SIZES)
    ratio_total = sum(RATIO.values())

    # 0. Val 데이터 precompute
    step_precompute_val()

    # 1. Generate: 최대 크기에 맞게 goals 생성
    if start <= 1:
        max_size = max(SIZES)
        max_counts = {ds: n * max_size // ratio_total for ds, n in RATIO.items()}
        print(f"=== 최대 크기({max_size})에 맞게 goals 생성 ===")
        step_generate(max_counts)

    # 2. Convert + Sample: goals → SFT 변환 후 크기별 샘플링
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
        print(f"병렬 학습 (train only): {n_exp}개 크기 실험 (GPU: {train_gpus})")
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
                    print(f"\n✓ [{tag}] train 완료")
                except Exception as e:
                    print(f"\n✗ [{tag}] 실패: {e}")
                    raise

        print("\n=== Train only 완료 ===")
        return

    # 3+4. Train → Infer+Eval 파이프라인 (train 완료되는 대로 즉시 infer 시작)
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
        print(f"병렬 학습 + 즉시 Infer: {n_exp}개 크기 실험 (GPU: {train_gpus})")
        print(f"{'='*60}")

        # infer+eval worker 를 먼저 기동 (GPU pool 은 비어 있으므로 대기)
        worker_threads = _start_workers(
            gpu_pool, infer_queue, eval_queue, all_submitted,
            n_total_eval, N_INFER_GPUS,
            infer_parts_remaining, infer_parts_lock,
            eval_parts_remaining, eval_parts_lock,
        )

        def train_then_enqueue(tag, counts, gpu_id):
            """train 완료 후 해당 GPU 를 infer pool 에 넘기고 infer 태스크 제출."""
            run_train_on_gpu(tag, counts, gpu_id)
            # train 끝난 GPU 를 infer pool 에 투입
            gpu_pool.put(gpu_id)
            print(f"[{tag}] GPU {gpu_id} → infer pool 투입")
            # 해당 tag 의 infer 분할 태스크 제출
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
                    print(f"\n✓ [{tag}] train+infer제출 완료")
                except Exception as e:
                    print(f"\n✗ [{tag}] 실패: {e}")
                    raise

        # 모든 train 완료 → 더 이상 infer 태스크 추가 없음
        all_submitted.set()

    else:
        # start == 4: train 건너뛰고 infer+eval 만 실행
        infer_gpus = detect_free_gpus(N_INFER_GPUS, strict=True)
        print(f"\n{'='*60}")
        print(f"Infer + Eval 시작 (GPU: {infer_gpus})")
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

    print("\n=== Phase 2 완료 ===")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase 2: 크기 탐색")
    parser.add_argument("--start", type=int, default=1, choices=[1, 2, 3, 4],
                        help="시작 단계 (1=generate, 2=convert+sample, 3=train, 4=infer+eval)")
    parser.add_argument("--train-only", action="store_true",
                        help="train 단계만 실행하고 모델 저장 후 종료 (infer+eval 생략)")
    args = parser.parse_args()
    main(start=args.start, train_only=args.train_only)
