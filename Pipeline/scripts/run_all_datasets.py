"""
Run Re-Guide experiments on all 6 datasets.

Architecture:
    - FAISS retriever runs in a dedicated HTTP server process
      (retriever_server.py) on its own GPUs. The launcher auto-spawns it if
      no server is already reachable at the configured URL; if an external
      server is already running, it is reused and NOT torn down on exit.
    - Each worker loads vLLM ONCE on its assigned GPUs and never swaps.

Sequential mode (single worker, default; QwQ-32B unless --preset is given):
    python run_all_datasets.py --gpus 2,3 --retriever_gpus 0,1
    python run_all_datasets.py --gpus 2,3 --retriever_gpus 0,1 --dataset hotpotqa,musique,2wiki,nq,triviaqa,ambigqa

Parallel mode (multiple independent workers, one GPU group each):
    # 2 GPUs for retriever, 6 GPUs split into three vLLM workers
    python run_all_datasets.py --parallel \\
        --retriever_gpus 0,1 --gpu_groups "2,3;4,5;6,7"

    # tail two launch logs at once
    tail -f /mnt/raid6/skbaek1223/project/Re-Guide/Pipeline/scripts/outputs/final_results/triviaqa.qwq-32b.re_guide.launch.log \\
            /mnt/raid6/skbaek1223/project/Re-Guide/Pipeline/scripts/outputs/final_results/hotpotqa.qwq-32b.re_guide.launch.log

Per-model presets (--preset <name>): injects the model path, context window,
and sampling defaults for a specific model, then applies one-GPU-per-dataset
auto-parallelization when --gpus is given (pass --no_auto_parallel to force
tensor parallelism across all listed GPUs on a single dataset instead). Any
flag you pass explicitly still takes precedence over the preset's defaults.

    qwen3_4b     Qwen3-4B.  max_model_len=32768, max_new_tokens=16384,
                 temperature=0.6, retry-word budget 30 (single-hop) / 45
                 (multi-hop), thinking_mode forced on. Datasets: nq, ambigqa,
                 hotpotqa, musique.
    qwen3_8b     Qwen3-8B.  Same shape as qwen3_4b. The fine-tuned planner
                 checkpoint (planner-5050-5000/final_merged) is also
                 Qwen3-8B based and shared across all Qwen3 runs.
    qwen3_14b    Qwen3-14B. Same shape as qwen3_4b but uses the default
                 60/90 retry-word budget (no override).
    r1_llama8b   DeepSeek-R1-Distill-Llama-8B. Retry-word budget 30/45.
                 Runs all 8 dataset entries except the triviaqa_a/_b split
                 (plain triviaqa only).
    r1_qwen14b   DeepSeek-R1-Distill-Qwen-14B. Default 60/90 budget. Same
                 dataset selection as r1_llama8b.

    python run_all_datasets.py --preset qwen3_8b --gpus 6,7 --retriever_gpus 0,1
    python run_all_datasets.py --preset r1_llama8b --gpus 6 --retriever_gpus 0,1 --dataset musique
"""

import argparse
import os
import subprocess
import sys
import time

os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SCRIPT_PATH = os.path.join(PROJECT_ROOT, "run_re_guide.py")
RETRIEVER_SERVER_PATH = os.path.join(PROJECT_ROOT, "retriever_server.py")

sys.path.insert(0, PROJECT_ROOT)
from retriever_utils import ensure_retriever_server, stop_retriever_server
DATA_DIR = os.path.join(PROJECT_ROOT, "..", "..", "data", "eval")
FINAL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "final_results")
DEFAULT_STEPS_MODEL = os.path.join(
    PROJECT_ROOT, "..", "..", "data", "experiments", "S7000", "checkpoints", "final_merged"
)

# ── Dataset definitions ────────────────────────────────────────────────
# (dataset_name, qa_data_filename, split)
DATASETS = [
    ("nq",          "nq_test.json",         "test"),
    ("triviaqa",    "triviaqa_test.json",   "test"),
    ("triviaqa_a",  "triviaqa_test.json",   "test"),
    ("triviaqa_b",  "triviaqa_test.json",   "test"),
    ("ambigqa",     "ambigqa_dev.json",     "dev"),
    ("hotpotqa",    "hotpotqa_dev.json",    "dev"),
    ("2wiki",       "2wiki_dev.json",       "dev"),
    ("musique",     "musique_dev.json",     "dev"),
]

MULTI_HOP_DATASETS = {"hotpotqa", "2wiki", "musique"}
QWEN3_DATASETS = {"nq", "ambigqa", "hotpotqa", "musique"}
TRIVIAQA_SPLIT = {"triviaqa_a", "triviaqa_b"}

# ── Per-model presets ────────────────────────────────────────────────────
PRESETS = {
    "qwen3_4b": dict(
        model_path="/mnt/raid6/skbaek1223/models/Qwen3-4B",
        max_model_len=32768, max_new_tokens=16384, temperature=0.6,
        budget_singlehop=30, budget_multihop=45,
        thinking_mode=True,
        keep_datasets=QWEN3_DATASETS,
    ),
    "qwen3_8b": dict(
        model_path="/mnt/raid6/skbaek1223/models/Qwen3-8B",
        max_model_len=32768, max_new_tokens=16384, temperature=0.6,
        budget_singlehop=30, budget_multihop=45,
        thinking_mode=True,
        keep_datasets=QWEN3_DATASETS,
    ),
    "qwen3_14b": dict(
        model_path="/mnt/raid6/skbaek1223/models/Qwen3-14B",
        max_model_len=32768, max_new_tokens=16384, temperature=0.6,
        thinking_mode=True,
        keep_datasets=QWEN3_DATASETS,
    ),
    "r1_llama8b": dict(
        model_path="/mnt/raid6/skbaek1223/models/DeepSeek-R1-Distill-Llama-8B",
        max_model_len=32768, max_new_tokens=16384, temperature=0.6,
        budget_singlehop=30, budget_multihop=45,
        skip_datasets=TRIVIAQA_SPLIT,
    ),
    "r1_qwen14b": dict(
        model_path="/mnt/raid6/skbaek1223/models/DeepSeek-R1-Distill-Qwen-14B",
        max_model_len=32768, max_new_tokens=16384, temperature=0.6,
        skip_datasets=TRIVIAQA_SPLIT,
    ),
}

# Set by main() from --preset; read by build_cmd() to inject --budget_base /
# --thinking_mode into the per-dataset worker command.
_ACTIVE_PRESET = {}


# ── Worker command construction ────────────────────────────────────────
def build_cmd(args, dataset_name, data_file, split, retriever_url):
    qa_data_path = os.path.normpath(os.path.join(DATA_DIR, data_file))
    model_short = args.model_path.split('/')[-1].lower().replace('-instruct', '')
    dataset_output_dir = os.path.normpath(
        os.path.join(FINAL_OUTPUT_DIR, f"{dataset_name}.{model_short}.re_guide")
    )
    cmd = [
        sys.executable, SCRIPT_PATH,
        "--dataset_name", dataset_name,
        "--split", split,
        "--qa_data_path", qa_data_path,
        "--model_path", args.model_path,
        "--output_dir", dataset_output_dir,
        "--max_turn", str(args.max_turn),
        "--top_k", str(args.top_k),
        "--max_search_limit", str(args.max_search_limit),
        "--max_model_len", str(args.max_model_len),
        "--max_new_tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--search_cache_suffix", dataset_name,
        "--retriever_url", retriever_url,
        "--retrieval_method", args.retrieval_method,
    ]
    if args.steps_model_path:
        cmd += ["--steps_model_path", args.steps_model_path]
    if args.subset_num != -1:
        cmd += ["--subset_num", str(args.subset_num)]
    if "budget_singlehop" in _ACTIVE_PRESET:
        budget = (_ACTIVE_PRESET["budget_multihop"] if dataset_name in MULTI_HOP_DATASETS
                  else _ACTIVE_PRESET["budget_singlehop"])
        cmd += ["--budget_base", str(budget)]
    if _ACTIVE_PRESET.get("thinking_mode"):
        cmd += ["--thinking_mode"]
    return cmd, qa_data_path, dataset_output_dir


def run_sequential(args, run_datasets, retriever_url):
    env_gpus = args.gpus.strip()
    results = []
    for dataset_name, data_file, split in run_datasets:
        cmd, qa_data_path, dataset_output_dir = build_cmd(
            args, dataset_name, data_file, split, retriever_url)
        if not os.path.exists(qa_data_path):
            print(f"[ERROR] Data file not found: {qa_data_path}, skipping {dataset_name}")
            results.append((dataset_name, "SKIPPED (file not found)"))
            continue
        print(f"\n{'='*60}\n  RUNNING: {dataset_name} (split={split})\n"
              f"  vLLM GPUs: {env_gpus}\n"
              f"  Retriever: {retriever_url}\n"
              f"  Data:      {qa_data_path}\n  Output:    {dataset_output_dir}\n{'='*60}\n")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = env_gpus
        t0 = time.time()
        ret = subprocess.run(cmd, env=env)
        elapsed = time.time() - t0
        status = "OK" if ret.returncode == 0 else f"FAIL (exit {ret.returncode})"
        results.append((dataset_name, status, f"{elapsed:.0f}s"))
        print(f"\n>> {dataset_name}: {status}  ({elapsed:.0f}s)\n")
    return results


def run_parallel(args, run_datasets, retriever_url):
    gpu_groups = [g.strip() for g in args.gpu_groups.split(";") if g.strip()]
    if not gpu_groups:
        raise SystemExit("--gpu_groups is empty")

    pending = []
    results = []
    for dataset_name, data_file, split in run_datasets:
        cmd, qa_data_path, dataset_output_dir = build_cmd(
            args, dataset_name, data_file, split, retriever_url)
        if not os.path.exists(qa_data_path):
            print(f"[ERROR] Data file not found: {qa_data_path}, skipping {dataset_name}")
            results.append((dataset_name, "SKIPPED (file not found)"))
            continue
        pending.append((dataset_name, cmd, dataset_output_dir))

    active_procs = []  # workers of the current batch only

    def _kill_active_workers():
        for name, p, log_f, _t0 in active_procs:
            if p.poll() is None:
                print(f"[launcher] terminating worker {name} (pid={p.pid})")
                try:
                    p.terminate()
                except Exception:
                    pass
        for name, p, log_f, _t0 in active_procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                except Exception:
                    pass
            try:
                log_f.close()
            except Exception:
                pass

    try:
        idx = 0
        while idx < len(pending):
            batch = pending[idx: idx + len(gpu_groups)]
            active_procs = []
            for (dataset_name, cmd, out_dir), group in zip(batch, gpu_groups):
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = group
                # Avoid distributed init-port collisions across parallel vLLM processes.
                env["VLLM_HOST_IP"] = "127.0.0.1"
                log_path = out_dir + ".launch.log"
                os.makedirs(out_dir, exist_ok=True)
                log_f = open(log_path, "w", buffering=1)
                print(f"[launcher] {dataset_name} -> GPUs {group}  (log: {log_path})")
                # start_new_session=True so Ctrl+C at the terminal goes to the launcher
                # only; we forward termination explicitly below.
                p = subprocess.Popen(cmd, env=env, stdout=log_f,
                                     stderr=subprocess.STDOUT, start_new_session=True)
                active_procs.append((dataset_name, p, log_f, time.time()))
            for dataset_name, p, log_f, t0 in active_procs:
                ret = p.wait()
                log_f.close()
                elapsed = time.time() - t0
                status = "OK" if ret == 0 else f"FAIL (exit {ret})"
                results.append((dataset_name, status, f"{elapsed:.0f}s"))
                print(f"[launcher] >> {dataset_name}: {status}  ({elapsed:.0f}s)")
            active_procs = []
            idx += len(gpu_groups)
    except KeyboardInterrupt:
        print("[launcher] KeyboardInterrupt — terminating workers")
        _kill_active_workers()
        raise
    finally:
        _kill_active_workers()
    return results


def main():
    global _ACTIVE_PRESET
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default=None, choices=sorted(PRESETS),
                        help="Per-model preset; see module docstring for details.")
    parser.add_argument("--no_auto_parallel", action="store_true",
                        help="With --preset, disable one-GPU-per-dataset auto-parallelization.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to the main reasoning LLM. "
                             "Default: the preset's model, or QwQ-32B.")
    parser.add_argument("--steps_model_path", type=str, default=DEFAULT_STEPS_MODEL,
                        help="Path to the planner model.")
    parser.add_argument("--subset_num", type=int, default=-1,
                        help="Limit samples per dataset (-1 = all).")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run specific dataset(s). Comma-separated list allowed "
                             "(e.g. 'nq,triviaqa,hotpotqa'). If omitted, run all "
                             "(or the preset's subset).")
    parser.add_argument("--max_turn", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--max_search_limit", type=int, default=20)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)

    # Sequential mode: GPUs for the single vLLM worker.
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU ids for vLLM (sequential mode). "
                             "Must NOT overlap --retriever_gpus. Default: 2,3.")

    # Parallel mode: each group is an independent vLLM worker.
    parser.add_argument("--parallel", action="store_true",
                        help="Run datasets concurrently; each GPU group is an independent "
                             "vLLM worker.")
    parser.add_argument("--gpu_groups", type=str, default=None,
                        help="Semicolon-separated vLLM GPU groups. "
                             "Example: '2,3;4,5;6,7'. Must NOT overlap --retriever_gpus.")

    # Retriever server.
    parser.add_argument("--retriever_gpus", type=str, default="0,1",
                        help="GPUs dedicated to the FAISS retriever server.")
    parser.add_argument("--retriever_host", type=str, default="127.0.0.1")
    parser.add_argument("--retriever_port", type=int, default=8765)
    parser.add_argument("--retrieval_method", type=str, default="e5",
                        choices=["e5", "bm25"])
    parser.add_argument("--retriever_startup_timeout", type=int, default=900,
                        help="Seconds to wait for retriever /health after spawn.")
    args = parser.parse_args()

    cfg = PRESETS.get(args.preset, {})
    _ACTIVE_PRESET = cfg

    if args.model_path is None:
        args.model_path = cfg.get("model_path", "/mnt/raid6/skbaek1223/models/QwQ-32B")
    if args.max_model_len is None:
        args.max_model_len = cfg.get("max_model_len", 40960)
    if args.max_new_tokens is None:
        args.max_new_tokens = cfg.get("max_new_tokens", 16384)
    if args.temperature is None:
        args.temperature = cfg.get("temperature", 0.7)

    # One GPU per dataset in parallel by default when a preset is active.
    if (args.preset and not args.no_auto_parallel and not args.parallel
            and args.gpu_groups is None and args.gpus):
        gpu_list = [g.strip() for g in args.gpus.split(",") if g.strip()]
        if len(gpu_list) >= 2:
            args.parallel = True
            args.gpu_groups = ";".join(gpu_list)

    if args.gpus is None:
        args.gpus = "2,3"
    if args.gpu_groups is None:
        args.gpu_groups = "2,3;4,5;6,7"

    if "keep_datasets" in cfg:
        dataset_pool = [d for d in DATASETS if d[0] in cfg["keep_datasets"]]
    elif "skip_datasets" in cfg:
        dataset_pool = [d for d in DATASETS if d[0] not in cfg["skip_datasets"]]
    else:
        dataset_pool = DATASETS

    if args.dataset:
        requested = [d.strip() for d in args.dataset.split(",") if d.strip()]
        valid = {n for n, _, _ in dataset_pool}
        unknown = [d for d in requested if d not in valid]
        if unknown:
            raise SystemExit(f"Unknown dataset(s): {unknown}. Valid: {sorted(valid)}")
        by_name = {n: (n, f, s) for n, f, s in dataset_pool}
        run_datasets = [by_name[d] for d in requested]
    else:
        run_datasets = dataset_pool

    # GPU overlap sanity check.
    retriever_set = {g.strip() for g in args.retriever_gpus.split(",") if g.strip()}
    if args.parallel:
        worker_sets = [{g.strip() for g in group.split(",") if g.strip()}
                       for group in args.gpu_groups.split(";") if group.strip()]
    else:
        worker_sets = [{g.strip() for g in args.gpus.split(",") if g.strip()}]
    for ws in worker_sets:
        overlap = ws & retriever_set
        if overlap:
            raise SystemExit(
                f"--retriever_gpus and vLLM GPUs overlap on {sorted(overlap)}. "
                f"Give the retriever its own GPUs.")

    retriever_url, retriever_proc = ensure_retriever_server(
        retriever_gpus=args.retriever_gpus,
        host=args.retriever_host,
        port=args.retriever_port,
        retrieval_method=args.retrieval_method,
        top_k=args.top_k,
        startup_timeout=args.retriever_startup_timeout,
        log_path=os.path.join(FINAL_OUTPUT_DIR, "retriever_server.log"),
        server_script=RETRIEVER_SERVER_PATH,
    )

    try:
        if args.parallel:
            results = run_parallel(args, run_datasets, retriever_url)
        else:
            results = run_sequential(args, run_datasets, retriever_url)
    finally:
        stop_retriever_server(retriever_proc)

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for entry in results:
        name = entry[0]
        status = entry[1]
        elapsed = entry[2] if len(entry) > 2 else ""
        print(f"  {name:12s}  {status:30s}  {elapsed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
