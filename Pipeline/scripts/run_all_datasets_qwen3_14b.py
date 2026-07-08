"""Re-Guide all-datasets launcher preset for Qwen3-14B (non-thinking mode).

Thin wrapper around run_all_datasets.py: injects Qwen3 recommended sampling
defaults and context window suited to the 14B model on a single A6000
(max_model_len=32768, max_new_tokens=16384).

Thinking mode is OFF by default (enable_thinking=False in apply_chat_template).
Add --thinking_mode to switch on Qwen3 internal chain-of-thought; be aware
this inflates agent-generated token counts relative to the paper's baselines.

GPU scheduling: each GPU in --gpus runs ONE dataset at a time (TP=1), and
all GPUs are saturated in parallel. With `--gpus 6,7`, two datasets run
concurrently. Pass --no_auto_parallel to force TP across all listed GPUs.

Examples:
    # 4 datasets (nq, ambigqa, hotpotqa, musique), 2 at a time (1 GPU per dataset)
    python run_all_datasets_qwen3_14b.py --gpus 6,7 --retriever_gpus 0,1

    # single dataset on one GPU
    python run_all_datasets_qwen3_14b.py --gpus 3 --retriever_gpus 0,1 --dataset musique

    # force TP=2 across 6,7 (sequential, one dataset uses both GPUs)
    python run_all_datasets_qwen3_14b.py --gpus 6,7 --retriever_gpus 4,5 \\
        --dataset ambigqa --no_auto_parallel
"""
import os
import sys

PRESET = {
    "--model_path":     "/mnt/raid6/skbaek1223/models/Qwen3-14B",
    "--max_model_len":  "32768",
    "--max_new_tokens": "16384",
    "--temperature":    "0.6",
}

# 14B uses the same default budget as QwQ-32B / DeepSeek-Qwen-14B (60/90),
# so no per-dataset budget patching is needed here.


def _has_flag(flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])


def _flag_value(flag: str):
    argv = sys.argv[1:]
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    for a in argv:
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def _remove_flag(flag: str):
    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == flag:
            i += 2
            continue
        if a.startswith(flag + "="):
            i += 1
            continue
        new_argv.append(a)
        i += 1
    sys.argv[:] = new_argv


for _flag, _val in PRESET.items():
    if not _has_flag(_flag):
        sys.argv += [_flag, _val]

# 1 GPU per dataset in parallel by default.
_disable_auto_parallel = _has_flag("--no_auto_parallel")
if _disable_auto_parallel:
    _remove_flag("--no_auto_parallel")
if (not _disable_auto_parallel
        and _has_flag("--gpus")
        and not _has_flag("--parallel")
        and not _has_flag("--gpu_groups")):
    _gpu_list = [g.strip() for g in (_flag_value("--gpus") or "").split(",")
                 if g.strip()]
    if len(_gpu_list) >= 2:
        _remove_flag("--gpus")
        sys.argv += ["--parallel", "--gpu_groups", ";".join(_gpu_list)]


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_datasets  # noqa: E402

run_all_datasets.SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "run_re_guide_2.py"
)

_orig_build_cmd = run_all_datasets.build_cmd


def _build_cmd_with_thinking(args, dataset_name, data_file, split, retriever_url):
    cmd, qa_data_path, dataset_output_dir = _orig_build_cmd(
        args, dataset_name, data_file, split, retriever_url)
    cmd += ["--thinking_mode"]
    return cmd, qa_data_path, dataset_output_dir


run_all_datasets.build_cmd = _build_cmd_with_thinking

_KEEP = {"nq", "ambigqa", "hotpotqa", "musique"}
run_all_datasets.DATASETS = [d for d in run_all_datasets.DATASETS if d[0] in _KEEP]

if __name__ == "__main__":
    run_all_datasets.main()
