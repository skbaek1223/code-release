"""Re-Guide all-datasets launcher preset for DeepSeek-R1-Distill-Llama-8B.

Thin wrapper around run_all_datasets.py: injects DeepSeek-R1 recommended
sampling defaults (temperature=0.6) and context window suited to the 8B
model on a single A6000 (max_model_len=32768, max_new_tokens=16384). The
model has 131k native context via llama3 RoPE scaling, so larger windows
are also valid if KV cache budget allows.

GPU scheduling: each GPU in --gpus runs ONE dataset at a time (TP=1), and
all GPUs are saturated in parallel. With `--gpus 6,7`, two datasets run
concurrently; as each finishes, the next pending dataset is dispatched to
the freed GPU. To override and use tensor parallelism across multiple GPUs
on a single dataset, pass `--no_auto_parallel`.

Any flag the user passes explicitly takes precedence.

Examples:
    # 8 datasets, 2 at a time (1 GPU per dataset)
    python run_all_datasets_r1_llama8b.py --gpus 6 --retriever_gpus 0,1 --dataset hotpotqa

    # single dataset on one GPU
    python run_all_datasets_r1_llama8b.py --gpus 6 --retriever_gpus 0,1 --dataset nq,triviaqa,ambigqa

    # force TP=2 across 6,7 (sequential, one dataset uses both GPUs)
    python run_all_datasets_r1_llama8b.py --gpus 6,7 --retriever_gpus 4,5 \\
        --dataset ambigqa --no_auto_parallel
"""
import os
import sys

PRESET = {
    "--model_path":     "/mnt/raid6/skbaek1223/models/DeepSeek-R1-Distill-Llama-8B",
    "--max_model_len":  "32768",
    "--max_new_tokens": "16384",
    "--temperature":    "0.6",
}

# Llama-8B uses a halved retry-word budget: 30 (single-hop) / 45 (multi-hop)
# vs. the QwQ-32B defaults of 60 / 90 in run_re_guide.py. These are not
# args of run_all_datasets.py — they are injected per-dataset into the
# run_re_guide_2.py command as --budget_base by patching build_cmd below.
BUDGET_SINGLEHOP = "30"
BUDGET_MULTIHOP = "45"
MULTI_HOP_DATASETS = {"hotpotqa", "2wiki", "musique"}


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


# 1 GPU per dataset: split --gpus 6,7 into parallel groups "6;7" so each
# GPU runs an independent vLLM worker (TP=1) and the launcher pulls
# pending datasets onto whichever GPU frees up next.
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

# Use run_re_guide_2.py instead of run_re_guide.py for this preset.
run_all_datasets.SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "run_re_guide_2.py"
)

# Wrap build_cmd to append --budget_base per-dataset (single-hop vs multi-hop).
_orig_build_cmd = run_all_datasets.build_cmd

def _build_cmd_with_budget(args, dataset_name, data_file, split, retriever_url):
    cmd, qa_data_path, dataset_output_dir = _orig_build_cmd(
        args, dataset_name, data_file, split, retriever_url)
    budget = BUDGET_MULTIHOP if dataset_name in MULTI_HOP_DATASETS else BUDGET_SINGLEHOP
    cmd += ["--budget_base", budget]
    return cmd, qa_data_path, dataset_output_dir

run_all_datasets.build_cmd = _build_cmd_with_budget

# Llama-8B preset: skip triviaqa_a/_b variants (run plain triviaqa only).
_SKIP = {"triviaqa_a", "triviaqa_b"}
run_all_datasets.DATASETS = [d for d in run_all_datasets.DATASETS if d[0] not in _SKIP]

if __name__ == "__main__":
    run_all_datasets.main()
