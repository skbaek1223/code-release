"""Search-o1 (wiki / FlashRAG e5) launcher preset for DeepSeek-R1-Distill-Qwen-14B.

Thin wrapper around run_search_o1_wiki.py: injects DeepSeek-R1 recommended
sampling defaults (temperature=0.6, top_p=0.95) and context window suited to
the 14B model on a single A6000 (max_model_len=32768, max_new_tokens=16384).
The base Qwen2.5 has 131k native context, so larger windows are also valid
if KV cache budget allows.

GPU scheduling: each GPU in --gpus runs ONE dataset at a time (TP=1), and
all GPUs are saturated in parallel. With `--gpus 6,7`, two datasets run
concurrently; as each finishes, the next pending dataset is dispatched to
the freed GPU. To override and use tensor parallelism across multiple GPUs
on a single dataset, pass `--no_auto_parallel`.

Any flag the user passes explicitly takes precedence.

Examples:
    # 8 datasets, 2 at a time (1 GPU per dataset)
    python run_search_o1_wiki_r1_qwen14b.py --gpus 6,7 --retriever_gpus 4,5

    # single dataset on one GPU
    python run_search_o1_wiki_r1_qwen14b.py --gpus 6 --retriever_gpus 0,1 --dataset nq,ambigqa,hotpotqa,musique

    # force TP=2 across 6,7 (sequential, one dataset uses both GPUs)
    python run_search_o1_wiki_r1_qwen14b.py --gpus 6,7 --retriever_gpus 4,5 \\
        --dataset ambigqa --no_auto_parallel
"""
import os
import sys

PRESET = {
    "--model_path":         "/mnt/raid6/skbaek1223/models/DeepSeek-R1-Distill-Qwen-14B",
    "--max_model_len":      "32768",
    "--max_new_tokens":     "16384",
    "--temperature":        "0.6",
    "--top_p":              "0.95",
    "--top_k_sampling":     "20",
    "--repetition_penalty": "1.0",
}


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
import run_search_o1_wiki  # noqa: E402

if __name__ == "__main__":
    run_search_o1_wiki.main()
