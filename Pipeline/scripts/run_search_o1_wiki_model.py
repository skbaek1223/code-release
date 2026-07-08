"""Search-o1 (wiki / FlashRAG e5) launcher: single entry point for every
per-model preset. Replaces the previous run_search_o1_wiki_r1_llama8b.py /
run_search_o1_wiki_r1_qwen14b.py files.

Thin wrapper around run_search_o1_wiki.py: injects the preset's model path,
context window, and sampling defaults, then hands off to
run_search_o1_wiki.main().

GPU scheduling: each GPU in --gpus runs ONE dataset at a time (TP=1), and
all GPUs are saturated in parallel. With `--gpus 6,7`, two datasets run
concurrently; as each finishes, the next pending dataset is dispatched to
the freed GPU. Pass --no_auto_parallel to force tensor parallelism across
all listed GPUs on a single dataset instead. Any flag you pass explicitly
takes precedence over the preset's defaults.

The QwQ-32B Search-o1 preset lives separately at
rerun/run_search_o1_wiki_qwq32b.py: it needs multi-GPU tensor parallelism
without the auto-parallel dataset splitting this launcher applies.

Presets:
    r1_llama8b   DeepSeek-R1-Distill-Llama-8B. The model has 131k native
                 context via llama3 RoPE scaling, so larger windows than
                 the default are also valid if KV cache budget allows.
    r1_qwen14b   DeepSeek-R1-Distill-Qwen-14B. Base Qwen2.5 has 131k native
                 context, same headroom note as r1_llama8b.

Examples:
    # 8 datasets, 2 at a time (1 GPU per dataset)
    python run_search_o1_wiki_model.py --preset r1_llama8b --gpus 6,7 --retriever_gpus 4,5

    # single dataset on one GPU
    python run_search_o1_wiki_model.py --preset r1_qwen14b --gpus 6 \\
        --retriever_gpus 0,1 --dataset 2wiki,musique

    # force TP=2 across 6,7 (sequential, one dataset uses both GPUs)
    python run_search_o1_wiki_model.py --preset r1_llama8b --gpus 6,7 \\
        --retriever_gpus 4,5 --dataset ambigqa --no_auto_parallel
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from launcher_common import apply_preset, auto_parallelize_gpus, flag_value, remove_flag

PRESETS = {
    "r1_llama8b": {
        "--model_path":         "/mnt/raid6/skbaek1223/models/DeepSeek-R1-Distill-Llama-8B",
        "--max_model_len":      "32768",
        "--max_new_tokens":     "16384",
        "--temperature":        "0.6",
        "--top_p":              "0.95",
        "--top_k_sampling":     "20",
        "--repetition_penalty": "1.0",
    },
    "r1_qwen14b": {
        "--model_path":         "/mnt/raid6/skbaek1223/models/DeepSeek-R1-Distill-Qwen-14B",
        "--max_model_len":      "32768",
        "--max_new_tokens":     "16384",
        "--temperature":        "0.6",
        "--top_p":              "0.95",
        "--top_k_sampling":     "20",
        "--repetition_penalty": "1.0",
    },
}


def main():
    preset_name = flag_value("--preset")
    if preset_name is None or preset_name not in PRESETS:
        raise SystemExit(f"--preset is required, one of: {sorted(PRESETS)}")
    remove_flag("--preset")

    apply_preset(PRESETS[preset_name])
    auto_parallelize_gpus()

    import run_search_o1_wiki  # noqa: E402
    run_search_o1_wiki.main()


if __name__ == "__main__":
    main()
