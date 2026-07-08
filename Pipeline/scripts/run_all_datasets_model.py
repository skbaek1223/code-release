"""Re-Guide all-datasets launcher: single entry point for every per-model
preset. Replaces the previous run_all_datasets_qwen3_4b.py / _8b.py / _14b.py
/ run_all_datasets_r1_llama8b.py / run_all_datasets_r1_qwen14b.py files.

Thin wrapper around run_all_datasets.py: injects the preset's model path,
context window, and sampling defaults, then hands off to run_all_datasets.main().

GPU scheduling: each GPU in --gpus runs ONE dataset at a time (TP=1), and
all GPUs are saturated in parallel. With `--gpus 6,7`, two datasets run
concurrently; as each finishes, the next pending dataset is dispatched to
the freed GPU. Pass --no_auto_parallel to force tensor parallelism across
all listed GPUs on a single dataset instead. Any flag you pass explicitly
takes precedence over the preset's defaults.

Presets:
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

Examples:
    # 4 datasets, 2 at a time (1 GPU per dataset)
    python run_all_datasets_model.py --preset qwen3_8b --gpus 6,7 --retriever_gpus 0,1

    # single dataset on one GPU
    python run_all_datasets_model.py --preset r1_llama8b --gpus 6 \\
        --retriever_gpus 0,1 --dataset musique

    # force TP=2 across 6,7 (sequential, one dataset uses both GPUs)
    python run_all_datasets_model.py --preset qwen3_14b --gpus 6,7 \\
        --retriever_gpus 4,5 --dataset ambigqa --no_auto_parallel
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from launcher_common import apply_preset, auto_parallelize_gpus, flag_value, remove_flag

MULTI_HOP_DATASETS = {"hotpotqa", "2wiki", "musique"}
QWEN3_DATASETS = {"nq", "ambigqa", "hotpotqa", "musique"}
TRIVIAQA_SPLIT = {"triviaqa_a", "triviaqa_b"}

PRESETS = {
    "qwen3_4b": dict(
        flags={
            "--model_path":     "/mnt/raid6/skbaek1223/models/Qwen3-4B",
            "--max_model_len":  "32768",
            "--max_new_tokens": "16384",
            "--temperature":    "0.6",
        },
        budget_singlehop="30", budget_multihop="45",
        thinking_mode=True,
        keep_datasets=QWEN3_DATASETS,
    ),
    "qwen3_8b": dict(
        flags={
            "--model_path":     "/mnt/raid6/skbaek1223/models/Qwen3-8B",
            "--max_model_len":  "32768",
            "--max_new_tokens": "16384",
            "--temperature":    "0.6",
        },
        budget_singlehop="30", budget_multihop="45",
        thinking_mode=True,
        keep_datasets=QWEN3_DATASETS,
    ),
    "qwen3_14b": dict(
        flags={
            "--model_path":     "/mnt/raid6/skbaek1223/models/Qwen3-14B",
            "--max_model_len":  "32768",
            "--max_new_tokens": "16384",
            "--temperature":    "0.6",
        },
        thinking_mode=True,
        keep_datasets=QWEN3_DATASETS,
    ),
    "r1_llama8b": dict(
        flags={
            "--model_path":     "/mnt/raid6/skbaek1223/models/DeepSeek-R1-Distill-Llama-8B",
            "--max_model_len":  "32768",
            "--max_new_tokens": "16384",
            "--temperature":    "0.6",
        },
        budget_singlehop="30", budget_multihop="45",
        skip_datasets=TRIVIAQA_SPLIT,
    ),
    "r1_qwen14b": dict(
        flags={
            "--model_path":     "/mnt/raid6/skbaek1223/models/DeepSeek-R1-Distill-Qwen-14B",
            "--max_model_len":  "32768",
            "--max_new_tokens": "16384",
            "--temperature":    "0.6",
        },
        skip_datasets=TRIVIAQA_SPLIT,
    ),
}


def main():
    preset_name = flag_value("--preset")
    if preset_name is None or preset_name not in PRESETS:
        raise SystemExit(f"--preset is required, one of: {sorted(PRESETS)}")
    remove_flag("--preset")
    cfg = PRESETS[preset_name]

    apply_preset(cfg["flags"])
    auto_parallelize_gpus()

    import run_all_datasets  # noqa: E402

    _orig_build_cmd = run_all_datasets.build_cmd

    def _build_cmd(args, dataset_name, data_file, split, retriever_url):
        cmd, qa_data_path, dataset_output_dir = _orig_build_cmd(
            args, dataset_name, data_file, split, retriever_url)
        if "budget_singlehop" in cfg:
            budget = (cfg["budget_multihop"] if dataset_name in MULTI_HOP_DATASETS
                      else cfg["budget_singlehop"])
            cmd += ["--budget_base", budget]
        if cfg.get("thinking_mode"):
            cmd += ["--thinking_mode"]
        return cmd, qa_data_path, dataset_output_dir

    run_all_datasets.build_cmd = _build_cmd

    if "keep_datasets" in cfg:
        run_all_datasets.DATASETS = [
            d for d in run_all_datasets.DATASETS if d[0] in cfg["keep_datasets"]]
    elif "skip_datasets" in cfg:
        run_all_datasets.DATASETS = [
            d for d in run_all_datasets.DATASETS if d[0] not in cfg["skip_datasets"]]

    run_all_datasets.main()


if __name__ == "__main__":
    main()
