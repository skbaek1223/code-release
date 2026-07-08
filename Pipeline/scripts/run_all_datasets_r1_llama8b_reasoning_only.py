"""Re-Guide ablation launcher (Llama-8B): reasoning-guide ONLY.

Disables the planner/steps model (--no_retrieval_guide). The user prompt
is built without a "Retrieval Guide:" section. The per-turn extractor +
evaluator + [Reasoning Guide] messages remain active.

Inherits all defaults (model path, sampling, GPU split, per-dataset
budget) from run_all_datasets_r1_llama8b.py. Output directories get a
".reasoning_only" suffix so they don't collide with full-pipeline runs.

Examples:
    # 8 datasets, 2 at a time
    python run_all_datasets_r1_llama8b_reasoning_only.py --gpus 6,7 --retriever_gpus 0,1

    # single dataset on one GPU
    python run_all_datasets_r1_llama8b_reasoning_only.py --gpus 6 --retriever_gpus 0,1 --dataset nq,ambigqa
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_datasets_r1_llama8b  # noqa: E402,F401  (imports for side effects)
import run_all_datasets  # noqa: E402

ABLATION_SUFFIX = ".reasoning_only"
ABLATION_FLAG = "--no_retrieval_guide"

_prev_build_cmd = run_all_datasets.build_cmd


def _build_cmd_ablation(args, dataset_name, data_file, split, retriever_url):
    cmd, qa_data_path, dataset_output_dir = _prev_build_cmd(
        args, dataset_name, data_file, split, retriever_url)
    new_out = dataset_output_dir + ABLATION_SUFFIX
    if "--output_dir" in cmd:
        i = cmd.index("--output_dir")
        cmd[i + 1] = new_out
    # The planner is disabled; strip the inherited --steps_model_path so we
    # don't carry along the planner checkpoint path the worker would ignore.
    while "--steps_model_path" in cmd:
        i = cmd.index("--steps_model_path")
        del cmd[i:i + 2]
    cmd += [ABLATION_FLAG]
    return cmd, qa_data_path, new_out


run_all_datasets.build_cmd = _build_cmd_ablation


if __name__ == "__main__":
    run_all_datasets.main()
