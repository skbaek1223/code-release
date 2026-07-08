"""Re-Guide ablation launcher (Llama-8B): reasoning guide WITHOUT budget hint.

The full pipeline runs (planner + extractor + evaluator + [Reasoning Guide])
but the word-budget hint ("use up to N words") is omitted from the
Insufficient [Reasoning Guide] message. All other behaviour is identical to
the standard full-pipeline run.

Inherits all defaults (model path, sampling, GPU split, per-dataset
budget_base) from run_all_datasets_r1_llama8b.py. Output directories get a
".no_budget" suffix so they don't collide with full-pipeline runs.

Examples:
    # 8 datasets, 2 at a time (1 GPU per dataset)
    python run_all_datasets_r1_llama8b_no_budget.py --gpus 6,7 --retriever_gpus 0,1

    # single dataset on one GPU
    python run_all_datasets_r1_llama8b_no_budget.py --gpus 6 --retriever_gpus 0,1 --dataset nq,ambigqa
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_datasets_r1_llama8b  # noqa: E402,F401  (imports for side effects)
import run_all_datasets  # noqa: E402

ABLATION_SUFFIX = ".no_budget"
ABLATION_FLAG = "--no_budget"

_prev_build_cmd = run_all_datasets.build_cmd


def _build_cmd_ablation(args, dataset_name, data_file, split, retriever_url):
    cmd, qa_data_path, dataset_output_dir = _prev_build_cmd(
        args, dataset_name, data_file, split, retriever_url)
    new_out = dataset_output_dir + ABLATION_SUFFIX
    if "--output_dir" in cmd:
        i = cmd.index("--output_dir")
        cmd[i + 1] = new_out
    cmd += [ABLATION_FLAG]
    return cmd, qa_data_path, new_out


run_all_datasets.build_cmd = _build_cmd_ablation


if __name__ == "__main__":
    run_all_datasets.main()
