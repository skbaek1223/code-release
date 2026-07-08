"""Re-Guide ablation launcher (Qwen-14B): retrieval-guide ONLY.

Disables the per-turn evaluator and the [Reasoning Guide] system
messages (--no_reasoning_guide). The per-turn extractor still runs and
its extracted facts are injected as the search result. The planner/steps
model that emits a numbered retrieval guide before reasoning starts also
remains active.

Inherits all defaults (model path, sampling, GPU split) from
run_all_datasets_r1_qwen14b.py. Output directories get a
".retrieval_only" suffix so they don't collide with full-pipeline runs.

Examples:
    # 8 datasets, 2 at a time
    python run_all_datasets_r1_qwen14b_retrieval_only.py --gpus 6,7 --retriever_gpus 0,1

    # single dataset on one GPU
    python run_all_datasets_r1_qwen14b_retrieval_only.py --gpus 3 --retriever_gpus 0,1 --dataset ambigqa,hotpotqa,musique
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_datasets_r1_qwen14b  # noqa: E402,F401  (imports for side effects)
import run_all_datasets  # noqa: E402

ABLATION_SUFFIX = ".retrieval_only"
ABLATION_FLAG = "--no_reasoning_guide"

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
