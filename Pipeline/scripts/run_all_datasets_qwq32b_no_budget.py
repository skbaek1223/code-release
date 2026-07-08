"""Re-Guide ablation launcher (QwQ-32B): full pipeline WITHOUT budget hint.

The full pipeline runs (planner + extractor + evaluator + [Reasoning Guide])
but the word-budget hint ("use up to N words") is omitted from the
Insufficient [Reasoning Guide] message. All other behaviour is identical to
the standard full-pipeline run.

Output directories get a ".no_budget" suffix so they don't collide with
full-pipeline runs.

Examples:
    python run_all_datasets_qwq32b_no_budget.py --gpus 2,3 --retriever_gpus 0,1

    python run_all_datasets_qwq32b_no_budget.py --gpus 2,3 --retriever_gpus 0,1 \\
        --dataset nq,ambigqa,hotpotqa,musique
"""
import os
import sys

PRESET = {
    "--model_path": "/mnt/raid6/skbaek1223/models/QwQ-32B",
}


def _has_flag(flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])


for _flag, _val in PRESET.items():
    if not _has_flag(_flag):
        sys.argv += [_flag, _val]


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_datasets  # noqa: E402

run_all_datasets.SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "run_re_guide_2.py"
)

_SKIP = {"triviaqa_a", "triviaqa_b"}
run_all_datasets.DATASETS = [d for d in run_all_datasets.DATASETS if d[0] not in _SKIP]

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
