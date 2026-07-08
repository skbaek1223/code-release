"""Re-Guide ablation launcher (QwQ-32B): retrieval-guide ONLY.

Disables the per-turn evaluator and the [Reasoning Guide] system
messages (--no_reasoning_guide). The per-turn extractor still runs and
its extracted facts are injected as the search result. The planner/steps
model that emits a numbered retrieval guide before reasoning starts also
remains active.

QwQ-32B uses run_all_datasets.py defaults (max_model_len=40960,
max_new_tokens=16384, temperature=0.7). Output directories get a
".retrieval_only" suffix so they don't collide with full-pipeline runs.

The ablation flags only exist in run_re_guide_2.py, so SCRIPT_PATH is
switched away from the default run_re_guide.py.

Examples:
    # sequential (TP=2 on a single 2-GPU group)
    python run_all_datasets_qwq32b_retrieval_only.py --gpus 2,3 --retriever_gpus 0,1

    # parallel (two independent 2-GPU workers)
    python run_all_datasets_qwq32b_retrieval_only.py --parallel \\
        --gpu_groups 2,3;4,5 --retriever_gpus 0,1

    # single dataset on one 2-GPU group
    python run_all_datasets_qwq32b_retrieval_only.py --gpus 2,3 --retriever_gpus 0,1 \\
        --dataset nq,ambigqa,hotpotqa,musique
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_datasets  # noqa: E402

ABLATION_SUFFIX = ".retrieval_only"
ABLATION_FLAG = "--no_reasoning_guide"

PRESET = {
    "--model_path": "/mnt/raid6/skbaek1223/models/QwQ-32B",
}


def _has_flag(flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])


for _flag, _val in PRESET.items():
    if not _has_flag(_flag):
        sys.argv += [_flag, _val]


# Ablation flags only exist in run_re_guide_2.py
run_all_datasets.SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "run_re_guide_2.py"
)

# QwQ-32B preset: skip triviaqa_a/_b variants (run plain triviaqa only).
_SKIP = {"triviaqa_a", "triviaqa_b"}
run_all_datasets.DATASETS = [d for d in run_all_datasets.DATASETS if d[0] not in _SKIP]


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
