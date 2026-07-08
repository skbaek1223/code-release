"""Launcher: 2WikiMQA compositional 500-sample validation with the patched
extractor prompt, on QwQ-32B.

Why a separate launcher: the main launcher's DATASETS list is hard-coded.
This script adds the 500-sample subset entry and points the worker at
run_re_guide_extractor_fix.py instead of run_re_guide.py. Everything else
(retriever startup, GPU sanity check, evaluation) reuses run_all_datasets.

Output goes to .extractor_fix-suffixed dir so it never collides with the main
2wiki QwQ result. The retrieval cache (search_cache.e5.2wiki.json) is reused
intact: cache keys are query strings, and the patched extractor does not
change query generation, so most lookups will hit the existing cache.

Usage:
    python run_extractor_fix_2wiki_qwq.py --gpus 2,3 --retriever_gpus 0,1
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_all_datasets  # noqa: E402

OUTPUT_SUFFIX = ".extractor_fix"
SUBSET_DATA_FILE = "2wiki_dev_compositional500.json"

PRESET = {
    "--model_path": "/mnt/raid6/skbaek1223/models/QwQ-32B",
    "--max_model_len": "40960",
    "--max_new_tokens": "16384",
    "--temperature": "0.7",
}


def _has_flag(flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:])


for _flag, _val in PRESET.items():
    if not _has_flag(_flag):
        sys.argv += [_flag, _val]


# Use our patched runner instead of the default.
run_all_datasets.SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "run_re_guide_extractor_fix.py"
)

# Replace DATASETS with just our 500-sample subset. The dataset_name is still
# "2wiki" so MULTI_HOP_DATASETS detection works and budgets resolve correctly.
run_all_datasets.DATASETS = [
    ("2wiki", SUBSET_DATA_FILE, "dev"),
]


_prev_build_cmd = run_all_datasets.build_cmd


def _build_cmd_fix(args, dataset_name, data_file, split, retriever_url):
    cmd, qa_data_path, dataset_output_dir = _prev_build_cmd(
        args, dataset_name, data_file, split, retriever_url)
    new_out = dataset_output_dir + OUTPUT_SUFFIX
    if "--output_dir" in cmd:
        i = cmd.index("--output_dir")
        cmd[i + 1] = new_out
    return cmd, qa_data_path, new_out


run_all_datasets.build_cmd = _build_cmd_fix


if __name__ == "__main__":
    run_all_datasets.main()
