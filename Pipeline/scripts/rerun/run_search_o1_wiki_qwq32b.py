"""Search-o1 (wiki / FlashRAG e5) launcher preset for QwQ-32B.

The original TriviaQA and 2Wiki QwQ-32B search_o1_wiki runs (April) predate
cost tracking and have no cost.json. This script re-runs those datasets with
the current pipeline so cost.json is generated alongside the results.

QwQ-32B requires tensor-parallelism across multiple GPUs (the model is ~64 GB
in float16). Pass at least 2 GPUs via --gpus. Unlike the 8B/14B launchers,
auto-parallel dataset splitting is NOT applied here: all specified GPUs are
dedicated to a single vLLM TP worker.

Any flag the user passes explicitly takes precedence.

Examples:
    # TriviaQA and 2Wiki sequentially, TP=2 on GPUs 2,3
    python run_search_o1_wiki_qwq32b.py --gpus 2,3 --retriever_gpus 0,1 \\
        --dataset triviaqa,2wiki

    # single dataset
    python run_search_o1_wiki_qwq32b.py --gpus 4,5 --retriever_gpus 0,1 \\
        --dataset triviaqa
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

# QwQ-32B needs all assigned GPUs for TP — do NOT split into per-dataset
# parallel groups the way the 8B/14B launchers do.

SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)
import run_search_o1_wiki  # noqa: E402

if __name__ == "__main__":
    run_search_o1_wiki.main()
