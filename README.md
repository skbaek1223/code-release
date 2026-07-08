# Re-Guide

Re-Guide augments agentic search-reasoning models with two guide modules:
- **Retrieval-Guide**: a fine-tuned planner that produces a retrieval plan (steps) before search begins.
- **Reasoning-Guide**: a per-turn evaluator that judges whether retrieved evidence is sufficient and injects extracted facts / budget hints into the reasoning trace.

This repository contains the code used to train the planner and run the Re-Guide inference pipeline and baselines. Raw experiment outputs, search caches, and large intermediate datasets are excluded (see [Data](#data) below).

## Repository layout

```
Planner/            Retrieval-Guide planner: training-data precompute, SFT, validation
  precompute/        Build (question, retrieval-steps) pairs from NQ / HotpotQA and judge them
sft/                 Planner SFT training entry point (TRL)
Pipeline/
  data/lambda_search/  Reasoning-budget lambda search results
  scripts/
    prompts.py             All prompt templates (QA, retrieval evaluator, extractor)
    retriever_server.py    FAISS + e5 retriever HTTP server (dedicated GPUs)
    retriever_utils.py     HTTP client used by the runners
    evaluate.py             Answer extraction / normalization / metrics
    run_re_guide_2.py       Re-Guide runner (current), with ablation flags
    run_re_guide.py         Earlier runner version, kept for reference
    run_re_guide_extractor_fix.py / prompts_extractor_fix.py
                            Patched extractor prompt used only for the 2WikiMQA
                            compositional validation run (main pipeline untouched)
    run_all_datasets.py     Multi-dataset launcher (spawns retriever server + vLLM workers)
    run_all_datasets_<model>[_<ablation>].py
                            Per-model launcher presets (Qwen3-4B/8B/14B, R1-Llama8B,
                            R1-Qwen14B, QwQ-32B) and ablations (reasoning_only,
                            retrieval_only, no_budget)
    run_search_o1_wiki*.py  Search-o1 baseline runners
    merge_lora.py           Merge a trained LoRA adapter into the base model
    measure_planner_latency.py / add_search_o1_wiki_infogen_cost.py
                            Latency/cost measurement scripts
checkpoints/qwen3-8b-planner/README.md   Model card for the fine-tuned planner (weights not included)
data/val/            Evaluation sets (NQ, HotpotQA, MuSiQue)
data/sft/            Planner SFT training data
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repo root with:

```
OPENAI_API_KEY=...
```

(used for LLM-as-judge / dataset-generation steps in `Planner/precompute`).

## Data

Included:
- `data/val/` — evaluation splits for NQ, HotpotQA, MuSiQue
- `data/sft/` — planner SFT training data
- `Pipeline/data/lambda_search/` — reasoning-budget lambda search results

Not included (regenerate locally, or point scripts at your own copies):
- Raw dataset downloads (NQ, HotpotQA, MuSiQue, 2WikiMultihopQA, TriviaQA, AmbigQA) — see `Planner/precompute/01_a_load_nq.py` / `01_b_load_hotpotqa.py` for the expected format.
- A FAISS retrieval index + Wikipedia corpus for `retriever_server.py`. This code uses the [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) `wiki18_100w` corpus with an `e5-base-v2` flat inner-product index; `retriever_server.py`'s `--index_path` / `--corpus_path` default to this machine's local paths and must be overridden.
- Search-result caches and run outputs (`Pipeline/scripts/cache/`, `retriever_cache/`, `outputs/`) — regenerated on first run.

## Pipeline

1. **Planner training data**: `Planner/precompute/01_*.py` → `02_*.py` (generate retrieval steps) → `03_*.py` (LLM-judge the steps) → `04_extract_hard.py` (extract the hard subset used for SFT).
2. **Planner SFT**: `Planner/04_sft_data_and_train.py` or `sft/train.py`, then `Pipeline/scripts/merge_lora.py` to merge the LoRA adapter into the base model.
3. **Retriever server**: `python Pipeline/scripts/retriever_server.py --gpus <ids> --index_path <path> --corpus_path <path>`. Runs on dedicated GPUs, independent of the vLLM workers.
4. **Re-Guide inference**: `python Pipeline/scripts/run_all_datasets.py --gpus <ids> --retriever_gpus <ids>` (auto-spawns the retriever server if none is reachable), or one of the per-model presets, e.g. `run_all_datasets_qwen3_8b.py`. Ablation flags (`--no_retrieval_guide`, `--no_reasoning_guide`, `--no_budget`) are documented in `run_re_guide_2.py`.
5. **Baselines**: `run_search_o1_wiki*.py`, `run_all_datasets_r1_*`, `run_all_datasets_qwq32b_*`.
6. **Evaluation**: `Pipeline/scripts/evaluate.py` (invoked by the runners; can also be run standalone on saved outputs).

## License

MIT — see [LICENSE](LICENSE).
