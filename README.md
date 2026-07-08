# Re-Guide

**Re-Guide** augments agentic search-reasoning models with two guide modules:

* **Retrieval-Guide**: a fine-tuned planner that generates a step-by-step retrieval plan before search begins.
* **Reasoning-Guide**: a per-turn evaluator that checks whether retrieved evidence is sufficient, extracts useful facts, and injects reasoning-budget hints into the reasoning trace.

This repository contains the code for training the Retrieval-Guide planner, running the Re-Guide inference pipeline (including baselines), and evaluating the results.

See [Data](#data) for what you need to prepare to run the pipeline.

---

## Repository Structure

```text
Planner/
  precompute/
    Build question–retrieval-step pairs from NQ / HotpotQA and judge them.

  sft/
    Planner supervised fine-tuning entry point using TRL.

scripts/
  prompts.py
    Prompt templates for QA, retrieval evaluation, and fact extraction.

  retriever_server.py
    FAISS + e5 retriever HTTP server. Intended to run on dedicated GPUs.

  retriever_utils.py
    HTTP client used by the Re-Guide runners.

  evaluate.py
    Answer extraction, normalization, and evaluation metrics.

  run_re_guide.py
    Re-Guide runner: the step-aware retrieval-reasoning loop.

  run_all_datasets.py
    Multi-dataset launcher. Spawns the retriever server and vLLM workers,
    then drives run_re_guide.py once per dataset. Supports --preset
    {qwen3_4b, qwen3_8b, qwen3_14b, r1_llama8b, r1_qwen14b} to switch models.

  run_search_o1_wiki.py
    Search-o1 baseline runner. Supports --preset {r1_llama8b, r1_qwen14b}.

  launcher_common.py
    Argv-patching helpers used by run_search_o1_wiki.py's --preset support.

  merge_lora.py
    Merge a trained LoRA adapter into the base model.

checkpoints/qwen3-8b-planner/
  README.md
    Model card for the fine-tuned planner.

data/
  val/
    Planner validation splits for NQ, HotpotQA, and MuSiQue.

  eval/
    Full-pipeline evaluation splits for NQ, AmbigQA, TriviaQA, HotpotQA,
    2WikiMultihopQA, and MuSiQue.

  sft/
    Planner SFT training data.

requirements.txt
```

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repository root:

```bash
OPENAI_API_KEY=...
```

The OpenAI API key is used to call `gpt-4.1` for goal generation in `Planner/02_generate_goals.py`. The retrieval-step generation and judging scripts in `Planner/precompute` instead call a local vLLM server, so they don't need this key.

---

## Data

To run the pipeline you need to prepare the following:

| Resource                    | Description                                                                |
| --------------------------- | --------------------------------------------------------------------------- |
| `data/sft/`                 | Planner SFT training data                                                    |
| Raw train datasets          | NQ and HotpotQA train splits — input to `Planner/precompute`, which builds the SFT data above |
| `data/val/`                 | Planner validation splits for NQ, HotpotQA, and MuSiQue                     |
| `data/eval/`                | Full-pipeline evaluation splits for NQ, AmbigQA, TriviaQA, HotpotQA, 2WikiMultihopQA, and MuSiQue — used by `run_all_datasets.py` / `run_search_o1_wiki.py` |
| FAISS index                 | Wikipedia retrieval index used by `retriever_server.py`                      |
| Wikipedia corpus            | FlashRAG `wiki18_100w` corpus                                                |

Search caches (`scripts/cache/`, `scripts/retriever_cache/`) and run outputs (`scripts/outputs/`) are not something you prepare — they're generated automatically as the pipeline runs.

For raw dataset formats, see:

```text
Planner/precompute/01_a_load_nq.py
Planner/precompute/01_b_load_hotpotqa.py
```

The retriever uses the [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) `wiki18_100w` corpus with an `e5-base-v2` flat inner-product index. The default `--index_path` and `--corpus_path` values in `retriever_server.py` are local machine paths and should be overridden.

---

## Pipeline

Commands below that aren't under `Planner/` are run from `scripts/`.

### 1. Generate Planner Training Data

Run the scripts in `Planner/precompute`:

```text
01_*.py  -> load raw datasets
02_*.py  -> generate retrieval steps
03_*.py  -> judge retrieval steps using LLM-as-judge
04_extract_hard.py -> extract the hard subset used for SFT
```

---

### 2. Train the Retrieval-Guide Planner

You can train the planner using either:

```bash
python Planner/04_sft_data_and_train.py
```

or:

```bash
python Planner/sft/train.py
```

After training, merge the LoRA adapter into the base model:

```bash
python merge_lora.py
```

---

### 3. Start the Retriever Server

Run the retriever server with dedicated GPUs:

```bash
python retriever_server.py \
  --gpus <retriever_gpu_ids> \
  --index_path <path_to_faiss_index> \
  --corpus_path <path_to_wikipedia_corpus>
```

The retriever server runs independently of the vLLM workers.

---

### 4. Run Re-Guide Inference

Run the full multi-dataset pipeline (QwQ-32B by default):

```bash
python run_all_datasets.py \
  --gpus <vllm_gpu_ids> \
  --retriever_gpus <retriever_gpu_ids>
```

If no retriever server is reachable, the launcher automatically starts one.

You can also use a per-model preset, for example:

```bash
python run_all_datasets.py --preset qwen3_8b \
  --gpus <vllm_gpu_ids> \
  --retriever_gpus <retriever_gpu_ids>
```

Available presets: `qwen3_4b`, `qwen3_8b`, `qwen3_14b`, `r1_llama8b`, `r1_qwen14b`.

---

### 5. Run Baselines

Search-o1 baseline (QwQ-32B by default), directly or via a per-model preset:

```bash
python run_search_o1_wiki.py \
  --gpus <vllm_gpu_ids> \
  --retriever_gpus <retriever_gpu_ids>

python run_search_o1_wiki.py --preset r1_llama8b \
  --gpus <vllm_gpu_ids> \
  --retriever_gpus <retriever_gpu_ids>
```

Available presets: `r1_llama8b`, `r1_qwen14b`.

---

### 6. Evaluate Results

Evaluation is automatically invoked by the runners, but it can also be run standalone:

```bash
python evaluate.py
```

The evaluation script handles answer extraction, normalization, and metric computation.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
