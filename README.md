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

Pipeline/
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
      then drives run_re_guide.py once per dataset.

    launcher_common.py
      Argv-patching helpers shared by the preset launchers below.

    run_all_datasets_model.py
      Per-model launcher, selected with --preset {qwen3_4b, qwen3_8b,
      qwen3_14b, r1_llama8b, r1_qwen14b}.

    run_search_o1_wiki.py
      Search-o1 baseline runner.

    run_search_o1_wiki_model.py
      Per-model Search-o1 launcher, selected with --preset {r1_llama8b,
      r1_qwen14b}. The QwQ-32B preset lives separately at
      rerun/run_search_o1_wiki_qwq32b.py (needs multi-GPU tensor
      parallelism without auto-parallel dataset splitting).

    merge_lora.py
      Merge a trained LoRA adapter into the base model.

checkpoints/qwen3-8b-planner/
  README.md
    Model card for the fine-tuned planner.

data/
  val/
    Evaluation sets for NQ, HotpotQA, and MuSiQue.

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

| Resource                      | Description                                                    |
| ------------------------------ | -------------------------------------------------------------- |
| `data/val/`                    | Evaluation splits for NQ, HotpotQA, and MuSiQue                 |
| `data/sft/`                    | Planner SFT training data                                      |
| Raw datasets                   | NQ, HotpotQA, MuSiQue, 2WikiMultihopQA, TriviaQA, and AmbigQA   |
| FAISS index                    | Wikipedia retrieval index used by `retriever_server.py`         |
| Wikipedia corpus               | FlashRAG `wiki18_100w` corpus                                   |
| Search caches                  | `Pipeline/scripts/cache/`, `retriever_cache/`                   |
| Run outputs                    | `outputs/`                                                      |

For raw dataset formats, see:

```text
Planner/precompute/01_a_load_nq.py
Planner/precompute/01_b_load_hotpotqa.py
```

The retriever uses the [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) `wiki18_100w` corpus with an `e5-base-v2` flat inner-product index. The default `--index_path` and `--corpus_path` values in `retriever_server.py` are local machine paths and should be overridden.

---

## Pipeline

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
python Pipeline/scripts/merge_lora.py
```

---

### 3. Start the Retriever Server

Run the retriever server with dedicated GPUs:

```bash
python Pipeline/scripts/retriever_server.py \
  --gpus <retriever_gpu_ids> \
  --index_path <path_to_faiss_index> \
  --corpus_path <path_to_wikipedia_corpus>
```

The retriever server runs independently of the vLLM workers.

---

### 4. Run Re-Guide Inference

Run the full multi-dataset pipeline:

```bash
python Pipeline/scripts/run_all_datasets.py \
  --gpus <vllm_gpu_ids> \
  --retriever_gpus <retriever_gpu_ids>
```

If no retriever server is reachable, the launcher automatically starts one.

You can also use a per-model preset, for example:

```bash
python Pipeline/scripts/run_all_datasets_model.py --preset qwen3_8b \
  --gpus <vllm_gpu_ids> \
  --retriever_gpus <retriever_gpu_ids>
```

Available presets: `qwen3_4b`, `qwen3_8b`, `qwen3_14b`, `r1_llama8b`, `r1_qwen14b`.

---

### 5. Run Baselines

Search-o1 baseline, directly or via a per-model preset:

```bash
python Pipeline/scripts/run_search_o1_wiki.py \
  --gpus <vllm_gpu_ids> \
  --retriever_gpus <retriever_gpu_ids>

python Pipeline/scripts/run_search_o1_wiki_model.py --preset r1_llama8b \
  --gpus <vllm_gpu_ids> \
  --retriever_gpus <retriever_gpu_ids>
```

Available presets: `r1_llama8b`, `r1_qwen14b`. The QwQ-32B Search-o1 preset lives separately at `Pipeline/scripts/rerun/run_search_o1_wiki_qwq32b.py`.

---

### 6. Evaluate Results

Evaluation is automatically invoked by the runners, but it can also be run standalone:

```bash
python Pipeline/scripts/evaluate.py
```

The evaluation script handles answer extraction, normalization, and metric computation.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
