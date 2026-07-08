"""
Search-o1 baseline on Re-Guide's wiki (FlashRAG e5 / bm25) corpus.

Architecture:
    - FAISS retriever runs in a dedicated HTTP server (retriever_server.py)
      on its own GPUs. The launcher auto-spawns it if no server is already
      reachable at the configured URL; if an external server is already
      running, it is reused and NOT torn down on exit.
    - Each worker loads vLLM ONCE on its assigned GPUs and never swaps.

Usage:

    # table-driven single/multiple datasets
    python run_search_o1_wiki.py --dataset triviaqa,2wiki --gpus 2,3 --retriever_gpus 0,1
    python run_search_o1_wiki.py --gpus 2,3 --retriever_gpus 0,1 --dataset nq,ambigqa,hotpotqa,musique

    # all datasets sequentially
    python run_search_o1_wiki.py --gpus 4,3 --retriever_gpus 0,1

    # parallel workers (each group runs one dataset in parallel)
    python run_search_o1_wiki.py --parallel \\
        --retriever_gpus 0,1 --gpu_groups "2,3;4,5;6,7"

Per-model presets (--preset <name>): injects the model path and sampling
defaults for a specific model, then applies one-GPU-per-dataset
auto-parallelization when --gpus is given (pass --no_auto_parallel to force
tensor parallelism across all listed GPUs on a single dataset instead). Any
flag you pass explicitly still takes precedence over the preset's defaults.

    r1_llama8b   DeepSeek-R1-Distill-Llama-8B. The model has 131k native
                 context via llama3 RoPE scaling, so larger windows than
                 the default are also valid if KV cache budget allows.
    r1_qwen14b   DeepSeek-R1-Distill-Qwen-14B. Base Qwen2.5 has 131k native
                 context, same headroom note as r1_llama8b.

    python run_search_o1_wiki.py --preset r1_llama8b --gpus 6,7 --retriever_gpus 4,5
    python run_search_o1_wiki.py --preset r1_qwen14b --gpus 6 --retriever_gpus 0,1 --dataset 2wiki,musique
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from launcher_common import apply_preset, auto_parallelize_gpus, flag_value, remove_flag

# ── Per-model presets ────────────────────────────────────────────────────
# Resolved via sys.argv (before argparse, and before _early_set_cuda_visible_devices()
# below reads --gpus/--gpu_groups) so preset-driven auto-parallelization is
# already reflected in the raw CLI args by the time CUDA_VISIBLE_DEVICES is set.
PRESETS = {
    "r1_llama8b": {
        "--model_path":         "/mnt/raid6/skbaek1223/models/DeepSeek-R1-Distill-Llama-8B",
        "--max_model_len":      "32768",
        "--max_new_tokens":     "16384",
        "--temperature":        "0.6",
        "--top_p":              "0.95",
        "--top_k_sampling":     "20",
        "--repetition_penalty": "1.0",
    },
    "r1_qwen14b": {
        "--model_path":         "/mnt/raid6/skbaek1223/models/DeepSeek-R1-Distill-Qwen-14B",
        "--max_model_len":      "32768",
        "--max_new_tokens":     "16384",
        "--temperature":        "0.6",
        "--top_p":              "0.95",
        "--top_k_sampling":     "20",
        "--repetition_penalty": "1.0",
    },
}

_preset_name = flag_value("--preset")
if _preset_name is not None:
    if _preset_name not in PRESETS:
        raise SystemExit(f"--preset must be one of {sorted(PRESETS)}, got {_preset_name!r}")
    remove_flag("--preset")
    apply_preset(PRESETS[_preset_name])
    auto_parallelize_gpus()


def _early_set_cuda_visible_devices():
    """Apply --gpus / --gpu_groups to CUDA_VISIBLE_DEVICES before torch/vllm
    import. Setting it later (after torch has probed devices) causes vLLM TP
    workers to inherit stale queued CUDA calls and crash with
    `device >= 0 && device < num_gpus`."""
    argv = sys.argv[1:]
    parallel = "--parallel" in argv

    def _val(flag):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return argv[i + 1]
        for a in argv:
            if a.startswith(flag + "="):
                return a.split("=", 1)[1]
        return None

    gpus_val = _val("--gpus")
    groups_val = _val("--gpu_groups")
    if not parallel and gpus_val:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus_val
    elif parallel and groups_val:
        all_gpus = sorted({g.strip() for grp in groups_val.split(";")
                           for g in grp.split(",") if g.strip()},
                          key=lambda x: int(x) if x.isdigit() else 0)
        if all_gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(all_gpus)


_early_set_cuda_visible_devices()

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")

import re
import glob
import json
import time
import argparse
import subprocess
from typing import Optional, List, Dict

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import run_evaluation, extract_answer
from retriever_utils import (
    RemoteRetriever,
    ensure_retriever_server,
    stop_retriever_server,
)

# Special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

MULTI_HOP_DATASETS = {'hotpotqa', '2wiki', 'musique'}

# ── Dataset table (used when --qa_data_path is not supplied) ───────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.normpath(os.path.join(PROJECT_ROOT, "..", "data", "eval"))
DEFAULT_FINAL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "final_results")

DATASETS = [
    ("nq",         "nq_test.json",        "test"),
    ("triviaqa",   "triviaqa_test.json",  "test"),
    ("triviaqa_a", "triviaqa_test.json",  "test"),
    ("triviaqa_b", "triviaqa_test.json",  "test"),
    ("ambigqa",    "ambigqa_dev.json",    "dev"),
    ("hotpotqa",   "hotpotqa_dev.json",   "dev"),
    ("2wiki",      "2wiki_dev.json",      "dev"),
    ("musique",    "musique_dev.json",    "dev"),
]
DATASET_NAMES = [d[0] for d in DATASETS]

# Named subsets of a base dataset: slice the full file and report metrics
# under the base dataset's eval key.
DATASET_SUBSETS = {
    "triviaqa_a": {"slice": (0, 4000),    "eval_name": "triviaqa"},
    "triviaqa_b": {"slice": (4000, None), "eval_name": "triviaqa"},
}


# ---------------------------------------------------------------------------
# Prompts (inlined from Search-o1/scripts/prompts.py)
# ---------------------------------------------------------------------------
def get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform searches "
        "to help you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write a search query in the format <|begin_search_query|> your query here <|end_search_query|>, written as concise keywords only.\n"
        "Then, the system will search and provide you with helpful information "
        "in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns processed information)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_multiqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform searches "
        "to help you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write a search query in the format <|begin_search_query|> your query here <|end_search_query|>, written as concise keywords only.\n"
        "Then, the system will search and provide you with helpful information "
        "in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_task_instruction_openqa(question, model_name=None):
    if model_name == 'qwq':
        return (
            'Please answer the following question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return (
        'Please answer the following question. You should think step by step to solve it.\n\n'
        'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
        f'Question:\n{question}\n\n'
    )


def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing retrieved passages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Retrieved Passages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Retrieved Passages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Retrieved Passages:**
- Carefully review the content of each retrieved passage.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the retrieved passages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the passages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the passages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**
{prev_reasoning}

- **Current Search Query:**
{search_query}

- **Retrieved Passages:**
{document}

Now you should analyze each passage and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run Search-o1 with wiki (FlashRAG) retrieval.")

    # Data
    p.add_argument('--dataset', type=str, default=None,
                   help="Dataset(s) from the built-in table. Comma-separated list allowed "
                        "(e.g. 'nq,triviaqa,hotpotqa'). If omitted, run all.")
    p.add_argument('--dataset_name', type=str, default=None, choices=DATASET_NAMES,
                   help="Alias for --dataset (single-dataset mode; use --dataset for lists).")
    p.add_argument('--split', type=str, default=None,
                   help="Split label; if omitted, taken from the DATASETS table.")
    p.add_argument('--qa_data_path', type=str, default=None,
                   help="Explicit QA file; if omitted, derived from --data_dir + DATASETS table.")
    p.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                   help="Directory containing the dataset QA json files.")
    p.add_argument('--subset_num', type=int, default=-1)

    # Retrieval (remote retriever_server over HTTP)
    p.add_argument('--max_search_limit', type=int, default=20)
    p.add_argument('--max_turn', type=int, default=20)
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--max_doc_len', type=int, default=3000)
    p.add_argument('--retrieval_method', type=str, default='e5', choices=['e5', 'bm25'],
                   help="Tag used for the on-disk search cache filename; must match server.")
    p.add_argument('--retriever_url', type=str, default=None,
                   help="Base URL of the retriever server. If omitted, auto-spawn one "
                        "on --retriever_gpus (launcher mode).")
    p.add_argument('--retriever_gpus', type=str, default='0,1',
                   help="GPUs for the auto-spawned retriever server.")
    p.add_argument('--retriever_host', type=str, default='127.0.0.1')
    p.add_argument('--retriever_port', type=int, default=8765)
    p.add_argument('--retriever_startup_timeout', type=int, default=900)

    # Model / sampling
    p.add_argument('--model_path', type=str,
                   default='/mnt/raid6/skbaek1223/models/QwQ-32B')
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.8)
    p.add_argument('--top_k_sampling', type=int, default=20)
    p.add_argument('--repetition_penalty', type=float, default=None)
    p.add_argument('--max_model_len', type=int, default=40960)
    p.add_argument('--max_new_tokens', type=int, default=16384)

    # Output
    p.add_argument('--output_dir', type=str, default=None,
                   help="Per-dataset output dir (single-dataset mode). "
                        "If omitted, derives from --output_base_dir/model/dataset.")
    p.add_argument('--output_base_dir', type=str, default=DEFAULT_FINAL_OUTPUT_DIR,
                   help="Base dir for per-dataset outputs when --output_dir is not set.")
    p.add_argument('--search_cache_suffix', type=str, default='',
                   help="Optional suffix for the per-process search cache file "
                        "(avoids races in parallel mode).")

    # Sequential mode: GPUs for the single vLLM worker.
    p.add_argument('--gpus', type=str, default=None,
                   help="Comma-separated GPU ids for vLLM (sequential mode). "
                        "Must NOT overlap --retriever_gpus.")

    # Parallel mode: each group is an independent vLLM worker.
    p.add_argument('--parallel', action='store_true',
                   help="Run datasets concurrently; each GPU group is an independent "
                        "vLLM worker.")
    p.add_argument('--gpu_groups', type=str, default='2,3;4,5;6,7',
                   help="Semicolon-separated vLLM GPU groups. "
                        "Example: '2,3;4,5;6,7'. Must NOT overlap --retriever_gpus.")

    # --preset and --no_auto_parallel are resolved from sys.argv before this
    # parser ever runs (see the top of the module) and are registered here
    # only so they show up in --help.
    p.add_argument('--preset', type=str, default=None, choices=sorted(PRESETS),
                   help="Per-model preset; see module docstring for details.")
    p.add_argument('--no_auto_parallel', action='store_true',
                   help="With --preset, disable one-GPU-per-dataset auto-parallelization.")

    return p.parse_args()


def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


def resolve_run_list(args) -> List[Dict]:
    """Return the list of dataset runs to execute, each as a dict."""
    raw = args.dataset or args.dataset_name
    requested = [d.strip() for d in raw.split(',') if d.strip()] if raw else []
    valid = set(DATASET_NAMES)
    unknown = [d for d in requested if d not in valid]
    if unknown:
        raise ValueError(f"Unknown dataset(s): {unknown}. Valid: {sorted(valid)}")
    runs: List[Dict] = []

    def _subset_meta(name):
        meta = DATASET_SUBSETS.get(name, {})
        return meta.get('slice'), meta.get('eval_name', name)

    if args.qa_data_path:
        # Fully explicit single-run
        if len(requested) != 1:
            raise ValueError("--qa_data_path requires exactly one --dataset or --dataset_name.")
        single = requested[0]
        split = args.split or next((s for n, _, s in DATASETS if n == single), 'test')
        slc, eval_name = _subset_meta(single)
        runs.append({
            'dataset_name': single,
            'eval_name': eval_name,
            'slice': slc,
            'qa_data_path': args.qa_data_path,
            'split': split,
            'output_dir': args.output_dir,
        })
        return runs

    # Table-driven
    if requested:
        by_name = {n: (n, f, s) for n, f, s in DATASETS}
        table = [by_name[d] for d in requested]
    else:
        table = list(DATASETS)

    model_short = args.model_path.split('/')[-1].lower().replace('-instruct', '')
    single_explicit = len(requested) == 1
    for name, fname, split in table:
        slc, eval_name = _subset_meta(name)
        runs.append({
            'dataset_name': name,
            'eval_name': eval_name,
            'slice': slc,
            'qa_data_path': os.path.normpath(os.path.join(args.data_dir, fname)),
            'split': args.split or split,
            'output_dir': args.output_dir if (single_explicit and args.output_dir)
            else os.path.normpath(os.path.join(
                args.output_base_dir, f"{name}.{model_short}.search_o1_wiki")),
        })
    return runs


# ---------------------------------------------------------------------------
# Per-dataset pipeline
# ---------------------------------------------------------------------------
def run_one_dataset(run, args, retriever, tokenizer, llm, search_cache,
                    save_search_cache) -> None:
    dataset_name = run['dataset_name']
    eval_name = run.get('eval_name') or dataset_name
    data_slice = run.get('slice')
    qa_data_path = run['qa_data_path']
    split = run['split']
    output_dir = run['output_dir']

    if not os.path.exists(qa_data_path):
        print(f"[ERROR] Data file not found: {qa_data_path}, skipping {dataset_name}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # evaluate.run_evaluation writes timestamped names ({split}.M.D,H:MM[.metrics].json),
    # so prior runs accumulate instead of overwriting. Remove them up front.
    for old in sorted(glob.glob(os.path.join(output_dir, f'{split}.*.json'))):
        try:
            os.remove(old)
            print(f"  [overwrite] removed old result: {old}")
        except OSError as e:
            print(f"  [overwrite] could not remove {old}: {e}")

    print(f"\n{'=' * 60}")
    print(f"  RUNNING: {dataset_name} (split={split})  [search_o1_wiki]")
    print(f"  Data:    {qa_data_path}")
    print(f"  Output:  {output_dir}")
    print(f"{'=' * 60}\n")

    # ----- Load QA -----
    with open(qa_data_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    filtered_data = list(qa_data)
    if data_slice is not None:
        start, end = data_slice
        filtered_data = filtered_data[start:end]
        print(f"  Subset:  {dataset_name} -> [{start}:{end}] -> {len(filtered_data)} items")
    if args.subset_num != -1:
        filtered_data = filtered_data[:args.subset_num]

    # ----- Build initial prompts -----
    input_list = []
    for item in filtered_data:
        question = item['Question']
        if eval_name in MULTI_HOP_DATASETS:
            instruction = get_multiqa_search_o1_instruction(args.max_search_limit)
        else:
            instruction = get_singleqa_search_o1_instruction(args.max_search_limit)
        model_tag = 'qwq' if 'qwq' in args.model_path.lower() else None
        user_prompt = get_task_instruction_openqa(question, model_name=model_tag)
        msg = [{"role": "user", "content": instruction + user_prompt}]
        prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        input_list.append(prompt)

    active_sequences = [{
        'item': item,
        'initial_prompt': prompt,
        'prompt': prompt,
        'output': '',
        'finished': False,
        'history': [],
        'search_count': 0,
        'executed_search_queries': set(),
        'module_tokens': {            # per-module input/output token counts (this question)
            'reasoning': {'input': 0, 'output': 0},
            'infogen':   {'input': 0, 'output': 0},
        },
    } for item, prompt in zip(filtered_data, input_list)]

    # Prompt-token budget: leave a minimum output room of 512 tokens.
    # vLLM caps max_tokens by the remaining context window at runtime.
    prompt_token_budget = args.max_model_len - 512
    truncation_marker = "\n\n[... earlier turns truncated to fit context window ...]\n\n"

    def _enforce_prompt_budget(seqs):
        """When a sequence's prompt exceeds the budget, keep the first history
        chunk and greedily pack as many of the most recent chunks as fit.
        Force-finish if still over after trimming."""
        def _tok_len(s):
            return len(tokenizer.encode(s, add_special_tokens=False))

        for s in seqs:
            n_tok = _tok_len(s['prompt'])
            if n_tok <= prompt_token_budget:
                continue
            hist = s['history']
            if len(hist) >= 2:
                initial_tok = _tok_len(s['initial_prompt'])
                first_tok = _tok_len(hist[0])
                marker_tok = _tok_len(truncation_marker)
                available = prompt_token_budget - initial_tok - first_tok - marker_tok

                tail_parts = []
                used = 0
                for h in reversed(hist[1:]):
                    h_tok = _tok_len(h)
                    if used + h_tok > available:
                        break
                    tail_parts.append(h)
                    used += h_tok
                tail_parts.reverse()

                s['prompt'] = (s['initial_prompt'] + hist[0]
                               + truncation_marker + "".join(tail_parts))
                n_tok = _tok_len(s['prompt'])
            if n_tok > prompt_token_budget:
                print(f"  [truncate] seq over budget after trim ({n_tok} tok); force finish")
                s['finished'] = True

    # ----- Sampling params -----
    main_sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k_sampling,
        repetition_penalty=args.repetition_penalty,
        stop=[END_SEARCH_QUERY, tokenizer.eos_token],
        include_stop_str_in_output=True,
    )
    infogen_sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
    )

    def run_generation(sequences: List[Dict]) -> List:
        prompts = [s['prompt'] for s in sequences]
        return llm.generate(prompts, sampling_params=main_sampling)

    def generate_webpage_to_reasonchain_batch(
            prev_reasonings: List[str],
            search_queries: List[str],
            documents: List[str],
            batch_output_records: List[Dict]):
        """Returns parallel lists (extracted_infos, in_tok_counts, out_tok_counts)."""
        user_prompts = [
            get_webpage_to_reasonchain_instruction(r, sq, doc)
            for r, sq, doc in zip(prev_reasonings, search_queries, documents)
        ]
        prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": up}],
                                          tokenize=False, add_generation_prompt=True)
            for up in user_prompts
        ]
        output = llm.generate(prompts, sampling_params=infogen_sampling)
        raw_outputs = [out.outputs[0].text for out in output]
        in_tok_counts  = [len(out.prompt_token_ids) for out in output]
        out_tok_counts = [len(out.outputs[0].token_ids) for out in output]
        extracted_infos = [extract_answer(raw, mode='infogen') for raw in raw_outputs]
        for p, r, e in zip(prompts, raw_outputs, extracted_infos):
            batch_output_records.append({
                'prompt': p,
                'raw_output': r,
                'extracted_info': e,
            })
        return extracted_infos, in_tok_counts, out_tok_counts

    # ----- Main loop -----
    batch_output_records: List[Dict] = []
    start_time = time.time()
    turn = 0

    while True:
        pending = [s for s in active_sequences if not s['finished']]
        if not pending:
            break
        turn += 1
        if turn > args.max_turn:
            print(f"Max turn {args.max_turn} reached.")
            break
        print(f"\n----- Turn {turn} | {len(pending)} active -----")

        _enforce_prompt_budget(pending)
        pending = [s for s in pending if not s['finished']]
        if not pending:
            break

        outputs = run_generation(pending)

        # Pass 1: parse outputs, decide per-seq fate, collect retrieval candidates.
        retrieval_candidates = []  # (seq, search_query)
        for seq, out in zip(pending, outputs):
            text = out.outputs[0].text
            seq['module_tokens']['reasoning']['input']  += len(out.prompt_token_ids)
            seq['module_tokens']['reasoning']['output'] += len(out.outputs[0].token_ids)
            seq['history'].append(text)
            seq['prompt'] += text
            seq['output'] += text

            search_query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)

            if not (search_query and seq['output'].rstrip().endswith(END_SEARCH_QUERY)):
                seq['finished'] = True
                continue

            if turn == args.max_turn:
                seq['finished'] = True
                continue

            if seq['search_count'] >= args.max_search_limit:
                msg = (f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. "
                       f"You are not allowed to search.\n{END_SEARCH_RESULT}\n")
                seq['prompt'] += msg
                seq['output'] += msg
                seq['history'].append(msg)
                continue

            if search_query in seq['executed_search_queries']:
                msg = (f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. "
                       f"Please refer to previous results.\n{END_SEARCH_RESULT}\n")
                seq['prompt'] += msg
                seq['output'] += msg
                seq['history'].append(msg)
                continue

            retrieval_candidates.append((seq, search_query))

        # Pass 2: fetch fresh retrievals from the remote retriever server.
        fresh_queries = [q for _, q in retrieval_candidates if q not in search_cache]
        unique_fresh = list(dict.fromkeys(fresh_queries))
        if unique_fresh:
            print(f"[retrieval] running {len(unique_fresh)} fresh queries via {retriever.url}")
            try:
                batch_docs = retriever.batch_search(unique_fresh, args.top_k)
            except Exception as e:
                print(f"Retrieval batch error: {e}")
                batch_docs = [[] for _ in unique_fresh]
            for q, docs in zip(unique_fresh, batch_docs):
                search_cache[q] = docs

        # Pass 3: build infogen batch from (now fully cached) docs.
        batch_prev_reasonings: List[str] = []
        batch_search_queries: List[str] = []
        batch_documents: List[str] = []
        batch_sequences: List[Dict] = []

        for seq, search_query in retrieval_candidates:
            docs = search_cache.get(search_query, [])
            formatted = ""
            for i, d in enumerate(docs):
                formatted += f"**Passage {i + 1}:**\n{json.dumps(d, ensure_ascii=False, indent=2)}\n"

            # Truncate prev reasoning (match Search-o1 behavior)
            all_reasoning_steps = seq['output'].replace('\n\n', '\n').split("\n")
            truncated_prev_reasoning = ""
            for i, step in enumerate(all_reasoning_steps):
                truncated_prev_reasoning += f"Step {i + 1}: {step}\n\n"
            prev_steps = truncated_prev_reasoning.split('\n\n')
            if len(prev_steps) > 5:
                kept = ""
                for i, step in enumerate(prev_steps):
                    if (i == 0 or i >= len(prev_steps) - 4
                            or BEGIN_SEARCH_QUERY in step or BEGIN_SEARCH_RESULT in step):
                        kept += step + '\n\n'
                    else:
                        if kept[-len('\n\n...\n\n'):] != '\n\n...\n\n':
                            kept += '...\n\n'
                truncated_prev_reasoning = kept
            truncated_prev_reasoning = truncated_prev_reasoning.strip('\n')

            batch_prev_reasonings.append(truncated_prev_reasoning)
            batch_search_queries.append(search_query)
            batch_documents.append(formatted)
            batch_sequences.append(seq)

            seq['search_count'] += 1
            seq['executed_search_queries'].add(search_query)

        if batch_sequences:
            print(f"[infogen] extracting info for {len(batch_sequences)} queries")
            analyses, infogen_in_toks, infogen_out_toks = generate_webpage_to_reasonchain_batch(
                prev_reasonings=batch_prev_reasonings,
                search_queries=batch_search_queries,
                documents=batch_documents,
                batch_output_records=batch_output_records,
            )
            for seq, analysis, n_in, n_out in zip(
                    batch_sequences, analyses, infogen_in_toks, infogen_out_toks):
                seq['module_tokens']['infogen']['input']  += n_in
                seq['module_tokens']['infogen']['output'] += n_out
                append_text = f"\n\n{BEGIN_SEARCH_RESULT}{analysis}{END_SEARCH_RESULT}\n\n"
                seq['prompt'] += append_text
                seq['output'] += append_text
                seq['history'].append(append_text)

    total_time = time.time() - start_time

    output_list = [seq['output'] for seq in active_sequences]
    per_q_module_tokens = {
        mod: {
            'input':  [s['module_tokens'][mod]['input']  for s in active_sequences],
            'output': [s['module_tokens'][mod]['output'] for s in active_sequences],
        }
        for mod in ('reasoning', 'infogen')
    }
    run_evaluation(filtered_data, input_list, output_list,
                   eval_name, output_dir, total_time, split,
                   tokenizer=tokenizer,
                   per_q_module_tokens=per_q_module_tokens)

    save_search_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _run_parallel(args, runs, retriever_url):
    gpu_groups = [g.strip() for g in args.gpu_groups.split(';') if g.strip()]
    if not gpu_groups:
        raise SystemExit("--gpu_groups is empty")

    # Build per-dataset CLI invocations of this same script.
    base_cmd = [sys.executable, os.path.abspath(__file__)]
    common_args = [
        '--retrieval_method', args.retrieval_method,
        '--retriever_url', retriever_url,
        '--top_k', str(args.top_k),
        '--max_doc_len', str(args.max_doc_len),
        '--max_search_limit', str(args.max_search_limit),
        '--max_turn', str(args.max_turn),
        '--model_path', args.model_path,
        '--temperature', str(args.temperature),
        '--top_p', str(args.top_p),
        '--top_k_sampling', str(args.top_k_sampling),
        '--max_model_len', str(args.max_model_len),
        '--max_new_tokens', str(args.max_new_tokens),
        '--output_base_dir', args.output_base_dir,
        '--data_dir', args.data_dir,
    ]
    if args.subset_num != -1:
        common_args += ['--subset_num', str(args.subset_num)]
    if args.repetition_penalty is not None:
        common_args += ['--repetition_penalty', str(args.repetition_penalty)]

    summary = []
    active_procs = []

    def _kill_active_workers():
        for name, p, log_f, _t0 in active_procs:
            if p.poll() is None:
                print(f"[launcher] terminating worker {name} (pid={p.pid})")
                try:
                    p.terminate()
                except Exception:
                    pass
        for name, p, log_f, _t0 in active_procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                except Exception:
                    pass
            try:
                log_f.close()
            except Exception:
                pass

    try:
        idx = 0
        while idx < len(runs):
            batch = runs[idx: idx + len(gpu_groups)]
            active_procs = []
            for run, group in zip(batch, gpu_groups):
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = group
                env['VLLM_HOST_IP'] = '127.0.0.1'
                cmd = base_cmd + common_args + [
                    '--dataset', run['dataset_name'],
                    '--split', run['split'],
                    '--qa_data_path', run['qa_data_path'],
                    '--search_cache_suffix', run['dataset_name'],
                ]
                if run.get('output_dir'):
                    cmd += ['--output_dir', run['output_dir']]
                log_path = (run.get('output_dir') or
                            os.path.join(args.output_base_dir, run['dataset_name'])) + '.launch.log'
                os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
                log_f = open(log_path, 'w', buffering=1)
                print(f"[launcher] {run['dataset_name']} -> GPUs {group}  (log: {log_path})")
                p = subprocess.Popen(cmd, env=env, stdout=log_f,
                                     stderr=subprocess.STDOUT, start_new_session=True)
                active_procs.append((run['dataset_name'], p, log_f, time.time()))
            for name, p, log_f, t0 in active_procs:
                ret = p.wait()
                log_f.close()
                elapsed = time.time() - t0
                status = 'OK' if ret == 0 else f'FAIL (exit {ret})'
                summary.append((name, status, f'{elapsed:.0f}s'))
                print(f"[launcher] >> {name}: {status}  ({elapsed:.0f}s)")
            active_procs = []
            idx += len(gpu_groups)
    except KeyboardInterrupt:
        print("[launcher] KeyboardInterrupt — terminating workers")
        _kill_active_workers()
        raise
    finally:
        _kill_active_workers()

    print("\n" + "=" * 60)
    print("  SUMMARY (search_o1_wiki, parallel)")
    print("=" * 60)
    for name, status, elapsed in summary:
        print(f"  {name:12s}  {status:30s}  {elapsed}")
    print("=" * 60)


def _resolve_retriever(args, *, needs_spawn_if_missing: bool):
    """Return (retriever_url, spawned_proc_or_None).

    Worker mode: --retriever_url is passed, so reuse it (no spawn).
    Launcher mode: --retriever_url is None, so auto-spawn if needed.
    """
    if args.retriever_url:
        return args.retriever_url, None
    if not needs_spawn_if_missing:
        raise SystemExit("--retriever_url is required when not in launcher mode.")
    server_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "retriever_server.py")
    log_path = os.path.join(args.output_base_dir, "retriever_server.log")
    return ensure_retriever_server(
        retriever_gpus=args.retriever_gpus,
        host=args.retriever_host,
        port=args.retriever_port,
        retrieval_method=args.retrieval_method,
        top_k=args.top_k,
        startup_timeout=args.retriever_startup_timeout,
        log_path=log_path,
        server_script=server_script,
    )


def main():
    args = parse_args()

    if args.repetition_penalty is None:
        args.repetition_penalty = 1.05 if 'qwq' in args.model_path.lower() else 1.0

    # In sequential mode --gpus overrides CUDA_VISIBLE_DEVICES (for convenience).
    # In parallel mode --gpu_groups is used per worker and this flag is ignored.
    if args.gpus and not args.parallel:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    runs = resolve_run_list(args)

    # GPU overlap sanity check.
    retriever_set = {g.strip() for g in args.retriever_gpus.split(",") if g.strip()}
    if args.parallel:
        worker_sets = [{g.strip() for g in group.split(",") if g.strip()}
                       for group in args.gpu_groups.split(";") if group.strip()]
    elif args.gpus:
        worker_sets = [{g.strip() for g in args.gpus.split(",") if g.strip()}]
    else:
        worker_sets = []
    for ws in worker_sets:
        overlap = ws & retriever_set
        if overlap and not args.retriever_url:
            raise SystemExit(
                f"--retriever_gpus and vLLM GPUs overlap on {sorted(overlap)}. "
                f"Give the retriever its own GPUs (or pass --retriever_url to reuse "
                f"an external server).")

    retriever_url, retriever_proc = _resolve_retriever(
        args, needs_spawn_if_missing=True)
    try:
        if args.parallel:
            _run_parallel(args, runs, retriever_url)
            return

        # ----- Search cache -----
        cache_dir = './cache'
        os.makedirs(cache_dir, exist_ok=True)
        _cache_tag = args.retrieval_method
        if args.search_cache_suffix:
            _cache_tag = f'{_cache_tag}.{args.search_cache_suffix}'
        search_cache_path = os.path.join(cache_dir, f'search_cache.{_cache_tag}.json')
        search_cache = json.load(open(search_cache_path, encoding='utf-8')) \
            if os.path.exists(search_cache_path) else {}

        def save_search_cache():
            with open(search_cache_path, 'w', encoding='utf-8') as f:
                json.dump(search_cache, f, ensure_ascii=False, indent=2)

        retriever = RemoteRetriever(retriever_url, max_doc_len=args.max_doc_len)

        # ----- Tokenizer + main LLM (loaded ONCE, never unloaded) -----
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        vllm_gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
        print(f"Loading main LLM on GPUs {vllm_gpus}")
        t0 = time.time()
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=len(vllm_gpus),
            gpu_memory_utilization=0.90,
            dtype="half",
            max_model_len=args.max_model_len,
            max_num_seqs=32,
        )
        print(f"Main LLM loaded in {time.time() - t0:.1f}s")

        # ----- Run datasets sequentially -----
        summary = []
        for run in runs:
            t0 = time.time()
            try:
                run_one_dataset(run, args, retriever, tokenizer, llm,
                                search_cache, save_search_cache)
                status = "OK"
            except Exception as e:
                status = f"FAIL ({type(e).__name__}: {e})"
                print(f"[ERROR] {run['dataset_name']}: {status}")
            elapsed = time.time() - t0
            summary.append((run['dataset_name'], status, f"{elapsed:.0f}s"))
            print(f"\n>> {run['dataset_name']}: {status}  ({elapsed:.0f}s)\n")

        # ── Summary ────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("  SUMMARY (search_o1_wiki)")
        print("=" * 60)
        for name, status, elapsed in summary:
            print(f"  {name:12s}  {status:30s}  {elapsed}")
        print("=" * 60)
        print("Done.")
    finally:
        stop_retriever_server(retriever_proc)


if __name__ == "__main__":
    main()
