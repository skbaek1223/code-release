import os
# Visible GPUs are dedicated to vLLM. FAISS runs in a separate retriever
# server process (see retriever_server.py) on its own GPUs, accessed over
# HTTP via --retriever_url.
_ALL_GPUS = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
import gc
import re
import json
import time
import argparse
from typing import List, Dict, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import run_evaluation
from prompts import (
    get_single_qa_instruction,
    get_multi_qa_instruction,
    get_retrieval_evaluator_instruction,
    get_extractor_instruction,
)
from retriever_utils import RemoteRetriever

MULTI_HOP_DATASETS = {'hotpotqa', '2wiki', 'musique'}

# Named subsets of a base dataset: slice the full file and report metrics
# under the base dataset's eval key.
DATASET_SUBSETS = {
    "triviaqa_a": {"slice": (0, 4000),    "eval_name": "triviaqa"},
    "triviaqa_b": {"slice": (4000, None), "eval_name": "triviaqa"},
}

# Special tokens
BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"
EXTRACTED_MARKER = "**Extracted Information**"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Run Re-Guide step-aware retrieval reasoning.")

    # Data
    p.add_argument('--dataset_name', type=str, required=True,
                   choices=['nq', 'triviaqa', 'triviaqa_a', 'triviaqa_b',
                            'popqa', 'hotpotqa', '2wiki', 'musique', 'bamboogle', 'ambigqa'])
    p.add_argument('--split', type=str, default='test')
    p.add_argument('--qa_data_path', type=str, default='',
                   help="Path to original QA dataset JSON (provides Question/answer).")
    p.add_argument('--subset_num', type=int, default=-1)

    # Retrieval (remote retriever_server over HTTP)
    p.add_argument('--max_search_limit', type=int, default=10)
    p.add_argument('--max_turn', type=int, default=5)
    p.add_argument('--top_k', type=int, default=5)
    p.add_argument('--max_doc_len', type=int, default=3000)
    p.add_argument('--retriever_url', type=str, required=True,
                   help="Base URL of the running FAISS retriever server, e.g. http://127.0.0.1:8765")
    p.add_argument('--retrieval_method', type=str, default='e5',
                   choices=['e5', 'bm25'],
                   help="Tag used for the on-disk search cache filename; must match the server.")

    # Model / sampling (main reasoning LLM)
    p.add_argument('--model_path', type=str, default='/mnt/raid6/skbaek1223/models/QwQ-32B')
    p.add_argument('--steps_model_path', type=str,
                   default=os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'experiments', 'S7000', 'checkpoints', 'final_merged'),
                   help="Fine-tuned planner model that emits a numbered retrieval guide per question.")
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.8)
    p.add_argument('--top_k_sampling', type=int, default=20)
    p.add_argument('--repetition_penalty', type=float, default=None)
    p.add_argument('--max_model_len', type=int, default=20480,
                   help="vLLM context window size (prompt + generation combined).")
    p.add_argument('--max_new_tokens', type=int, default=4096,
                   help="Max new tokens per main decoding call.")

    # Output
    p.add_argument('--output_dir', type=str, default=None,
                   help="Custom output directory. If not set, defaults to ./outputs/{dataset}.{model}.re_guide")
    p.add_argument('--search_cache_suffix', type=str, default='',
                   help="Optional suffix for the per-process search cache file (avoids races in parallel mode).")

    # Retry control
    # Insufficient retry budget. If --budget_base is not set, defaults to
    # 60 for single-hop and 90 for multi-hop datasets.
    p.add_argument('--budget_base', type=float, default=None)
    p.add_argument('--budget_lambda', type=float, default=0.6)
    p.add_argument('--budget_r', type=float, default=2.0)

    # Ablation switches
    # --no_retrieval_guide: skip the planner/steps model. The user prompt is
    #   built without a "Retrieval Guide:" section and few-shot examples are
    #   swapped for ones that don't show one. Reasoning-guide / evaluation
    #   module remain active.
    # --no_reasoning_guide: skip the per-turn evaluator and the [Reasoning
    #   Guide] system messages. The extractor still runs; its extracted
    #   facts are injected as the search result in place of raw documents.
    #   Retrieval guide (planner) remains active.
    # --no_budget: keep the full reasoning-guide pipeline but omit the
    #   "use up to N words" budget hint from the Insufficient message.
    p.add_argument('--no_retrieval_guide', action='store_true',
                   help="Ablation: disable the planner/steps model.")
    p.add_argument('--no_reasoning_guide', action='store_true',
                   help="Ablation: disable the evaluator and the "
                        "[Reasoning Guide] system messages. Extractor "
                        "still runs.")
    p.add_argument('--no_budget', action='store_true',
                   help="Ablation: omit the word-budget hint ('use up to N words') "
                        "from the Insufficient [Reasoning Guide] message.")

    # Qwen3 thinking mode. When False (default), apply_chat_template receives
    # enable_thinking=False so Qwen3 skips the <think> block and responds
    # directly. Set --thinking_mode to enable internal chain-of-thought.
    # Non-Qwen3 tokenizers ignore the enable_thinking kwarg silently.
    p.add_argument('--thinking_mode', action='store_true',
                   help="Enable Qwen3 thinking mode (enable_thinking=True in "
                        "apply_chat_template). Default: off.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def extract_between(text: str, start_tag: str, end_tag: str) -> Optional[str]:
    pattern = re.escape(start_tag) + r"(.*?)" + re.escape(end_tag)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


def parse_evaluator_output(text: str):
    """Return (is_sufficient: bool, confidence: float in [0,1])."""
    ans = extract_between(text, "<answer>", "</answer>")
    conf = extract_between(text, "<confidence>", "</confidence>")

    label = (ans or "").strip().lower()
    if label == "sufficient":
        is_suff = True
    elif label == "insufficient":
        is_suff = False
    else:
        is_suff = False  # conservative fallback

    c = 0.5
    if conf:
        m = re.search(r'(\d+(?:\.\d+)?)', conf)
        if m:
            c = float(m.group(1)) / 100.0
            c = max(0.0, min(1.0, c))
    return is_suff, c


def compute_budget(c: float, base: float, lam: float, r: float) -> int:
    # Higher confidence -> larger retry budget
    return int(round(base * (lam + c)** r))


def has_complete_boxed(text: str) -> bool:
    """Check if text contains a complete \\boxed{...} (with balanced braces)."""
    match = re.search(r'\\boxed\{', text)
    if not match:
        return False
    depth = 1
    for i in range(match.end(), len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        if depth == 0:
            return True
    return False


def truncate_after_boxed(text: str) -> str:
    """Truncate text right after the first complete \\boxed{...}."""
    match = re.search(r'\\boxed\{', text)
    if not match:
        return text
    depth = 1
    for i in range(match.end(), len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        if depth == 0:
            return text[:i + 1]
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.repetition_penalty is None:
        args.repetition_penalty = 1.05 if 'qwq' in args.model_path.lower() else 1.0

    subset_meta = DATASET_SUBSETS.get(args.dataset_name, {})
    eval_name = subset_meta.get('eval_name', args.dataset_name)
    data_slice = subset_meta.get('slice')

    if args.budget_base is None:
        args.budget_base = 90.0 if eval_name in MULTI_HOP_DATASETS else 60.0

    # ----- Load QA -----
    with open(args.qa_data_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    filtered_data = list(qa_data)
    if data_slice is not None:
        start, end = data_slice
        filtered_data = filtered_data[start:end]
        print(f"  Subset:  {args.dataset_name} -> [{start}:{end}] -> {len(filtered_data)} items")
    if args.subset_num != -1:
        filtered_data = filtered_data[:args.subset_num]

    # ----- Search cache (query -> formatted docs) -----
    cache_dir = './cache'
    os.makedirs(cache_dir, exist_ok=True)
    _cache_tag = args.retrieval_method
    if args.search_cache_suffix:
        _cache_tag = f'{_cache_tag}.{args.search_cache_suffix}'
    search_cache_path = os.path.join(cache_dir, f'search_cache.{_cache_tag}.json')
    search_cache = json.load(open(search_cache_path, encoding='utf-8')) if os.path.exists(search_cache_path) else {}

    def save_caches():
        with open(search_cache_path, 'w', encoding='utf-8') as f:
            json.dump(search_cache, f, ensure_ascii=False, indent=2)

    # ----- Retriever client (remote FAISS server over HTTP) -----
    retriever = RemoteRetriever(args.retriever_url, max_doc_len=args.max_doc_len)

    # ----- Steps model: generate retrieval guide per question, then free -----
    retrieval_guides = ["" for _ in filtered_data]
    if args.no_retrieval_guide:
        args.steps_model_path = ""
    # Per-question planner token counts. OUTPUT = stripped retrieval guide;
    # INPUT = chat-templated system + user prompt the planner would receive.
    # Both stay at 0 under --no_retrieval_guide.
    planner_in_tokens_per_q = [0] * len(filtered_data)
    planner_out_tokens_per_q = [0] * len(filtered_data)
    steps_tok = None
    steps_system = ("You are an information retrieval planning expert. Given a question, "
                    "generate an ordered sequence of concrete retrieval steps required to "
                    "find the answer. Each step represents one retrieval action.\n\n"
                    "Write one step per line, numbered. No other text.")
    if args.steps_model_path:
        steps_cache_dir = os.path.join(cache_dir, 'steps')
        os.makedirs(steps_cache_dir, exist_ok=True)
        steps_model_tag = os.path.basename(args.steps_model_path.rstrip('/')) or 'steps'
        if args.search_cache_suffix:
            steps_model_tag = f'{steps_model_tag}.{args.search_cache_suffix}'
        steps_cache_path = os.path.join(steps_cache_dir, f'{steps_model_tag}.json')
        steps_cache = json.load(open(steps_cache_path, encoding='utf-8')) \
            if os.path.exists(steps_cache_path) else {}

        missing_idx = [i for i, item in enumerate(filtered_data)
                       if item['Question'] not in steps_cache]
        for i, item in enumerate(filtered_data):
            if item['Question'] in steps_cache:
                retrieval_guides[i] = steps_cache[item['Question']]
        print(f"[steps] cache hit {len(filtered_data) - len(missing_idx)}/{len(filtered_data)}")

        # Load planner tokenizer unconditionally (cheap, no GPU). Used both
        # for chat template (if we need to regenerate) and to count tokens
        # of cached guides in the planner's own vocabulary.
        steps_tok = AutoTokenizer.from_pretrained(args.steps_model_path, trust_remote_code=True)

        if missing_idx:
            # Steps model shares the same GPU pool as the main LLM. It is
            # destroyed before the main LLM loads (single transition, unlike
            # the old per-turn FAISS swap).
            print(f"Loading steps model: {args.steps_model_path}")
            steps_llm = LLM(
                model=args.steps_model_path,
                tensor_parallel_size=len(_ALL_GPUS),
                gpu_memory_utilization=0.90,
                dtype="half",
            )
            steps_prompts = [
                steps_tok.apply_chat_template(
                    [{"role": "system", "content": steps_system},
                     {"role": "user", "content": f"Question: {filtered_data[i]['Question']}"}],
                    tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
                for i in missing_idx
            ]
            steps_sp = SamplingParams(max_tokens=1024, temperature=0.0, top_p=1.0,
                                      repetition_penalty=1.1)
            print(f"[steps] generating retrieval guides for {len(steps_prompts)} questions")
            steps_outs = steps_llm.generate(steps_prompts, sampling_params=steps_sp)
            for i, out in zip(missing_idx, steps_outs):
                guide = re.sub(r"<think>.*?</think>\s*", "", out.outputs[0].text, flags=re.DOTALL).strip()
                retrieval_guides[i] = guide
                steps_cache[filtered_data[i]['Question']] = guide
            with open(steps_cache_path, 'w', encoding='utf-8') as f:
                json.dump(steps_cache, f, ensure_ascii=False, indent=2)
            print(f"[steps] cache saved to {steps_cache_path}")

            del steps_llm
            gc.collect()
            torch.cuda.empty_cache()
            try:
                from vllm.distributed.parallel_state import destroy_model_parallel
                destroy_model_parallel()
            except Exception:
                pass

        # Count planner tokens uniformly (same rule for fresh / cache-hit):
        #   output = stripped guide tokenized by the planner's tokenizer
        #   input  = chat-templated system + user prompt the planner would receive
        for i, item in enumerate(filtered_data):
            guide = retrieval_guides[i]
            if not guide:
                continue
            planner_out_tokens_per_q[i] = len(steps_tok(
                guide, add_special_tokens=False)['input_ids'])
            chat_prompt = steps_tok.apply_chat_template(
                [{"role": "system", "content": steps_system},
                 {"role": "user", "content": f"Question: {item['Question']}"}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            planner_in_tokens_per_q[i] = len(steps_tok(
                chat_prompt, add_special_tokens=False)['input_ids'])

    # ----- Model -----
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    MAX_MODEL_LEN = args.max_model_len
    MIN_OUTPUT_ROOM = 512

    # Main LLM: loaded ONCE on all visible GPUs and kept resident until the
    # process exits. FAISS lives in a separate server so no swapping needed.
    print(f"Loading main LLM on GPUs {_ALL_GPUS}")
    t0 = time.time()
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=len(_ALL_GPUS),
        gpu_memory_utilization=0.90,
        dtype="half",
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=256,
    )
    print(f"Main LLM loaded in {time.time() - t0:.1f}s")

    model_short = args.model_path.split('/')[-1].lower().replace('-instruct', '')
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = f'./outputs/{args.dataset_name}.{model_short}.re_guide'
    os.makedirs(output_dir, exist_ok=True)

    # ----- Build initial prompts -----
    GUIDE_MAX_TOKENS = 512
    input_list = []
    for item, guide in zip(filtered_data, retrieval_guides):
        question = item['Question']
        if eval_name in MULTI_HOP_DATASETS:
            instruction = get_multi_qa_instruction(
                no_reasoning_guide=args.no_reasoning_guide,
                no_retrieval_guide=args.no_retrieval_guide)
        else:
            instruction = get_single_qa_instruction(
                no_reasoning_guide=args.no_reasoning_guide,
                no_retrieval_guide=args.no_retrieval_guide)
            if not args.no_retrieval_guide:
                non_empty_lines = [l for l in guide.splitlines() if l.strip()]
                if len(non_empty_lines) == 1:
                    guide = re.sub(r"^\s*\d+\.\s*", "", guide).strip()
        if args.no_retrieval_guide:
            user_prompt = (
                f"Question:\n{question}\n\n"
                f"Begin reasoning, performing searches by writing search queries as needed."
            )
        else:
            guide_ids = tokenizer(guide, add_special_tokens=False)['input_ids']
            if len(guide_ids) > GUIDE_MAX_TOKENS:
                guide = tokenizer.decode(guide_ids[:GUIDE_MAX_TOKENS], skip_special_tokens=True)
            user_prompt = (
                f"Question:\n{question}\n\n"
                f"You may refer to the retrieval guide below to inform your search strategy, or feel free to take a different approach.\n\n"
                f"Retrieval Guide:\n{guide}\n\n"
                f"Begin reasoning, performing searches by writing search queries as needed."
            )
        msg = [{"role": "user", "content": instruction + "\n" + user_prompt}]
        prompt = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=args.thinking_mode)
        input_list.append(prompt)

    active_sequences = [{
        'item': item,
        'guide': guide,
        'base_prompt': prompt,
        'steps': [],                  # list of committed step strings
        'current_step': '',           # buffer for the step being built this turn
        'output': '',
        'finished': False,
        'search_count': 0,
        'executed_search_queries': set(),
        'last_model_chunk': '',
        'pinned_steps': set(),        # step indices that must always be kept in prompt
        'pin_next_step': False,       # flag: pin the next committed step
        'module_tokens': {            # per-module input/output token counts (this question)
            'planner':   {'input': planner_in,  'output': planner_out},
            'reasoning': {'input': 0,           'output': 0},
            'extractor': {'input': 0,           'output': 0},
            'evaluator': {'input': 0,           'output': 0},
        },
    } for item, guide, prompt, planner_in, planner_out in zip(
        filtered_data, retrieval_guides, input_list,
        planner_in_tokens_per_q, planner_out_tokens_per_q)]

    RECENT_K = 5
    IS_MULTI_HOP = eval_name in MULTI_HOP_DATASETS

    ELLIPSIS = "\n\n[... omitted intermediate reasoning steps ...]\n\n"

    def _compute_keep(seq: Dict, recent_k: int) -> set:
        steps = seq['steps']
        n = len(steps)
        keep = set()
        if n > 0:
            keep.add(0)
        # Always keep pinned steps (sufficient + intermediate answer steps)
        keep.update(seq['pinned_steps'])
        keep.update(range(max(0, n - recent_k), n))
        return keep

    def _assemble(seq: Dict, keep: set) -> str:
        steps = seq['steps']
        ordered = sorted(keep)
        parts = []
        prev = -1
        for i in ordered:
            if prev != -1 and i != prev + 1:
                parts.append(ELLIPSIS)
            parts.append(steps[i])
            prev = i
        return seq['base_prompt'] + ''.join(parts) + seq['current_step']

    def build_prompt(seq: Dict, max_prompt_tokens: int = None) -> str:
        for k in range(RECENT_K, -1, -1):
            keep = _compute_keep(seq, recent_k=k)
            prompt = _assemble(seq, keep)
            if max_prompt_tokens is None or count_tokens(prompt) <= max_prompt_tokens:
                if k < RECENT_K:
                    print(f"  [truncate] reduced recent_k {RECENT_K} -> {k} to fit budget")
                return prompt
        # Last resort: truncate current_step search results to fit
        return _truncate_to_fit(seq, max_prompt_tokens)

    def _truncate_to_fit(seq: Dict, max_prompt_tokens: int) -> str:
        keep = _compute_keep(seq, recent_k=0)
        base = seq['base_prompt']
        ordered = sorted(keep)
        parts = []
        prev = -1
        for i in ordered:
            if prev != -1 and i != prev + 1:
                parts.append(ELLIPSIS)
            parts.append(seq['steps'][i])
            prev = i
        history_text = ''.join(parts)
        cur = seq['current_step']
        # Measure how many tokens we need to cut from current_step
        overhead = count_tokens(base + history_text + cur) - max_prompt_tokens
        if overhead <= 0:
            return base + history_text + cur
        # Trim search-result blocks inside current_step first
        sr_start = cur.rfind(BEGIN_SEARCH_RESULT)
        sr_end = cur.rfind(END_SEARCH_RESULT)
        if sr_start != -1 and sr_end != -1 and sr_end > sr_start:
            sr_content = cur[sr_start + len(BEGIN_SEARCH_RESULT):sr_end]
            sr_ids = tokenizer(sr_content, add_special_tokens=False)['input_ids']
            keep_ids = sr_ids[:max(0, len(sr_ids) - overhead)]
            trimmed_sr = tokenizer.decode(keep_ids, skip_special_tokens=True)
            cur = cur[:sr_start + len(BEGIN_SEARCH_RESULT)] + trimmed_sr + cur[sr_end:]
            result = base + history_text + cur
            if count_tokens(result) <= max_prompt_tokens:
                print(f"  [truncate] trimmed search result by ~{overhead} tokens")
                return result
        # Final fallback: hard-truncate current_step from the left
        cur_ids = tokenizer(cur, add_special_tokens=False)['input_ids']
        avail = max_prompt_tokens - count_tokens(base + history_text)
        if avail > 0:
            cur = tokenizer.decode(cur_ids[-avail:], skip_special_tokens=True)
            print(f"  [truncate] hard-truncated current_step to {avail} tokens")
        else:
            cur = ''
            print(f"  [truncate] dropped current_step entirely")
        return base + history_text + cur

    def commit_step(seq: Dict):
        if not seq['current_step']:
            return
        idx = len(seq['steps'])
        seq['steps'].append(seq['current_step'])
        seq['current_step'] = ''
        # Pin this step if flagged by a previous sufficient evaluation
        if seq['pin_next_step']:
            seq['pinned_steps'].add(idx)
            seq['pin_next_step'] = False

    # ----- Sampling params -----
    MAIN_MAX_NEW_TOKENS = args.max_new_tokens

    def make_main_sampling(max_new: int) -> SamplingParams:
        return SamplingParams(
            max_tokens=max_new,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k_sampling,
            repetition_penalty=args.repetition_penalty,
            stop=[END_SEARCH_QUERY],
            include_stop_str_in_output=True,
        )

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)['input_ids'])
    short_sampling = SamplingParams(
        max_tokens=4096,
        temperature=0.0,
        top_p=1.0,
        stop=["</confidence>"],
        include_stop_str_in_output=True,
    )

    def llm_chat_batch(user_prompts: List[str], sp: SamplingParams):
        """Returns parallel lists (texts, in_tok_counts, out_tok_counts)."""
        prompts = [
            tokenizer.apply_chat_template([{"role": "user", "content": up}],
                                          tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
            for up in user_prompts
        ]
        outs = llm.generate(prompts, sampling_params=sp)
        texts = [o.outputs[0].text for o in outs]
        in_toks = [len(o.prompt_token_ids) for o in outs]
        out_toks = [len(o.outputs[0].token_ids) for o in outs]
        return texts, in_toks, out_toks

    # ----- Main loop -----
    batch_output_records = []
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

        # 1) Generate next chunk of reasoning, stop at END_SEARCH_QUERY.
        safe_pending, safe_sps, safe_prompts = [], [], []
        budget = MAX_MODEL_LEN - MIN_OUTPUT_ROOM
        for s in pending:
            if len(s['output']) > 40_000:  # char-based runaway cap (~6.4K tokens)
                s['finished'] = True
                continue
            built = build_prompt(s, max_prompt_tokens=budget)
            prompt_len = count_tokens(built)
            room = MAX_MODEL_LEN - prompt_len
            if room < MIN_OUTPUT_ROOM:
                # base_prompt alone exceeds the budget- cannot recover
                msg = (f"\n{BEGIN_SEARCH_RESULT}\nContext limit reached "
                       f"(prompt {prompt_len} tokens). Terminating.\n{END_SEARCH_RESULT}\n")
                s['current_step'] += msg
                s['output'] += msg

                commit_step(s)
                s['finished'] = True
                continue
            safe_pending.append(s)
            safe_sps.append(make_main_sampling(min(MAIN_MAX_NEW_TOKENS, room)))
            safe_prompts.append(built)
        if not safe_pending:
            continue
        print(f"[reasoning] generating next reasoning chunk for {len(safe_pending)} sequences ")
        outputs = llm.generate(safe_prompts, sampling_params=safe_sps)

        # Pass 1: parse outputs, decide per-seq fate, collect queries needing retrieval.
        retrieval_candidates = []  # (seq, query) pairs that will need FAISS
        for seq, out in zip(safe_pending, outputs):
            text = out.outputs[0].text
            seq['module_tokens']['reasoning']['input']  += len(out.prompt_token_ids)
            seq['module_tokens']['reasoning']['output'] += len(out.outputs[0].token_ids)

            seq['last_model_chunk'] = text
            seq['current_step'] = text
            seq['output'] += text

            query = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
            if not (query and text.rstrip().endswith(END_SEARCH_QUERY)):
                if has_complete_boxed(text):
                    truncated = truncate_after_boxed(text)
                    seq['output'] = seq['output'][:len(seq['output']) - len(text)] + truncated
                    commit_step(seq)
                    seq['finished'] = True
                else:
                    commit_step(seq)
                continue

            if turn == args.max_turn:
                commit_step(seq)
                seq['finished'] = True
                continue

            if seq['search_count'] >= args.max_search_limit:
                msg = f"\n{BEGIN_SEARCH_RESULT}\nSearch limit reached. Terminating.\n{END_SEARCH_RESULT}\n"
                seq['current_step'] += msg
                seq['output'] += msg
                commit_step(seq)
                seq['finished'] = True
                continue
            if query in seq['executed_search_queries']:
                if args.no_reasoning_guide:
                    msg = (f"\n{BEGIN_SEARCH_RESULT}\nQuery already searched. "
                           f"Refer to previous results.\n{END_SEARCH_RESULT}\n")
                else:
                    msg = (f"\n{BEGIN_SEARCH_RESULT}\nQuery already searched. Refer to previous results.\n{END_SEARCH_RESULT}\n"
                           f"[Reasoning Guide]: This query was already searched. Try a different search query to find new information.\n")
                seq['current_step'] += msg
                seq['output'] += msg
                commit_step(seq)
                continue

            retrieval_candidates.append((seq, query))

        # Pass 2: fetch fresh retrievals from the remote retriever server.
        fresh_queries = [q for _, q in retrieval_candidates if q not in search_cache]
        unique_fresh = list(dict.fromkeys(fresh_queries))  # dedupe, preserve order
        if unique_fresh:
            print(f"[retrieval] running {len(unique_fresh)} fresh queries via {retriever.url}")
            try:
                batch_docs = retriever.batch_search(unique_fresh, args.top_k)
            except Exception as e:
                print(f"Retrieval batch error: {e}")
                batch_docs = [[] for _ in unique_fresh]
            for q, docs in zip(unique_fresh, batch_docs):
                search_cache[q] = docs

        # Pass 3: build retrieval-dependent batch from (now fully cached) docs.
        batch_seqs, batch_queries, batch_documents = [], [], []
        for seq, query in retrieval_candidates:
            docs = search_cache.get(query, [])
            formatted = ""
            for i, d in enumerate(docs):
                formatted += f"**Doc {i+1}:**\n{json.dumps(d, ensure_ascii=False, indent=2)}\n"
            seq['search_count'] += 1
            seq['executed_search_queries'].add(query)
            batch_seqs.append(seq)
            batch_queries.append(query)
            batch_documents.append(formatted)

        if not batch_seqs:
            continue

        # 4) Extractor (fact extraction grounded in recent reasoning).
        extractor_prompts = []
        for seq, q, docs in zip(batch_seqs, batch_queries, batch_documents):
            last_chunk = seq.get('last_model_chunk', '')
            recent_reasoning = last_chunk.split(BEGIN_SEARCH_QUERY)[0].strip()
            if not recent_reasoning:
                recent_reasoning = seq['item']['Question']
            extractor_prompts.append(get_extractor_instruction(
                question=seq['item']['Question'],
                recent_reasoning=recent_reasoning,
                search_query=q,
                documents=docs,
            ))
        print(f"[extraction] extracting query-relevant facts from retrieved docs "
              f"for {len(extractor_prompts)} queries")
        extractor_raw, extractor_in_toks, extractor_out_toks = llm_chat_batch(
            extractor_prompts, short_sampling)
        for seq, n_in, n_out in zip(batch_seqs, extractor_in_toks, extractor_out_toks):
            seq['module_tokens']['extractor']['input']  += n_in
            seq['module_tokens']['extractor']['output'] += n_out
        extracted_facts = []
        for raw in extractor_raw:
            if EXTRACTED_MARKER in raw:
                f = raw.split(EXTRACTED_MARKER)[-1].strip().strip("`").strip()
                extracted_facts.append(f if f else "NONE")
            else:
                extracted_facts.append("NONE")

        # 5) Evaluator (sufficient / insufficient + confidence).
        # Skipped entirely under --no_reasoning_guide ablation (extractor still
        # runs above; only the evaluator + [Reasoning Guide] system message
        # are removed). Otherwise: skip the evaluator for NONE extractions and
        # treat them as insufficient with 100% confidence so the retry budget
        # is at its maximum for the next turn.
        if args.no_reasoning_guide:
            evaluator_raw = [None] * len(batch_seqs)
        else:
            evaluator_prompts = []
            eval_indices = []
            for idx, (seq, q, fact) in enumerate(zip(batch_seqs, batch_queries, extracted_facts)):
                if fact == "NONE":
                    continue
                last_chunk = seq.get('last_model_chunk', '')
                recent_reasoning = last_chunk.split(BEGIN_SEARCH_QUERY)[0].strip()
                if not recent_reasoning:
                    recent_reasoning = seq['item']['Question']
                evaluator_prompts.append(get_retrieval_evaluator_instruction(
                    question=seq['item']['Question'],
                    recent_reasoning=recent_reasoning,
                    search_query=q,
                    search_result=fact,
                ))
                eval_indices.append(idx)
            skipped = len(batch_seqs) - len(evaluator_prompts)
            if skipped:
                print(f"[evaluation] skipping evaluator for {skipped} NONE extractions "
                      f"(treated as insufficient, confidence 100%)")
            print(f"[evaluation] judging sufficiency/confidence of extracted facts "
                  f"on {len(evaluator_prompts)} sequences")
            if evaluator_prompts:
                evaluator_raw_partial, evaluator_in_toks, evaluator_out_toks = llm_chat_batch(
                    evaluator_prompts, short_sampling)
            else:
                evaluator_raw_partial, evaluator_in_toks, evaluator_out_toks = [], [], []
            evaluator_raw = [None] * len(batch_seqs)
            for idx, raw in zip(eval_indices, evaluator_raw_partial):
                evaluator_raw[idx] = raw
            for idx, n_in, n_out in zip(eval_indices, evaluator_in_toks, evaluator_out_toks):
                batch_seqs[idx]['module_tokens']['evaluator']['input']  += n_in
                batch_seqs[idx]['module_tokens']['evaluator']['output'] += n_out

        # 6) Inject result back into each sequence based on evaluator decision.
        # Under --no_reasoning_guide, ev_raw is None for every seq; we record
        # the extracted fact and inject the search-result block without any
        # [Reasoning Guide] follow-up message.
        for seq, q, fact, ev_raw in zip(
                batch_seqs, batch_queries, extracted_facts, evaluator_raw):
            if args.no_reasoning_guide:
                is_suff, conf = None, None
                ev_raw = "[skipped: --no_reasoning_guide ablation]"
            elif fact == "NONE":
                is_suff, conf = False, 1.0
                ev_raw = "[skipped: NONE extraction -> insufficient, confidence 100%]"
            else:
                is_suff, conf = parse_evaluator_output(ev_raw)

            batch_output_records.append({
                'id': seq['item'].get('id'),
                'turn': turn,
                'query': q,
                'extracted': fact,
                'evaluator_raw': ev_raw,
                'sufficient': is_suff,
                'confidence': conf,
            })

            if args.no_reasoning_guide:
                msg = f"\n{BEGIN_SEARCH_RESULT}\n{fact}\n{END_SEARCH_RESULT}\n"
                seq['current_step'] += msg
                seq['output'] += msg
                commit_step(seq)
                continue

            if is_suff:
                if IS_MULTI_HOP:
                    # Pin this step (sufficient result) and the next step
                    # (intermediate answer + next query) so they survive truncation
                    seq['pinned_steps'].add(len(seq['steps']))
                    seq['pin_next_step'] = True
                    eval_guide = (f"Sufficient.\n"
                                  f"1) If there is still more information to retrieve before fully answering the original question \"{seq['item']['Question']}\", "
                                  f"derive an intermediate answer based on the retrieved information, then continue reasoning toward the next retrieval. "
                                  f"You may refer to the reasoning context above or the retrieval guide to inform your search strategy, or feel free to take a different approach.\n"
                                  f"2) If you have sufficient information to fully answer the original question, provide the final answer in the format \\boxed{{YOUR_ANSWER}}.")
                else:
                    eval_guide = (f"Sufficient.\n"
                                  f"You now have enough information to answer the original question \"{seq['item']['Question']}\". Provide the final answer in the format \\boxed{{YOUR_ANSWER}}.")

            else:
                if args.no_budget:
                    eval_guide = (f"Insufficient (confidence {conf:.0%}).\n "
                                  f"Feel free to explore alternative paths, such as trying a different search query or taking other retrieval steps as needed, to derive an answer to the question: {seq['item']['Question']}. "
                                  f"Please think carefully.")
                else:
                    budget = compute_budget(conf, args.budget_base,
                                            args.budget_lambda, args.budget_r)
                    eval_guide = (f"Insufficient (confidence {conf:.0%}).\n "
                                  f"Feel free to explore alternative paths, such as trying a different search query or taking other retrieval steps as needed, to derive an answer to the question: {seq['item']['Question']}. "
                                  f"Please think carefully, and use up to {budget} words.")

            msg = (f"\n{BEGIN_SEARCH_RESULT}\n{fact}\n{END_SEARCH_RESULT}\n"
                   f"[Reasoning Guide]: {eval_guide}\n")
            seq['current_step'] += msg
            seq['output'] += msg
            commit_step(seq)

    total_time = time.time() - start_time

    output_list = [seq['output'] for seq in active_sequences]
    per_q_module_tokens = {
        mod: {
            'input':  [s['module_tokens'][mod]['input']  for s in active_sequences],
            'output': [s['module_tokens'][mod]['output'] for s in active_sequences],
        }
        for mod in ('planner', 'reasoning', 'extractor', 'evaluator')
    }
    run_evaluation(filtered_data, input_list, output_list,
                   eval_name, output_dir, total_time, args.split,
                   tokenizer=tokenizer,
                   per_q_module_tokens=per_q_module_tokens)

    save_caches()
    print("Done.")


if __name__ == "__main__":
    main()
