"""
Usage:
  # Step 1 — PREFETCH ONLY: scan each dataset's saved Outputs for search
  # queries, check the on-disk caches (search_cache.<method>[.<dataset>].json),
  # and for any missing queries spin up the retriever, fetch the top-k docs,
  # and persist them to the per-dataset caches. No vLLM is loaded. Re-run this
  # right before Step 2 if there's any chance the result files or caches
  # changed since the last prefetch — it's a no-op when everything is already
  # cached, and otherwise repopulates whatever is missing.
  python add_search_o1_wiki_infogen_cost.py \
      --prefetch_only \
      --datasets nq,ambigqa,hotpotqa,musique \
      --retriever_gpus "0,1" \
      --model_path /mnt/raid6/skbaek1223/models/QwQ-32B

  # Step 2 — vLLM ONLY, parallel work-queue (after prefetch has populated the
  # per-dataset caches): retriever is torn down, so its GPUs are free.
  # --gpu_groups defines a pool of GPU groups; each group runs one dataset
  # at a time, and as soon as a group's worker finishes the next queued
  # dataset is dispatched immediately (no batch-level barriers, so a fast
  # dataset finishing won't leave its GPUs idle). --no_retriever aborts on
  # any cache miss so silent misses can't slip through.
  python add_search_o1_wiki_infogen_cost.py \
      --no_retriever --parallel --gpu_groups "0,1;2,3" \
      --datasets nq,ambigqa,hotpotqa,musique \
      --model_path /mnt/raid6/skbaek1223/models/QwQ-32B
"""
import os
import sys


def _early_set_cuda_visible_devices():
    """CUDA_VISIBLE_DEVICES must be set before torch / vLLM are imported,
    otherwise tensor-parallel workers crash on a stale device probe.

    In --parallel mode the parent process never loads vLLM (it only runs the
    retriever phase, which has its own GPUs), so do not constrain CUDA there
    — workers receive their own --gpus via subprocess env."""
    argv = sys.argv[1:]
    if "--parallel" in argv:
        return

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
    if gpus_val:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpus_val


_DRY_RUN_FLAG = ("--dry_run" in sys.argv) or ("--dry-run" in sys.argv)
if not _DRY_RUN_FLAG:
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
from typing import Dict, List, Optional, Tuple

# vLLM / transformers / run_search_o1_wiki are imported lazily inside main()
# so --dry_run never triggers a CUDA probe or model download.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import _write_cost_report

# Search-o1 special tokens (kept inline so dry_run stays import-free of vLLM).
BEGIN_SEARCH_QUERY  = "<|begin_search_query|>"
END_SEARCH_QUERY    = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT   = "<|end_search_result|>"


def truncate_prev_reasoning_for_infogen(seq_output: str) -> str:
    """Verbatim copy of the inline truncation block in
    `run_search_o1_wiki.run_one_dataset` (lines following the
    'Truncate prev reasoning (match Search-o1 behavior)' comment).
    DO NOT modify — this must stay byte-identical to the live run path
    so backfilled `prev_reasoning` matches what was sent originally."""
    all_reasoning_steps = seq_output.replace('\n\n', '\n').split("\n")
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
    return truncated_prev_reasoning


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "outputs", "final_results")
DEFAULT_CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# (dataset, default split label used to glob the result file)
DATASET_SPLIT = {
    "nq": "test",
    "ambigqa": "dev",
    "hotpotqa": "dev",
    "musique": "dev",
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", default="nq,ambigqa,hotpotqa,musique",
                   help="Comma-separated dataset names.")
    p.add_argument("--model_path", default="/mnt/raid6/skbaek1223/models/QwQ-32B",
                   help="Model used by the run (and reused here for infogen).")
    p.add_argument("--model_short", default=None,
                   help="Override the model-short string used in the output dir name. "
                        "Defaults to basename(model_path).lower().replace('-instruct','').")
    p.add_argument("--output_base_dir", default=DEFAULT_OUTPUT_BASE_DIR,
                   help="Where the per-dataset search_o1_wiki dirs live.")
    p.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR,
                   help="Directory holding search_cache.<method>[.<suffix>].json files.")
    p.add_argument("--retrieval_method", default="e5", choices=["e5", "bm25"],
                   help="Cache filename tag — must match the original run.")
    p.add_argument("--gpus", default=None,
                   help="GPUs to use for vLLM (sets CUDA_VISIBLE_DEVICES).")
    p.add_argument("--max_model_len", type=int, default=40960)
    p.add_argument("--max_new_tokens", type=int, default=16384)
    p.add_argument("--max_search_limit", type=int, default=20,
                   help="Mirror of the value used at run time (skip queries past this index).")
    p.add_argument("--top_k", type=int, default=5,
                   help="Top-k passages per query (must match the original run).")
    p.add_argument("--max_doc_len", type=int, default=3000,
                   help="Per-passage truncation length (must match the original run).")
    p.add_argument("--batch_size", type=int, default=512,
                   help="Max prompts handed to vLLM at once.")
    p.add_argument("--force", action="store_true",
                   help="Re-run even if cost.json already exists.")
    p.add_argument("--dry_run", action="store_true",
                   help="Plan + count infogen calls per dataset, but do not load vLLM "
                        "and do not contact the retriever server.")

    # Retriever fallback (only used when a query is missing from the on-disk
    # search caches — e2 retrieval is deterministic, so re-querying recovers
    # the exact same docs the original run saw).
    p.add_argument("--retriever_url", default=None,
                   help="Base URL of an already-running retriever_server. "
                        "If omitted and --retriever_gpus is given, a server is auto-spawned.")
    p.add_argument("--retriever_gpus", default="0,1",
                   help="GPUs for the auto-spawned retriever server (only used when "
                        "--retriever_url is not provided and there are missing-doc queries).")
    p.add_argument("--retriever_host", default="127.0.0.1")
    p.add_argument("--retriever_port", type=int, default=8765)
    p.add_argument("--retriever_startup_timeout", type=int, default=900)
    p.add_argument("--retriever_save_cache", action="store_true",
                   help="Write back any newly fetched docs into "
                        "search_cache.<method>.<dataset>.json so subsequent runs "
                        "can reuse them without re-querying.")

    # Parallel mode (parent-only): retriever runs once with all 4 datasets'
    # missing queries, persists results to disk caches, then is torn down so
    # its GPUs can be reused. Worker subprocesses each handle one dataset
    # with their own vLLM on a disjoint GPU group.
    p.add_argument("--parallel", action="store_true",
                   help="Run datasets concurrently across --gpu_groups. The "
                        "parent does the retriever phase once (saving cache to "
                        "disk), then spawns one worker per dataset.")
    p.add_argument("--gpu_groups", default="0,1;2,3",
                   help="Semicolon-separated vLLM GPU groups for parallel mode. "
                        "Example: '0,1;2,3' runs two datasets at a time, two "
                        "GPUs each. Total concurrent GPU usage = sum of one "
                        "round of groups (retriever is already shut down).")
    p.add_argument("--no_retriever", action="store_true",
                   help="Worker mode: do not contact the retriever; if any "
                        "query is missing from the on-disk caches, raise.")
    p.add_argument("--prefetch_only", action="store_true",
                   help="Run only the retriever phase (fetch missing docs and "
                        "persist them to the per-dataset disk caches), then "
                        "exit before loading vLLM. Useful for warming the "
                        "cache while a retriever server is already up on its "
                        "own GPUs, so the vLLM pass can be run later on those "
                        "GPUs without contending with the retriever.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
_SQ_RE = re.compile(
    re.escape(BEGIN_SEARCH_QUERY) + r"(.*?)" + re.escape(END_SEARCH_QUERY),
    re.DOTALL,
)
_RESULT_HEAD_RE = re.compile(re.escape(BEGIN_SEARCH_RESULT))


def _was_system_injected(result_content: str) -> bool:
    """True if the search-result block was a system-side injection
    (max-search-limit / duplicate-query message), in which case infogen
    did NOT run for the preceding query."""
    low = result_content.lower()
    return ("maximum search limit" in low
            or "searched this query" in low)


def extract_infogen_calls(output_text: str, max_search_limit: int
                          ) -> List[Tuple[str, str]]:
    """Walk the saved Output for one item and return a list of
    (prev_reasoning_truncated, search_query) pairs in the order infogen
    would have been invoked at run time."""
    calls: List[Tuple[str, str]] = []
    executed: set = set()
    search_count = 0

    for m in _SQ_RE.finditer(output_text):
        end_idx = m.end()
        query = m.group(1).strip()
        if not query:
            continue

        # Find what (if anything) follows this query.
        after = output_text[end_idx:]
        head = _RESULT_HEAD_RE.search(after)
        if not head:
            # Last query of the run had no result emitted -> never reached infogen.
            break

        result_open = end_idx + head.end()
        result_close_rel = after.find(END_SEARCH_RESULT, head.end())
        result_content = (after[head.end():result_close_rel]
                          if result_close_rel != -1 else after[head.end():])

        skip_due_to_inject = _was_system_injected(result_content)
        skip_due_to_dup = query in executed
        skip_due_to_limit = search_count >= max_search_limit

        if not (skip_due_to_inject or skip_due_to_dup or skip_due_to_limit):
            # `output_text[:end_idx]` is the prefix of the saved Output up
            # through (and including) this query's <|end_search_query|> —
            # exactly what `seq['output']` held at run time when infogen was
            # invoked for this query. Pass it through the verbatim copy of
            # the original truncation rule.
            prev_reasoning = truncate_prev_reasoning_for_infogen(
                output_text[:end_idx])
            calls.append((prev_reasoning, query))

        executed.add(query)
        search_count += 1

        if result_close_rel == -1:
            break  # malformed tail -> stop

    return calls


# ---------------------------------------------------------------------------
# Search cache loader
# ---------------------------------------------------------------------------
def load_search_caches(cache_dir: str, retrieval_method: str,
                       dataset_name: str) -> List[Dict]:
    """Return a priority-ordered list of cache dicts to consult.
    Dataset-suffixed cache wins (parallel runs wrote there); the global cache
    is the fallback (sequential / qwq runs wrote there)."""
    caches = []
    candidates = [
        f"search_cache.{retrieval_method}.{dataset_name}.json",
        f"search_cache.{retrieval_method}.json",
    ]
    for fname in candidates:
        path = os.path.join(cache_dir, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                caches.append(json.load(f))
            print(f"  [cache] loaded {path} ({len(caches[-1])} keys)")
    return caches


def lookup_docs(caches: List[Dict], query: str) -> Optional[List[Dict]]:
    for c in caches:
        if query in c:
            return c[query]
    return None


def format_docs(docs: List[Dict]) -> str:
    formatted = ""
    for i, d in enumerate(docs):
        formatted += f"**Passage {i + 1}:**\n{json.dumps(d, ensure_ascii=False, indent=2)}\n"
    return formatted


# ---------------------------------------------------------------------------
# Per-dataset processing
# ---------------------------------------------------------------------------
def find_run_files(output_dir: str, split: str
                   ) -> Tuple[Optional[str], Optional[str], str]:
    """Pick the most recently modified result JSON in `output_dir`, plus the
    metrics file to read existing aggregates from, plus the canonical
    sidecar path to (re)write metrics back to.

    Returns (result_path, metrics_read_path, metrics_write_path).

    Layouts accepted:
      - timestamped:     {split}.M.D,H:MM.json + {split}.M.D,H:MM.metrics.json
      - bare:            {split}.json + (optional) {split}.metrics.json
      - bare + 'metrics' (no extension) sidecar — found in some legacy dirs

    `metrics_write_path` is ALWAYS `<result_base>.metrics.json` so that
    `_write_cost_report` can derive `<result_base>.cost.json/.cost.txt`
    correctly. `metrics_read_path` may be None (no source to seed from)."""
    candidates = []
    candidates.extend(
        p for p in glob.glob(os.path.join(output_dir, f"{split}.*.json"))
        if not p.endswith(".metrics.json")
        and not p.endswith(".cost.json")
    )
    bare = os.path.join(output_dir, f"{split}.json")
    if os.path.exists(bare) and bare not in candidates:
        candidates.append(bare)
    if not candidates:
        return None, None, ""
    candidates.sort(key=os.path.getmtime)
    result_path = candidates[-1]
    metrics_write_path = result_path.replace(".json", ".metrics.json")

    metrics_read_path = None
    sidecar = result_path.replace(".json", ".metrics.json")
    if os.path.exists(sidecar):
        metrics_read_path = sidecar
    else:
        bare_metrics = os.path.join(output_dir, "metrics")
        if os.path.isfile(bare_metrics):
            metrics_read_path = bare_metrics
    return result_path, metrics_read_path, metrics_write_path


def synthesize_metrics_from_items(result_path: str) -> Dict:
    """Aggregate per-item `Metrics` from the result JSON into a `{overall: ...}`
    dict suitable for use as a metrics file body. Only fills the fields needed
    downstream (em/acc/f1/num_valid_answer); `query_latency` is set to 'N/A'
    and `avg_output_tokens` is omitted (cost-report consumers treat those as
    optional)."""
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n = len(data)
    if n == 0:
        return {"overall": {"num_valid_answer": "0 of 0"}}
    em = [it.get("Metrics", {}).get("em", 0) for it in data]
    acc = [it.get("Metrics", {}).get("acc", 0) for it in data]
    f1 = [it.get("Metrics", {}).get("f1", 0) for it in data]
    math_eq = [it.get("Metrics", {}).get("math_equal", 0) for it in data]
    valid = sum(1 for it in data if it.get("Metrics", {}).get("is_valid_answer"))
    return {"overall": {
        "em": sum(em) / n,
        "acc": sum(acc) / n,
        "f1": sum(f1) / n,
        "math_equal": sum(math_eq) / n,
        "num_valid_answer": f"{valid} of {n}",
        "query_latency": "N/A",
    }}


def gather_calls_for_dataset(result_path: str, max_search_limit: int,
                             caches: List[Dict]
                             ) -> Tuple[List[Dict], int, int, int, List[str]]:
    """Read the result JSON, parse all items' Outputs, return:
       calls: [{'item_idx': int, 'prev_reasoning': str, 'query': str,
                'docs': [...] | None}],
       n_items, n_calls_total, n_calls_missing_docs, missing_query_list

    Calls whose docs were not in the cache are still appended (with docs=None)
    so the caller can fill them in via the retriever before building prompts."""
    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    n_items = len(data)
    calls: List[Dict] = []
    total = 0
    missing = 0
    missing_queries: List[str] = []
    for idx, item in enumerate(data):
        out_text = item.get("Output", "") or ""
        for prev_reasoning, query in extract_infogen_calls(out_text, max_search_limit):
            total += 1
            docs = lookup_docs(caches, query)
            if docs is None:
                missing += 1
                missing_queries.append(query)
            calls.append({
                "item_idx": idx,
                "prev_reasoning": prev_reasoning,
                "query": query,
                "docs": docs,
            })
    return calls, n_items, total, missing, missing_queries


def aggregate_per_question(n_items: int, calls: List[Dict],
                           in_toks: List[int], out_toks: List[int]
                           ) -> Tuple[List[int], List[int]]:
    per_in = [0] * n_items
    per_out = [0] * n_items
    for c, n_in, n_out in zip(calls, in_toks, out_toks):
        per_in[c["item_idx"]] += n_in
        per_out[c["item_idx"]] += n_out
    return per_in, per_out


def update_metrics_and_cost(metrics_read_path: Optional[str],
                            metrics_write_path: str,
                            result_path: str,
                            dataset_name: str,
                            per_in: List[int], per_out: List[int],
                            avg_main_agent_tokens_str: Optional[str]):
    """Inject module_tokens into the metrics file and (re)write the
    cost.json / cost.txt sidecars.

    If `metrics_read_path` is None, the metrics body is synthesized on the
    fly by aggregating per-item `Metrics` from `result_path` (used for runs
    that were saved without a metrics sidecar)."""
    if metrics_read_path and os.path.exists(metrics_read_path):
        with open(metrics_read_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = synthesize_metrics_from_items(result_path)

    overall = metrics.get("overall", metrics)

    # reasoning.output is already represented via avg_output_tokens (saved at run
    # time using evaluate._per_q_output_tokens). We re-expose it under
    # module_tokens for cost-report consumers; reasoning.input is unknown for
    # legacy runs and is omitted (treated as 0 by _summarize_module_tokens-style
    # consumers).
    reasoning_entry = {}
    if avg_main_agent_tokens_str is not None:
        try:
            reasoning_entry["output"] = float(avg_main_agent_tokens_str)
        except ValueError:
            pass

    n = len(per_in)
    infogen_in_avg = (sum(per_in) / n) if n else 0.0
    infogen_out_avg = (sum(per_out) / n) if n else 0.0
    infogen_entry = {"input": infogen_in_avg, "output": infogen_out_avg}

    total_in = reasoning_entry.get("input", 0.0) + infogen_entry["input"]
    total_out = reasoning_entry.get("output", 0.0) + infogen_entry["output"]
    module_tokens = {
        "reasoning": reasoning_entry,
        "infogen": infogen_entry,
        "total": {
            "input": total_in,
            "output": total_out,
            "combined": total_in + total_out,
        },
        "_num_questions": n,
    }
    overall["module_tokens"] = module_tokens
    if "overall" in metrics:
        metrics["overall"] = overall
    else:
        metrics = {"overall": overall}

    with open(metrics_write_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    output_dir = os.path.dirname(metrics_write_path)
    metrics_name = os.path.basename(metrics_write_path)
    _write_cost_report(output_dir, metrics_name, dataset_name, overall)


# ---------------------------------------------------------------------------
# Phase helpers
# ---------------------------------------------------------------------------
def build_plan(args, datasets: List[str], model_short: str) -> List[Dict]:
    """Discover (dataset -> result/metrics paths) entries to backfill.
    Skips ones that have no result file or already have cost.json
    (unless --force). If no metrics file exists, one will be synthesized
    later from per-item Metrics."""
    plan = []
    for dataset in datasets:
        split = DATASET_SPLIT[dataset]
        output_dir = os.path.join(
            args.output_base_dir, f"{dataset}.{model_short}.search_o1_wiki")
        result_path, metrics_read_path, metrics_write_path = find_run_files(
            output_dir, split)
        if not result_path:
            print(f"[skip] {dataset}: no result file in {output_dir}")
            continue
        cost_json = result_path.replace(".json", ".cost.json")
        if os.path.exists(cost_json) and not args.force:
            print(f"[skip] {dataset}: cost already present at {cost_json} "
                  f"(re-run with --force to recompute)")
            continue
        plan.append({
            "dataset": dataset,
            "split": split,
            "output_dir": output_dir,
            "result_path": result_path,
            "metrics_read_path": metrics_read_path,
            "metrics_write_path": metrics_write_path,
        })
    return plan


def gather_phase(args, plan: List[Dict]) -> None:
    """For each plan entry, parse Outputs and look up docs in disk caches.
    Mutates each entry to add 'calls', 'n_items', 'missing_queries'."""
    for entry in plan:
        print(f"\n=== {entry['dataset']} ===")
        print(f"  result:        {entry['result_path']}")
        print(f"  metrics read:  "
              f"{entry['metrics_read_path'] or '(none, will synthesize from per-item Metrics)'}")
        print(f"  metrics write: {entry['metrics_write_path']}")
        caches = load_search_caches(args.cache_dir, args.retrieval_method,
                                    entry["dataset"])
        calls, n_items, n_total, n_missing, missing_queries = (
            gather_calls_for_dataset(
                entry["result_path"], args.max_search_limit, caches))
        entry["calls"] = calls
        entry["n_items"] = n_items
        entry["missing_queries"] = missing_queries
        print(f"  items:   {n_items}")
        print(f"  infogen calls parsed: {n_total} "
              f"(cached: {n_total - n_missing}, missing -> retriever: {n_missing})")


def retriever_phase(args, plan: List[Dict]):
    """Spin up retriever (if needed), batch_search ALL unique missing queries
    once across the entire plan, write back to disk, and tear the retriever
    back down so its GPUs are released. Returns (proc_to_kill_or_None,
    spawned_url_or_None) — proc is None when nothing was spawned (no missing
    queries OR external --retriever_url)."""
    from retriever_utils import (
        RemoteRetriever, ensure_retriever_server, stop_retriever_server,
    )
    all_missing_unique = sorted({q for e in plan for q in e["missing_queries"]})
    if not all_missing_unique:
        print("\n[retriever] all queries already cached — skipping retriever phase")
        return

    print(f"\n[retriever] need {len(all_missing_unique)} unique queries "
          f"not present in cache; spinning up retriever if not already running")
    retriever_url = args.retriever_url
    retriever_proc = None
    try:
        if not retriever_url:
            server_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "retriever_server.py")
            log_path = os.path.join(args.output_base_dir, "retriever_server.log")
            retriever_url, retriever_proc = ensure_retriever_server(
                retriever_gpus=args.retriever_gpus,
                host=args.retriever_host,
                port=args.retriever_port,
                retrieval_method=args.retrieval_method,
                top_k=args.top_k,
                startup_timeout=args.retriever_startup_timeout,
                log_path=log_path,
                server_script=server_script,
            )
        retriever = RemoteRetriever(retriever_url, max_doc_len=args.max_doc_len)
        t0 = time.time()
        fetched_docs = retriever.batch_search(all_missing_unique, args.top_k)
        print(f"[retriever] fetched {len(all_missing_unique)} queries in "
              f"{time.time() - t0:.1f}s")
        fetched_map = dict(zip(all_missing_unique, fetched_docs))

        # Fill in-memory `calls.docs`. Important for sequential mode where
        # the same process moves on to do inference.
        for entry in plan:
            for c in entry["calls"]:
                if c["docs"] is None:
                    c["docs"] = fetched_map.get(c["query"], [])

        # Persist newly fetched docs into per-dataset caches so worker
        # subprocesses (parallel mode) can read them. Always-on in parallel
        # mode; opt-in via --retriever_save_cache otherwise.
        write_back = args.parallel or args.retriever_save_cache or args.prefetch_only
        if write_back:
            for entry in plan:
                cache_path = os.path.join(
                    args.cache_dir,
                    f"search_cache.{args.retrieval_method}.{entry['dataset']}.json")
                if os.path.exists(cache_path):
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cache_blob = json.load(f)
                else:
                    cache_blob = {}
                added = 0
                for q in set(entry["missing_queries"]):
                    if q in fetched_map and q not in cache_blob:
                        cache_blob[q] = fetched_map[q]
                        added += 1
                if added:
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(cache_blob, f, ensure_ascii=False, indent=2)
                    print(f"[cache] wrote {added} new entries to {cache_path}")
    finally:
        if retriever_proc is not None:
            stop_retriever_server(retriever_proc)
            print("[retriever] released retriever GPUs")


def infogen_replay_for_dataset(args, entry: Dict, tokenizer, llm, infogen_sampling,
                               get_webpage_to_reasonchain_instruction):
    """Run vLLM infogen for a single dataset entry and write metrics + cost."""
    dataset = entry["dataset"]
    calls = entry["calls"]
    n_items = entry["n_items"]
    print(f"\n--- replaying infogen for {dataset} "
          f"({len(calls)} calls over {n_items} items) ---")

    prompts = []
    for c in calls:
        user_prompt = get_webpage_to_reasonchain_instruction(
            c["prev_reasoning"], c["query"], format_docs(c["docs"] or []))
        prompts.append(tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False, add_generation_prompt=True,
        ))

    in_toks: List[int] = []
    out_toks: List[int] = []
    if prompts:
        for batch_start in range(0, len(prompts), args.batch_size):
            batch = prompts[batch_start: batch_start + args.batch_size]
            t0 = time.time()
            outs = llm.generate(batch, sampling_params=infogen_sampling)
            in_toks.extend(len(o.prompt_token_ids) for o in outs)
            out_toks.extend(len(o.outputs[0].token_ids) for o in outs)
            print(f"  batch {batch_start // args.batch_size + 1}: "
                  f"{len(batch)} prompts in {time.time() - t0:.1f}s")

    per_in, per_out = aggregate_per_question(n_items, calls, in_toks, out_toks)

    avg_main_tok_str = None
    if entry.get("metrics_read_path") and os.path.exists(entry["metrics_read_path"]):
        with open(entry["metrics_read_path"], "r", encoding="utf-8") as f:
            existing = json.load(f)
        overall_existing = existing.get("overall", existing)
        avg_main_tok_str = overall_existing.get("avg_output_tokens")

    update_metrics_and_cost(
        entry.get("metrics_read_path"), entry["metrics_write_path"],
        entry["result_path"], dataset, per_in, per_out, avg_main_tok_str)
    print(f"  wrote cost: {entry['result_path'].replace('.json', '.cost.json')}")


def run_parallel_workers(args, plan: List[Dict], model_short: str) -> List[Tuple[str, str, str]]:
    """Dispatch dataset workers across --gpu_groups using a work-queue: each
    group runs one dataset at a time, and as soon as any worker finishes the
    freed group is handed the next queued dataset (no batch-level barriers).
    Each worker runs `--no_retriever` on its assigned GPUs.
    Returns [(dataset, status, elapsed_str), ...]."""
    gpu_groups = [g.strip() for g in args.gpu_groups.split(";") if g.strip()]
    if not gpu_groups:
        raise SystemExit("--gpu_groups is empty")

    base_cmd = [sys.executable, os.path.abspath(__file__)]
    common = [
        "--model_path", args.model_path,
        "--model_short", model_short,
        "--output_base_dir", args.output_base_dir,
        "--cache_dir", args.cache_dir,
        "--retrieval_method", args.retrieval_method,
        "--max_model_len", str(args.max_model_len),
        "--max_new_tokens", str(args.max_new_tokens),
        "--max_search_limit", str(args.max_search_limit),
        "--top_k", str(args.top_k),
        "--max_doc_len", str(args.max_doc_len),
        "--batch_size", str(args.batch_size),
        "--no_retriever",
    ]
    if args.force:
        common.append("--force")

    def _launch(entry, group):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = group
        env["VLLM_HOST_IP"] = "127.0.0.1"
        cmd = base_cmd + common + [
            "--datasets", entry["dataset"],
            "--gpus", group,
        ]
        log_path = entry["output_dir"] + ".infogen_cost.launch.log"
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        log_f = open(log_path, "w", buffering=1)
        print(f"[launcher] {entry['dataset']} -> GPUs {group}  (log: {log_path})")
        p = subprocess.Popen(cmd, env=env, stdout=log_f,
                             stderr=subprocess.STDOUT, start_new_session=True)
        return {"name": entry["dataset"], "proc": p, "log_f": log_f,
                "group": group, "t0": time.time()}

    summary: List[Tuple[str, str, str]] = []
    queue = list(plan)
    free_groups = list(gpu_groups)
    active: List[Dict] = []

    while free_groups and queue:
        active.append(_launch(queue.pop(0), free_groups.pop(0)))

    while active:
        time.sleep(2)
        still = []
        for w in active:
            ret = w["proc"].poll()
            if ret is None:
                still.append(w)
                continue
            w["log_f"].close()
            elapsed = time.time() - w["t0"]
            status = "OK" if ret == 0 else f"FAIL (exit {ret})"
            summary.append((w["name"], status, f"{elapsed:.0f}s"))
            print(f"[launcher] >> {w['name']}: {status}  ({elapsed:.0f}s)")
            if queue:
                still.append(_launch(queue.pop(0), w["group"]))
            else:
                free_groups.append(w["group"])
        active = still
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    unknown = [d for d in datasets if d not in DATASET_SPLIT]
    if unknown:
        raise SystemExit(f"Unsupported dataset(s): {unknown}. "
                         f"Supported: {sorted(DATASET_SPLIT)}")

    model_short = (args.model_short
                   or args.model_path.rstrip("/").split("/")[-1].lower()
                   .replace("-instruct", ""))

    # GPU overlap sanity check (mirrors run_search_o1_wiki.py).
    if not args.dry_run and not args.retriever_url:
        retriever_set = {g.strip() for g in args.retriever_gpus.split(",") if g.strip()}
        worker_sets = []
        if args.parallel:
            worker_sets = [
                {g.strip() for g in grp.split(",") if g.strip()}
                for grp in args.gpu_groups.split(";") if grp.strip()
            ]
        elif args.gpus:
            worker_sets = [{g.strip() for g in args.gpus.split(",") if g.strip()}]
        for ws in worker_sets:
            overlap = ws & retriever_set
            if overlap and not args.no_retriever:
                # Overlap is fine for parallel mode IF the retriever has been
                # torn down before workers start; but for sequential mode (or
                # when retriever stays up), it would deadlock — so block.
                if not args.parallel:
                    raise SystemExit(
                        f"--retriever_gpus and worker GPUs overlap on {sorted(overlap)}. "
                        f"In sequential mode the retriever stays up while inference "
                        f"runs, so they must be disjoint. (In --parallel mode the "
                        f"retriever is shut down before workers start, so overlap "
                        f"is allowed.)")

    # ----- Plan + parse phase (no GPU) -----
    plan = build_plan(args, datasets, model_short)
    if not plan:
        print("Nothing to do.")
        return
    gather_phase(args, plan)

    if args.dry_run:
        total_missing = sum(len(e["missing_queries"]) for e in plan)
        unique_missing = len({q for e in plan for q in e["missing_queries"]})
        print(f"\n[dry_run] Skipping model + retriever. "
              f"Would re-query retriever for {total_missing} calls "
              f"({unique_missing} unique queries).")
        return

    # ----- Strict cache mode (worker subprocesses): refuse to silently miss -----
    if args.no_retriever:
        offenders = [(e["dataset"], len(e["missing_queries"])) for e in plan
                     if e["missing_queries"]]
        if offenders:
            details = ", ".join(f"{d}: {n}" for d, n in offenders)
            raise SystemExit(
                f"--no_retriever is set but {sum(n for _, n in offenders)} "
                f"queries are missing from the on-disk caches ({details}). "
                f"Run the parent (without --no_retriever) so the retriever phase "
                f"populates the per-dataset caches first.")

    # ----- Retriever phase (parent only; spawns + tears down) -----
    if not args.no_retriever:
        retriever_phase(args, plan)

    if args.prefetch_only:
        print("\n[prefetch_only] retriever phase done; skipping vLLM. "
              "Re-run without --prefetch_only to compute infogen cost.")
        return

    # ----- Parallel mode: dispatch one worker subprocess per dataset -----
    if args.parallel:
        summary = run_parallel_workers(args, plan, model_short)
        print("\n" + "=" * 60)
        print("  SUMMARY (infogen-cost backfill, parallel)")
        print("=" * 60)
        for name, status, elapsed in summary:
            print(f"  {name:12s}  {status:30s}  {elapsed}")
        print("=" * 60)
        return

    # ----- Sequential mode: load vLLM once, replay each dataset in turn -----
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from run_search_o1_wiki import get_webpage_to_reasonchain_instruction

    print(f"\nLoading tokenizer + LLM from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    vllm_gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g]
    print(f"  GPUs: {vllm_gpus}")
    t0 = time.time()
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=len(vllm_gpus) or 1,
        gpu_memory_utilization=0.90,
        dtype="half",
        max_model_len=args.max_model_len,
        max_num_seqs=32,
    )
    print(f"  loaded in {time.time() - t0:.1f}s")

    infogen_sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
    )

    for entry in plan:
        infogen_replay_for_dataset(args, entry, tokenizer, llm, infogen_sampling,
                                   get_webpage_to_reasonchain_instruction)

    print("\nDone.")


if __name__ == "__main__":
    main()
