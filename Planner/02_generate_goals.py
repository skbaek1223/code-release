from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MAX_CONTEXT_CHARS = 12_000  # A4 약 3장 분량 (비용 절감)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _truncate_context(ctx_text: str, answer: str) -> str:
    """context가 너무 길면 answer가 포함된 supporting sentence 중심으로 잘라낸다."""
    sents = _SENT_SPLIT.split(ctx_text)

    if len(sents) >= 2:
        # supporting sentence 찾기
        sup_idx = None
        for i, s in enumerate(sents):
            if answer in s:
                sup_idx = i
                break
        if sup_idx is None:
            sup_idx = len(sents) // 2

        sup_sent = sents[sup_idx]
        budget = MAX_CONTEXT_CHARS - len(sup_sent)

        before = " ".join(sents[:sup_idx])
        after = " ".join(sents[sup_idx + 1:])
        half = budget // 2

        if before and len(before) > half:
            before = "[...] " + before[-(half - 6):]
        if after and len(after) > half:
            after = after[:half - 6] + " [...]"

        parts = [p for p in (before, sup_sent, after) if p]
        return " ".join(parts)

    # 문장 1개 (테이블 등): 단어 단위로 answer 중심 앞뒤 균등 자르기
    words = ctx_text.split()
    ans_idx = None
    for i, w in enumerate(words):
        if answer in w:
            ans_idx = i
            break
    if ans_idx is None:
        ans_idx = 0

    half = MAX_CONTEXT_CHARS // 2
    before_words = []
    used = 0
    for w in reversed(words[:ans_idx]):
        cost = len(w) + 1
        if used + cost > half:
            break
        before_words.append(w)
        used += cost
    before_words.reverse()

    after_words = []
    used = 0
    for w in words[ans_idx:]:
        cost = len(w) + 1
        if used + cost > half:
            break
        after_words.append(w)
        used += cost

    prefix = "[...] " if len(before_words) < ans_idx else ""
    suffix = " [...]" if len(after_words) < len(words) - ans_idx else ""
    return prefix + " ".join(before_words + after_words) + suffix


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key)


client = get_client()

DATA_DIR = Path(__file__).parent.parent / "data"
HARD_SELECTED_DIR = DATA_DIR / "hard_selected"
GOALS_DIR = DATA_DIR / "goals"
BATCH_DIR = DATA_DIR / "batches"

MODEL = "gpt-4.1"

SINGLEHOP_GOAL_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {"type": "string"},
    },
    "required": ["plan"],
    "additionalProperties": False,
}

MULTIHOP_GOAL_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 6,
        },
    },
    "required": ["steps"],
    "additionalProperties": False,
}

SINGLEHOP_SYSTEM = """You are an information retrieval planning expert. Given a single-hop question, a supporting context, and the answer, generate a single concrete retrieval step required to find the answer.

Output a JSON object with one field:
- "plan" (string): a single retrieval step.

Ensure:
1. The step should be targeted enough that a retrieval system would reliably return the supporting context and the answer can be confidently derived from it. Each step must target a specific answer-bearing attribute.
2. The plan must be derivable solely from the question. Do not include specific facts, names, or details that are only found in the answer or supporting context."""

SINGLEHOP_FEWSHOT = [
    {
        "input": {
            "question": "who sang the most wonderful summer of my life",
            "supporting_context": [
                {
                    "text": "Jackie Ward ( born Jacqueline McDonnell , 1941 ) , better known as Robin Ward , is an American singer , regarded as a `` one - hit wonder '' of 1963 million - selling song `` Wonderful Summer '' . However , using her real name she was highly accomplished and successful singing in groups ."
                },
            ],
            "answer": "Jackie Ward",
        },
        "output": {
            "plan": "Find the performer or singer of the song 'The Most Wonderful Summer of My Life'"
        },
    },
    {
        "input": {
            "question": "a spider with a skull on its back",
            "supporting_context": [
                {
                    "text": "Steatoda nobilis has a brown bulbous abdomen with cream coloured markings that are often likened to the shape of a skull . The legs are reddish - orange ."
                },
            ],
            "answer": "Steatoda nobilis",
        },
        "output": {
            "plan": "Identify the type of spider that has a skull-shaped marking on its back"
        },
    },
    {
        "input": {
            "question": "at what output is marginal product of labour the highest",
            "supporting_context": [
                {
                    "text": "When the MP is above the AP the AP will increase . Eventually the MP reaches it maximum value at the point of diminishing returns . Beyond this point MP will decrease ."
                },
            ],
            "answer": "at the point of diminishing returns",
        },
        "output": {
            "plan": "Determine the output level at which the marginal product of labour reaches its maximum value"
        },
    },
]
MULTIHOP_SYSTEM = """You are an information retrieval planning expert. Given a multi-hop question, context sources, and the answer, generate an ordered sequence of retrieval steps required to find the answer. Each step represents one concrete retrieval action.

Output a JSON object with one field:
- "steps" (array of strings): the ordered retrieval plan (2 to 6 steps).

Ensure:
1. The retrieval steps should be targeted enough that a retrieval system would reliably return all context sources needed to derive the answer. Each step must target a specific answer-bearing attribute.
2. The answer should be confidently derived from them.
3. Each step must be derivable solely from the question and the results of prior steps. Do not include specific facts, names, or details that are only found in the answer or supporting context."""

HOTPOTQA_FEWSHOT = [
    {
        "input": {
            "question": "Gunmen from Laredo starred which narrator of \"Frontier\"?",
            "supporting_context": [
                {
                    "text": "Gunmen from Laredo is a 1959 American western film produced and directed by Wallace MacDonald, which stars Robert Knapp, Maureen Hingert, and Walter Coy."
                },
                {
                    "text": "Walter Darwin Coy (January 31, 1909 – December 11, 1974) was an American stage, radio, film, and, principally, television actor, originally from Great Falls, Montana.  He was best known for narrating the NBC western anthology series, \"Frontier\", which aired early Sunday evenings in the 1955–1956 season."
                },
            ],
            "answer": "Walter Darwin Coy",
        },
        "output": {
            "steps": [
                "Find the actors who starred in 'Gunmen from Laredo'",
                "Determine which of these actors was the narrator of the series 'Frontier'",
            ],
        },
    },
    {
        "input": {
            "question": "Who invented the type of script used in autographs?",
            "supporting_context": [
                {
                    "text": "An autograph in Assyriology is the hand-copy of a cuneiform clay-tablet.  Producing an autograph is often the first step of a tablet's archaeological interpretation and the autograph is frequently the authoritative form that is published as source material."
                },
                {
                    "text": "Cuneiform script ( or or ), one of the earliest systems of writing, was invented by the Sumerians.  It is distinguished by its wedge-shaped marks on clay tablets, made by means of a blunt reed for a stylus."
                },
            ],
            "answer": "the Sumerians",
        },
        "output": {
            "steps": [
                "Determine the type of script used in autographs in the relevant field",
                "Find who invented that type of script",
            ],
        },
    },
    {
        "input": {
            "question": "The Bass Rock Lighthouse was next to what Castle?",
            "supporting_context": [
                {
                    "text": "Canty Bay is a coastal hamlet off the A198, in East Lothian, Scotland, situated opposite the Bass Rock and Tantallon Castle.  Settlements nearby include Auldhame, Scoughall, Seacliff, and the Peffer Sands."
                },
                {
                    "text": " The island belongs to Sir Hew Hamilton-Dalrymple, whose family acquired it in 1706, and before to the Lauder family for almost six centuries.  The Bass Rock Lighthouse was constructed on the rock in 1902, and the remains of an ancient chapel survive."
                },
            ],
            "answer": "Tantallon Castle",
        },
        "output": {
            "steps": [
                "Find the location of the Bass Rock Lighthouse",
                "Identify landmarks or castles located next to that location",
            ],
        },
    },
]


FEWSHOT_EXCLUDE_IDS: dict[str, set[str]] = {
    "nq": {"6500121756220242048", "8064791005932956515", "3817967454337065898"},
    "hotpotqa": {"5aba66c855429939ce03dcdb", "5a74a3bd55429979e28829da", "5add85b65542997545bbbd61"},
}

def make_user_prompt(item: dict, dataset: str = "nq") -> str:
    lines = [f"Question: {item['question']}"]
    ctx_list = item.get("supporting_context", [])
    answer = item.get("answer", "")
    if ctx_list:
        if dataset == "nq":
            ctx_text = "\n\n".join(s["text"] for s in ctx_list)
            if len(ctx_text) > MAX_CONTEXT_CHARS:
                ctx_text = _truncate_context(ctx_text, answer)
            lines.append("Supporting context:\n" + ctx_text)
        else:
            per_source_budget = MAX_CONTEXT_CHARS // len(ctx_list)
            ctx_parts = []
            for i, s in enumerate(ctx_list, 1):
                text = s["text"]
                if len(text) > per_source_budget:
                    text = _truncate_context(text, answer)
                ctx_parts.append(f"[Source {i}] {text}")
            lines.append("Context sources:\n" + "\n\n".join(ctx_parts))
    lines.append(f"Answer: {answer}")
    return "\n\n".join(lines)


def build_fewshot_messages(fewshot: list[dict], dataset: str = "nq") -> list[dict]:
    messages = []
    for ex in fewshot:
        messages.append({"role": "user", "content": make_user_prompt(ex["input"], dataset)})
        messages.append({"role": "assistant", "content": json.dumps(ex["output"], ensure_ascii=False)})
    return messages


def build_batch_requests(
    items: list[dict],
    system_prompt: str,
    schema: dict,
    fewshot: list[dict] | None = None,
    dataset: str = "nq",
) -> list[dict]:
    fewshot_messages = build_fewshot_messages(fewshot, dataset) if fewshot else []
    requests = []
    for item in items:
        requests.append(
            {
                "custom_id": item["id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        *fewshot_messages,
                        {"role": "user", "content": make_user_prompt(item, dataset)},
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "goal_decomposition",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                    "max_tokens": 256,
                },
            }
        )
    return requests


def submit_batch(requests: list[dict], description: str) -> str:
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    input_path = BATCH_DIR / f"{description}_input.jsonl"

    with open(input_path, "w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    with open(input_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    print(f"Batch submitted: {batch.id} ({description})")
    return batch.id


def wait_and_download(
    batch_id: str, description: str, timeout: int = 7200, stall_timeout: int = 600
) -> list[dict]:
    print(f"Waiting for batch {batch_id} (timeout={timeout}s, stall={stall_timeout}s)...")
    start = time.time()
    last_completed = -1
    last_progress_time = start
    cancelled = False

    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        counts = getattr(batch, "request_counts", None)
        completed = getattr(counts, "completed", 0) if counts else 0
        total = getattr(counts, "total", 0) if counts else 0
        failed = getattr(counts, "failed", 0) if counts else 0

        print(f"  status: {status} | completed: {completed}/{total} | failed: {failed}")

        if status == "completed":
            break

        if status == "cancelled":
            if getattr(batch, "output_file_id", None):
                print(f"  Batch cancelled — downloading {completed} partial results...")
                break
            raise RuntimeError(f"Batch {batch_id} cancelled with no output file.")

        if status in ("failed", "expired"):
            raise RuntimeError(f"Batch {batch_id} ended with status: {status}")

        now = time.time()

        if completed != last_completed:
            last_completed = completed
            last_progress_time = now

        stalled = now - last_progress_time > stall_timeout and completed > 0
        timed_out = now - start > timeout

        if (stalled or timed_out) and not cancelled:
            reason = "stall" if stalled else "timeout"
            remaining = total - completed - failed
            print(
                f"  {reason.title()} detected — {remaining} request(s) still pending. "
                f"Cancelling batch to retrieve {completed} completed results..."
            )
            client.batches.cancel(batch_id)
            cancelled = True

        if cancelled and now - start > timeout + 300:
            raise RuntimeError(
                f"Batch {batch_id}: cancel did not complete within 5 min after timeout. "
                f"Retrieve results manually with batch id."
            )

        time.sleep(30)

    if not getattr(batch, "output_file_id", None):
        raise RuntimeError(f"Batch {batch_id} completed but has no output_file_id.")

    output_path = BATCH_DIR / f"{description}_output.jsonl"
    content = client.files.content(batch.output_file_id)
    output_path.write_bytes(content.read())
    print(f"Downloaded: {output_path}")

    results = []
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    return results


def extract_message_content(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content

    if isinstance(message_content, list):
        text_parts = []
        for part in message_content:
            if isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    text_parts.append(part["text"])
                elif "content" in part and isinstance(part["content"], str):
                    text_parts.append(part["content"])
        if text_parts:
            return "".join(text_parts)

    raise ValueError(f"Unsupported message content format: {type(message_content)}")


def parse_results(results: list[dict], items_by_id: dict) -> list[dict]:
    parsed = []

    for result in results:
        custom_id = result.get("custom_id")
        item = items_by_id.get(custom_id)
        if item is None:
            continue

        response = result.get("response", {})
        if response.get("status_code") != 200:
            print(f"  SKIP {custom_id}: status {response.get('status_code')}")
            continue

        try:
            message = response["body"]["choices"][0]["message"]
            raw_content = extract_message_content(message["content"])
            goal_data = json.loads(raw_content)

            if "plan" in goal_data:
                goal = re.sub(r"\s+", " ", goal_data["plan"]).strip()
                parsed.append({**item, "plan": goal})
            else:
                steps = goal_data.get("steps", [])
                cleaned_steps = [re.sub(r"\s+", " ", s).strip() for s in steps]
                parsed.append({**item, "steps": cleaned_steps})

        except Exception as e:
            print(f"  SKIP {custom_id}: parse error {e}")
            continue

    return parsed


# ═══════════════════════════════════════════════════════════════
#  LLM-as-Judge: goals 판정
# ═══════════════════════════════════════════════════════════════

def normalize_goals(items: list[dict]) -> list[dict]:
    """goals 의 plan/steps 를 predicted_steps 로 정규화."""
    normalized = []
    for item in items:
        item = dict(item)
        if "plan" in item:
            item["predicted_steps"] = [item["plan"]]
        elif "steps" in item:
            item["predicted_steps"] = item["steps"]
        else:
            continue
        normalized.append(item)
    return normalized


def judge_goals(
    dataset: str,
    goals_path: Path | None = None,
    judged_path: Path | None = None,
    gpu: str | None = None,
    port_override: int | None = None,
    max_workers: int | None = None,
    shard_id: int | None = None,
    num_shards: int | None = None,
):
    """goals 파일을 LLM-as-Judge 로 판정하여 judged 파일에 저장."""
    import sys as _sys
    from importlib import import_module as _import
    _sys.path.insert(0, str(Path(__file__).parent))
    eval_mod = _import("06_evaluate_val")

    num_workers = max_workers or int(os.environ.get("NUM_WORKERS", "12"))
    is_shard = shard_id is not None and num_shards is not None

    if goals_path is None:
        goals_path = GOALS_DIR / f"{dataset}_goals.jsonl"
    if judged_path is None:
        suffix = f"_s{shard_id}" if is_shard else ""
        judged_path = GOALS_DIR / f"{dataset}_goals_judged{suffix}.jsonl"

    if not goals_path.exists():
        print(f"[{dataset}] goals 파일 없음: {goals_path}")
        return []

    items = []
    with open(goals_path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    items = normalize_goals(items)

    # 샤딩: 전체 아이템 중 해당 샤드만 선택
    if is_shard:
        items = [item for i, item in enumerate(items) if i % num_shards == shard_id]
        print(f"[{dataset}] shard {shard_id}/{num_shards}: {len(items)}개")
    else:
        print(f"[{dataset}] goals 로드: {len(items)}개")

    # 이미 판정된 ID 확인 → 이어쓰기
    done: dict[str, dict] = {}
    if judged_path.exists():
        with open(judged_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["id"]] = r
                except (json.JSONDecodeError, KeyError):
                    pass
        if done:
            print(f"[{dataset}] 기존 {len(done)}개 판정 완료, 나머지 이어쓰기")

    remaining = [item for item in items if item["id"] not in done]
    if not remaining:
        print(f"[{dataset}] 모든 항목 판정 완료")
    else:
        cfg = eval_mod.DATASET_CONFIGS[dataset]
        model_name = cfg["model_name"]
        model_path = cfg["model_path"]
        port = port_override if port_override is not None else cfg["port"]

        gpu_id = gpu if gpu is not None else eval_mod.find_free_gpus(1)[0]
        proc = eval_mod.start_vllm(model_path, model_name, port, str(gpu_id))

        from openai import OpenAI as _OpenAI
        oai_client = _OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")

        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from tqdm import tqdm

            print(f"[{dataset}] 판정 대상: {len(remaining)}개 (model: {model_name}, workers: {num_workers})")

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(
                        eval_mod.judge_single, item, dataset, oai_client, model_name,
                    ): item["id"]
                    for item in remaining
                }
                shard_label = f"[{dataset}/s{shard_id}]" if is_shard else f"[{dataset}]"
                with tqdm(total=len(remaining), desc=f"{shard_label} judge", unit="item") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        done[result["id"]] = result
                        pbar.update(1)
        finally:
            eval_mod.stop_vllm(proc, port)

    all_judged = list(done.values())
    with open(judged_path, "w", encoding="utf-8") as f:
        for r in all_judged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pass = sum(1 for r in all_judged if r.get("pass"))
    n_fail = len(all_judged) - n_pass
    print(f"[{dataset}] 판정 완료: PASS {n_pass} / FAIL {n_fail} (총 {len(all_judged)}개)")
    return all_judged


def merge_and_filter(dataset: str, num_shards: int):
    """샤드별 judged 파일을 병합하고 filtered 파일을 생성한다."""
    all_judged: dict[str, dict] = {}
    for s in range(num_shards):
        shard_path = GOALS_DIR / f"{dataset}_goals_judged_s{s}.jsonl"
        if not shard_path.exists():
            raise FileNotFoundError(f"{shard_path} not found")
        with open(shard_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                all_judged[r["id"]] = r

    judged_path = GOALS_DIR / f"{dataset}_goals_judged.jsonl"
    with open(judged_path, "w", encoding="utf-8") as f:
        for r in all_judged.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_pass = sum(1 for r in all_judged.values() if r.get("pass"))
    n_total = len(all_judged)
    print(f"[{dataset}] 병합 완료: PASS {n_pass} / FAIL {n_total - n_pass} (총 {n_total}개)")

    write_filtered(dataset, list(all_judged.values()))

    for s in range(num_shards):
        shard_path = GOALS_DIR / f"{dataset}_goals_judged_s{s}.jsonl"
        shard_path.unlink(missing_ok=True)


def write_filtered(dataset: str, all_judged: list[dict]):
    """PASS 판정된 goals 만 filtered 파일로 저장."""
    goals_path = GOALS_DIR / f"{dataset}_goals.jsonl"
    filtered_path = GOALS_DIR / f"{dataset}_goals_filtered.jsonl"

    pass_ids = {r["id"] for r in all_judged if r.get("pass")}
    filtered = []
    with open(goals_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["id"] in pass_ids:
                filtered.append(item)

    with open(filtered_path, "w", encoding="utf-8") as f:
        for item in filtered:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  필터링 결과 → {filtered_path} ({len(filtered)}개)")


# ═══════════════════════════════════════════════════════════════
#  메인: generate → judge → filtered
# ═══════════════════════════════════════════════════════════════

def run(dataset: str, limit: int | None = None,
        gpu: str | None = None, port: int | None = None,
        skip_judge: bool = False):
    raw_path = HARD_SELECTED_DIR / f"{dataset}_hard_selected.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(f"{raw_path} not found. Run precompute/04_extract_hard.py first.")

    items = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    config = {
        "nq": (SINGLEHOP_SYSTEM, SINGLEHOP_GOAL_SCHEMA, SINGLEHOP_FEWSHOT),
        "hotpotqa": (MULTIHOP_SYSTEM, MULTIHOP_GOAL_SCHEMA, HOTPOTQA_FEWSHOT),
    }

    if dataset not in config:
        raise ValueError("Usage: python 02_generate_goals.py [nq|hotpotqa]")

    system_prompt, schema, fewshot = config[dataset]

    exclude_ids = FEWSHOT_EXCLUDE_IDS.get(dataset, set())
    items = [item for item in items if item["id"] not in exclude_ids]

    # 기존 goals 파일이 있으면 이미 생성된 ID 제외 (중복 생성 방지)
    out_path = GOALS_DIR / f"{dataset}_goals.jsonl"
    existing_ids: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                existing_ids.add(json.loads(line)["id"])
        print(f"기존 goals: {len(existing_ids)}개 (중복 제외)")

    items = [item for item in items if item["id"] not in existing_ids]

    if limit is not None:
        items = items[:limit]
        print(f"limit={limit} 적용 → {len(items)}개 생성 예정")

    if not items:
        print(f"{dataset}: 생성할 항목 없음 (이미 충분)")
    else:
        items_by_id = {item["id"]: item for item in items}

        requests = build_batch_requests(items, system_prompt, schema, fewshot, dataset)
        batch_id = submit_batch(requests, dataset)
        results = wait_and_download(batch_id, dataset)

        GOALS_DIR.mkdir(parents=True, exist_ok=True)
        parsed = parse_results(results, items_by_id)

        # 파일 쓰기 직전에 기존 ID를 다시 읽어서 중복 방지 (동시 실행 등 대비)
        if out_path.exists():
            existing_ids.clear()
            with open(out_path, encoding="utf-8") as f:
                for line in f:
                    existing_ids.add(json.loads(line)["id"])

        # parsed 내부 중복 제거 + 기존 ID 제외
        seen: set[str] = set(existing_ids)
        deduped: list[dict] = []
        for item in parsed:
            if item["id"] not in seen:
                seen.add(item["id"])
                deduped.append(item)

        if not deduped:
            print(f"\n{dataset}: 새로 추가할 항목 없음 (모두 중복)")
        else:
            with open(out_path, "a", encoding="utf-8") as f:
                for item in deduped:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

            total = len(existing_ids) + len(deduped)
            print(f"\n{dataset}: +{len(deduped)} 추가 (총 {total}개) → {out_path}")

    # LLM-as-Judge
    if skip_judge:
        print(f"\n[{dataset}] judge 건너뛰기 (--skip-judge)")
        return

    print(f"\n=== [{dataset}] LLM-as-Judge ===")
    all_judged = judge_goals(dataset, gpu=gpu, port_override=port)
    if all_judged:
        write_filtered(dataset, all_judged)


if __name__ == "__main__":
    import argparse as _ap

    parser = _ap.ArgumentParser(
        description="Goals 생성 (Batch API) + LLM-as-Judge 판정")
    parser.add_argument("dataset", nargs="?", default="nq", choices=["nq", "hotpotqa"])
    parser.add_argument("--limit", type=int, default=None, help="생성할 최대 개수 (기존 제외 후 적용)")
    parser.add_argument("--gpu", type=str, default=None, help="사용할 GPU ID")
    parser.add_argument("--port", type=int, default=None, help="vLLM 포트")
    parser.add_argument("--skip-judge", action="store_true", help="LLM judge 건너뛰기")
    parser.add_argument("--judge-only", action="store_true", help="생성 건너뛰고 judge 만 실행")
    parser.add_argument("--shard-id", type=int, default=None, help="샤드 ID (0-based)")
    parser.add_argument("--num-shards", type=int, default=None, help="전체 샤드 수")
    args = parser.parse_args()

    if args.judge_only:
        all_judged = judge_goals(
            args.dataset, gpu=args.gpu, port_override=args.port,
            shard_id=args.shard_id, num_shards=args.num_shards,
        )
        if all_judged and args.shard_id is None:
            write_filtered(args.dataset, all_judged)
    else:
        run(args.dataset, args.limit, gpu=args.gpu, port=args.port,
            skip_judge=args.skip_judge)