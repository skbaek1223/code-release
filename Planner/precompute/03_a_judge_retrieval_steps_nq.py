"""
Step 3 (NQ): Qwen2.5-14B-Instruct-GPTQ-Int4로 retrieval plan 품질 판정

Judge 입력: question, answer, supporting context (single source), predicted_steps
Judge 출력: {"pass": bool, "reason": str}

NQ는 single-hop 데이터셋으로 supporting_context가 항상 1개.
- pass=True  → 단계가 충분히 구체적 (쉬운 문제) → hard_selected에서 제외
- pass=False → 단계가 너무 모호하거나 답을 도출할 수 없음 → hard_selected에 포함

출력: data/precompute/nq_all_judged.jsonl
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

_raw_urls = os.environ.get("JUDGE_BASE_URL", "http://localhost:8001/v1")
JUDGE_BASE_URLS = [u.strip() for u in _raw_urls.split(",") if u.strip()]
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4")
JUDGE_MODEL_PATH = os.environ.get("JUDGE_MODEL_PATH", "/mnt/raid6/skbaek1223/models/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4")
JUDGE_PORT = int(os.environ.get("JUDGE_PORT", "8001"))
AUTO_VLLM = os.environ.get("AUTO_VLLM", "1") == "1"
MIN_FREE_MB = int(os.environ.get("MIN_FREE_MB", "15000"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "12"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
MAX_RETRIES = 3
MAX_TOKENS = 256
MAX_CONTEXT_CHARS = 24_000  # 보수적: 숫자/특수문자 많으면 1토큰≈1.5글자까지 내려감

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "precompute"
IN_PATH = DATA_DIR / "nq_all_with_steps.jsonl"
OUT_PATH = DATA_DIR / "nq_all_judged.jsonl"

client: OpenAI | None = None


# ──────────────────────────────────────────────
# GPU / vLLM 관리
# ──────────────────────────────────────────────

def find_free_gpus(n: int = 1, min_free_mb: int = MIN_FREE_MB) -> list[str]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    gpus: list[str] = []
    for line in result.stdout.strip().splitlines():
        idx, free_mb = line.split(", ")
        if int(free_mb.strip()) >= min_free_mb:
            gpus.append(idx.strip())
        if len(gpus) == n:
            break
    if len(gpus) < n:
        raise RuntimeError(
            f"여유 GPU {n}장 없음 (기준: {min_free_mb} MB 이상, 찾은 수: {len(gpus)})"
        )
    return gpus


def _ping(base_url: str) -> bool:
    try:
        OpenAI(api_key="EMPTY", base_url=base_url).models.list()
        return True
    except Exception:
        return False


def start_vllm(model_path: str, port: int, gpu_id: str, timeout: int = 300) -> subprocess.Popen | None:
    base_url = f"http://localhost:{port}/v1"
    if _ping(base_url):
        print(f"[port {port}] vLLM 서버 이미 실행 중 — 기존 서버 사용.")
        return None

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "1200")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")

    log_path = f"/tmp/vllm_{port}_stderr.log"
    print(f"[port {port}] vLLM 시작 중... (model={Path(model_path).name}, GPU={gpu_id})")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--served-model-name", JUDGE_MODEL,
            "--tensor-parallel-size", "1",
            "--port", str(port),
            "--dtype", "float16",
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=open(log_path, "w"),
    )

    for _ in range(timeout):
        time.sleep(1)
        if proc.poll() is not None:
            raise RuntimeError(f"[port {port}] vLLM 프로세스가 예기치 않게 종료됨. 로그: {log_path}")
        if _ping(base_url):
            print(f"[port {port}] vLLM 준비 완료. (GPU {gpu_id})")
            return proc

    proc.terminate()
    raise RuntimeError(f"[port {port}] vLLM 시작 시간 초과 ({timeout}초). 로그: {log_path}")


def stop_vllm(proc: subprocess.Popen | None, port: int):
    if proc is None:
        return
    print(f"[port {port}] vLLM 종료 중...")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()

JUDGE_SYSTEM = """You are a quality judge for a single-hop retrieval plan.

You will be given a question, a predicted retrieval plan, the supporting context, and the answer.

Pass if the retrieval plan is targeted enough that a retrieval system would reliably return the supporting context and the answer can be confidently derived from it.

Respond in exactly two lines:
Line 1: PASS or FAIL
Line 2: One sentence — explain why the step passes or fails.

Example (PASS):
Question: where was donovan mitchell picked in the draft
Predicted retrieval plan: Find Donovan Mitchell's draft pick number.
Supporting context:
Mitchell was drafted by the Denver Nuggets with the 13th overall pick in the 2017 NBA draft only to be traded to the Utah Jazz for the 24th pick ( Tyler Lydon ) and Trey Lyles . On July 5 , 2017 , Mitchell signed a four - year rookie scale contract with the Jazz .
Answer: 13th

PASS
The step directly asks for Donovan Mitchell's draft pick number, which reliably retrieves the supporting context about his 13th overall selection from which the answer can be confidently derived.

Example (FAIL):
Question: what does april's baby have on grey's anatomy
Predicted retrieval plan: Find the plot or specific details about April's baby in the TV show Grey's Anatomy.
Supporting context:
Not long after their fight , April realizes she is pregnant . April and Jackson 's baby is diagnosed during pregnancy with Osteogenesis Imperfecta type 2 , and learn that the baby will not survive long after birth . Jackson believes that termination is the best option , however April would rather give birth to the baby knowing it will not live very long .
Answer: Osteogenesis Imperfecta type 2

FAIL
The step asks for general plot details about April's baby rather than specifically asking what medical condition the baby was diagnosed with, so it would retrieve many unrelated plot points instead of reliably targeting the supporting context about the Osteogenesis Imperfecta diagnosis."""


def _truncate_context(ctx_text: str, answer: str) -> str:
    """context가 너무 길면 answer 기준으로 잘라낸다."""
    sents = _SENT_SPLIT.split(ctx_text)

    if len(sents) >= 2:
        # 문장 2개 이상: supporting sentence 보존, ±1 문장에서 자르기
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
    # answer 앞쪽: answer 직전부터 역순으로 half만큼
    before_words = []
    used = 0
    for w in reversed(words[:ans_idx]):
        cost = len(w) + 1
        if used + cost > half:
            break
        before_words.append(w)
        used += cost
    before_words.reverse()
    # answer 뒤쪽: answer부터 순서대로 half만큼
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


def make_judge_prompt(item: dict) -> str:
    lines = [
        f"Question: {item['question']}",
    ]

    steps = item.get("predicted_steps", [])
    if steps:
        lines.append(f"Predicted retrieval plan: {steps[0]}")

    ctx_parts = [snippet["text"] for snippet in item.get("supporting_context", [])]
    if ctx_parts:
        ctx_text = "\n\n".join(ctx_parts)
        if len(ctx_text) > MAX_CONTEXT_CHARS:
            ctx_text = _truncate_context(ctx_text, item.get("answer", ""))
        lines.append("Supporting context:\n" + ctx_text)

    lines.append(f"Answer: {item['answer']}")

    lines.append("Now evaluate the predicted retrieval plan above.")

    return "\n\n".join(lines)


_VERDICT_RE = re.compile(r"^\s*(PASS|FAIL)\s*$", re.MULTILINE | re.IGNORECASE)


def judge_item(item: dict) -> dict | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": make_judge_prompt(item)},
                ],
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            )
            content = response.choices[0].message.content.strip()
            verdict_match = _VERDICT_RE.search(content)
            if not verdict_match:
                print(f"  SKIP {item['id']}: no PASS/FAIL found")
                return None
            judge_pass = verdict_match.group(1).upper() == "PASS"
            reason = content[verdict_match.end():].strip()
            return {**item, "judge_pass": judge_pass, "judge_reason": reason}
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt
                print(f"  RETRY {item['id']} ({attempt}/{MAX_RETRIES}): {e} — {wait}s 대기")
                time.sleep(wait)
            else:
                print(f"  ERROR {item['id']}: {e} (재시도 {MAX_RETRIES}회 실패)")
                return None


def load_done_ids(out_path: Path) -> set[str]:
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["id"])
    return done


def main():
    global client

    proc = None
    if AUTO_VLLM:
        gpus = find_free_gpus(1)
        print(f"사용 GPU: {gpus[0]} (port {JUDGE_PORT})")
        proc = start_vllm(JUDGE_MODEL_PATH, JUDGE_PORT, gpus[0])
        client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{JUDGE_PORT}/v1")
    else:
        client = OpenAI(api_key="EMPTY", base_url=JUDGE_BASE_URLS[0])

    try:
        _main_process()
    finally:
        stop_vllm(proc, JUDGE_PORT)


def _main_process():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"{IN_PATH} not found. Run 02a_generate_retrieval_steps_nq.py first.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(OUT_PATH)
    print(f"기존 완료 항목: {len(done_ids)}개 (이어서 처리)")

    written = 0
    failed = 0
    passed = 0
    skipped = 0
    total_processed = 0
    batch: list[dict] = []

    with open(OUT_PATH, "a", encoding="utf-8") as f_out, \
         open(IN_PATH, encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if item["id"] in done_ids:
                skipped += 1
                continue
            batch.append(item)

            if len(batch) >= CHUNK_SIZE:
                with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    futures = {executor.submit(judge_item, it): it for it in batch}
                    for future in as_completed(futures):
                        result = future.result()
                        if result is not None:
                            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f_out.flush()
                            written += 1
                            if result["judge_pass"]:
                                passed += 1
                        else:
                            failed += 1
                total_processed += len(batch)
                print(f"  Progress: {total_processed} new + {skipped} skipped (pass={passed}, fail={written-passed}, error={failed})")
                batch = []

        # 마지막 남은 batch 처리
        if batch:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {executor.submit(judge_item, it): it for it in batch}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush()
                        written += 1
                        if result["judge_pass"]:
                            passed += 1
                    else:
                        failed += 1
            total_processed += len(batch)

    print(f"\nDone. New: {written}, Skipped: {skipped}, Failed: {failed}")
    print(f"  Judge PASS: {passed} ({100*passed/max(written,1):.1f}%)")
    print(f"  Judge FAIL: {written - passed} ({100*(written-passed)/max(written,1):.1f}%)")
    print(f"  Total in output: {len(done_ids) + written}")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
