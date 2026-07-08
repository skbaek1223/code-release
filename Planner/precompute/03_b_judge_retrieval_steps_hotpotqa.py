"""
Step 3: Qwen2.5-32B-Instruct-GPTQ-Int4로 retrieval plan 품질 판정

Judge 입력: question, answer, supporting context, predicted_steps
Judge 출력: {"pass": bool, "reason": str}

- pass=True  → Qwen이 충분한 plan 생성 (쉬운 문제) → hard_selected에서 제외
- pass=False → plan이 불충분 (어려운 문제) → hard_selected에 포함

GPU 2장을 사용하여 입력을 반으로 나눠 병렬 처리 후 합침.
출력: data/precompute/hotpotqa_all_judged.jsonl
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

_raw_urls = os.environ.get("JUDGE_BASE_URL", "http://localhost:8001/v1")
JUDGE_BASE_URLS = [u.strip() for u in _raw_urls.split(",") if u.strip()]
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4")
JUDGE_MODEL_PATH = os.environ.get("JUDGE_MODEL_PATH", "/mnt/raid6/skbaek1223/models/Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4")
JUDGE_PORT_BASE = int(os.environ.get("JUDGE_PORT", "8002"))
AUTO_VLLM = os.environ.get("AUTO_VLLM", "1") == "1"
MIN_FREE_MB = int(os.environ.get("MIN_FREE_MB", "15000"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "12"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
MAX_RETRIES = 3
MAX_TOKENS = 256
NUM_GPUS = 2

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "precompute"
IN_PATH = DATA_DIR / "hotpotqa_all_with_steps.jsonl"
OUT_PATH = DATA_DIR / "hotpotqa_all_judged.jsonl"


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

JUDGE_SYSTEM = """You are a quality judge for a multi-hop retrieval plan.

You will be given a question, predicted retrieval steps, numbered context sources, and the answer.

Pass if:
1. The retrieval steps are targeted enough that a retrieval system would reliably return all context sources needed to derive the answer.
2. The answer can be confidently derived from them.

Respond in exactly two lines:
Line 1: PASS or FAIL
Line 2: One sentence — explain why the plan passes or fails.

Example (PASS):
Question: The Oberoi family is part of a hotel company that has a head office in what city?
Step 1: Find which hotel company the Oberoi family is part of.
Step 2: Find the city where that hotel company has its head office.
[Source 1] The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.
[Source 2] The Oberoi Group is a hotel company with its head office in Delhi.  Founded in 1934, the company owns and/or operates 30+ luxury hotels and two river cruise ships in six countries, primarily under its Oberoi Hotels & Resorts and Trident Hotels brands.
Answer: Delhi

PASS
Step 1 targets [Source 1] to retrieve the hotel company name (The Oberoi Group), and Step 2 uses that name to target [Source 2] for the head office city — both sources are reachable in the correct order, and each step's goal is specific enough to retrieve exactly the bridging fact needed.

Example (FAIL):
Question: Of the film directors Kenji Mizoguchi and Andrzej Żuławski, which one often went against mainstream commercialism in his films?
Step 1: Find information about Kenji Mizoguchi's approach to mainstream commercialism in his films.
Step 2: Find information about Andrzej Żuławski's approach to mainstream commercialism in his films.
[Source 1] Kenji Mizoguchi (溝口 健二 , Mizoguchi Kenji , May 16, 1898 – August 24, 1956) was a Japanese film director and screenwriter.
[Source 2] Andrzej Żuławski (22 November 1940 – 17 February 2016) was a Polish film director and writer.  He was born in Lwów, Poland (now Ukraine).  Żuławski often went against mainstream commercialism in his films, and enjoyed success mostly with European art-house audiences.
Answer: Andrzej Żuławski

FAIL
Both steps vaguely ask to "find information about" each director's approach to mainstream commercialism rather than specifically asking whether each director went against it — the overly broad queries would retrieve general biographical information instead of reliably targeting the specific fact in [Source 2] about Żuławski's anti-commercial stance."""


def make_judge_prompt(item: dict) -> str:
    lines = [
        f"Question: {item['question']}",
    ]

    steps = item.get("predicted_steps", [])
    lines.append(
        "Predicted retrieval plan:\n" + "\n".join(f"  Step {i+1}: {s}" for i, s in enumerate(steps))
    )

    ctx_parts = [f"[Source {i}] {snippet['text']}" for i, snippet in enumerate(item.get("supporting_context", []), 1)]
    if ctx_parts:
        lines.append("Context sources:\n" + "\n\n".join(ctx_parts))

    lines.append(f"Answer: {item['answer']}")

    lines.append("Now evaluate the predicted retrieval plan above.")

    return "\n\n".join(lines)


_VERDICT_RE = re.compile(r"^\s*(PASS|FAIL)\s*$", re.MULTILINE | re.IGNORECASE)


def judge_item(item: dict, oai_client: OpenAI) -> dict | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = oai_client.chat.completions.create(
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


def _process_shard(
    shard_id: int,
    items: list[dict],
    oai_client: OpenAI,
    shard_out_path: Path,
) -> dict:
    """한 GPU 샤드의 항목을 처리하고 결과를 shard 파일에 append."""
    written = 0
    failed = 0
    passed = 0
    total_processed = 0
    batch: list[dict] = []

    with open(shard_out_path, "a", encoding="utf-8") as f_out:
        for item in items:
            batch.append(item)

            if len(batch) >= CHUNK_SIZE:
                with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    futures = {executor.submit(judge_item, it, oai_client): it for it in batch}
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
                print(f"  [Shard {shard_id}] Progress: {total_processed}/{len(items)} (pass={passed}, fail={written-passed}, error={failed})")
                batch = []

        # 마지막 남은 batch 처리
        if batch:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {executor.submit(judge_item, it, oai_client): it for it in batch}
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

    print(f"  [Shard {shard_id}] Done. Written: {written}, Failed: {failed}")
    return {"written": written, "failed": failed, "passed": passed}


def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"{IN_PATH} not found. Run 02b_generate_retrieval_steps_hotpotqa.py first.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 최종 출력 + 샤드 파일에서 이미 완료된 ID 수집
    done_ids = load_done_ids(OUT_PATH)
    for sp in [DATA_DIR / "hotpotqa_all_judged_shard0.jsonl",
               DATA_DIR / "hotpotqa_all_judged_shard1.jsonl"]:
        done_ids |= load_done_ids(sp)

    # 입력 로드 (완료된 항목 제외)
    all_items: list[dict] = []
    total_input = 0
    with open(IN_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                total_input += 1
                item = json.loads(line)
                if item["id"] not in done_ids:
                    all_items.append(item)
    print(f"입력 항목: {total_input}개, 기존 완료: {len(done_ids)}개, 남은 처리 대상: {len(all_items)}개")

    if not all_items:
        print("모든 항목이 이미 처리됨. 종료.")
        return

    # 남은 항목을 반으로 나누기
    mid = len(all_items) // 2
    shards = [all_items[:mid], all_items[mid:]]
    ports = [JUDGE_PORT_BASE, JUDGE_PORT_BASE + 1]
    shard_paths = [
        DATA_DIR / "hotpotqa_all_judged_shard0.jsonl",
        DATA_DIR / "hotpotqa_all_judged_shard1.jsonl",
    ]

    # GPU 확보 & vLLM 시작 (병렬)
    procs: list[subprocess.Popen | None] = [None, None]
    clients: list[OpenAI] = []

    if AUTO_VLLM:
        gpus = find_free_gpus(NUM_GPUS)
        vllm_threads: list[threading.Thread] = []
        vllm_errors: list[Exception | None] = [None] * NUM_GPUS

        def _start(i: int):
            try:
                print(f"사용 GPU: {gpus[i]} (port {ports[i]})")
                procs[i] = start_vllm(JUDGE_MODEL_PATH, ports[i], gpus[i])
            except Exception as e:
                vllm_errors[i] = e

        for i in range(NUM_GPUS):
            t = threading.Thread(target=_start, args=(i,))
            vllm_threads.append(t)
            t.start()
        for t in vllm_threads:
            t.join()

        for i in range(NUM_GPUS):
            if vllm_errors[i] is not None:
                # 이미 시작된 서버 정리
                for j in range(NUM_GPUS):
                    stop_vllm(procs[j], ports[j])
                raise vllm_errors[i]
            clients.append(OpenAI(api_key="EMPTY", base_url=f"http://localhost:{ports[i]}/v1"))
    else:
        for i in range(NUM_GPUS):
            url = JUDGE_BASE_URLS[i] if i < len(JUDGE_BASE_URLS) else JUDGE_BASE_URLS[0]
            clients.append(OpenAI(api_key="EMPTY", base_url=url))

    try:
        # 두 샤드를 스레드로 병렬 처리
        results: list[dict] = [{}] * NUM_GPUS
        threads: list[threading.Thread] = []

        def _run_shard(idx: int):
            results[idx] = _process_shard(idx, shards[idx], clients[idx], shard_paths[idx])

        for i in range(NUM_GPUS):
            t = threading.Thread(target=_run_shard, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # 샤드 파일 합치기 → 최종 출력
        print("\n샤드 파일 합치기...")
        seen_ids: set[str] = set()
        total_written = 0

        # 기존 최종 출력 보존
        if OUT_PATH.exists():
            with open(OUT_PATH, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        seen_ids.add(row["id"])
                        total_written += 1

        with open(OUT_PATH, "a", encoding="utf-8") as f_out:
            for sp in shard_paths:
                if sp.exists():
                    with open(sp, encoding="utf-8") as f_in:
                        for line in f_in:
                            line = line.strip()
                            if not line:
                                continue
                            row = json.loads(line)
                            if row["id"] not in seen_ids:
                                f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                                seen_ids.add(row["id"])
                                total_written += 1

        # 샤드 파일 정리
        for sp in shard_paths:
            if sp.exists():
                sp.unlink()
                print(f"  삭제: {sp.name}")

        # 통계
        total_new = sum(r.get("written", 0) for r in results)
        total_failed = sum(r.get("failed", 0) for r in results)
        total_passed = sum(r.get("passed", 0) for r in results)
        print(f"\nDone. New: {total_new}, Failed: {total_failed}")
        print(f"  Judge PASS: {total_passed} ({100*total_passed/max(total_new,1):.1f}%)")
        print(f"  Judge FAIL: {total_new - total_passed} ({100*(total_new-total_passed)/max(total_new,1):.1f}%)")
        print(f"  Total in output: {total_written}")
        print(f"Saved → {OUT_PATH}")

    finally:
        for i in range(NUM_GPUS):
            stop_vllm(procs[i], ports[i])


if __name__ == "__main__":
    main()
