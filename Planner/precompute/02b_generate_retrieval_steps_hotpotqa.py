"""
Step 2b: Generate HotpotQA retrieval steps with Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4

- Input is the question only (no supporting context)
- vLLM OpenAI-compatible API (default port 8000)
- Output: data/precompute/hotpotqa_all_with_steps.jsonl
- Format: <id>\t<step> (one step per line)
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

_raw_urls = os.environ.get("GENERATOR_BASE_URL", "http://localhost:8000/v1")
GENERATOR_BASE_URLS = [u.strip() for u in _raw_urls.split(",") if u.strip()]
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4")
GENERATOR_MODEL_PATH = os.environ.get("GENERATOR_MODEL_PATH", "/mnt/raid6/skbaek1223/models/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4")
GENERATOR_PORT = int(os.environ.get("GENERATOR_PORT", "8000"))
AUTO_VLLM = os.environ.get("AUTO_VLLM", "1") == "1"
MIN_FREE_MB = int(os.environ.get("MIN_FREE_MB", "15000"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "32"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "2000"))
MAX_TOKENS = 256

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "precompute"
IN_PATH = DATA_DIR / "hotpotqa_all.jsonl"
OUT_PATH = DATA_DIR / "hotpotqa_all_with_steps.jsonl"

_clients: list[OpenAI] = []
_client_counter = 0

SYSTEM_PROMPT = """You are an information retrieval planning expert. Given a multi-hop question, generate an ordered sequence of retrieval steps required to find the answer. Each step represents one concrete retrieval action.

Write one step per line, numbered. No other text.

Example 1 (comparison):
Question: Which magazine was started first Arthur's Magazine or First for Women?
1. Find the start date of Arthur's Magazine.
2. Find the start date of First for Women.

Example 2 (bridge):
Question: The Oberoi family is part of a hotel company that has a head office in what city?
1. Find which hotel company the Oberoi family is part of.
2. Find the city where that hotel company has its head office."""

_STEP_RE = re.compile(r"^\s*\d+[\.\)]\s*(.+)", re.MULTILINE)


# ──────────────────────────────────────────────
# GPU / vLLM management
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
            f"Not enough free GPUs: need {n} (>= {min_free_mb} MB free), found {len(gpus)}"
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
        print(f"[port {port}] vLLM server already running — reusing it.")
        return None

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env.setdefault("VLLM_ENGINE_READY_TIMEOUT_S", "1200")
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env.setdefault("NCCL_IB_DISABLE", "1")

    log_path = f"/tmp/vllm_{port}_stderr.log"
    print(f"[port {port}] Starting vLLM... (model={Path(model_path).name}, GPU={gpu_id})")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--served-model-name", GENERATOR_MODEL,
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
            raise RuntimeError(f"[port {port}] vLLM process exited unexpectedly. Log: {log_path}")
        if _ping(base_url):
            print(f"[port {port}] vLLM ready. (GPU {gpu_id})")
            return proc

    proc.terminate()
    raise RuntimeError(f"[port {port}] vLLM startup timed out ({timeout}s). Log: {log_path}")


def stop_vllm(proc: subprocess.Popen | None, port: int):
    if proc is None:
        return
    print(f"[port {port}] Stopping vLLM...")
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


# ──────────────────────────────────────────────

def _get_client() -> OpenAI:
    global _client_counter
    c = _clients[_client_counter % len(_clients)]
    _client_counter += 1
    return c


def generate_steps(item: dict) -> dict | None:
    prompt = f"Question: {item['question']}"
    try:
        response = _get_client().chat.completions.create(
            model=GENERATOR_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        content = response.choices[0].message.content.strip()
        steps = [m.group(1).strip() for m in _STEP_RE.finditer(content)]
        if not steps:
            print(f"  SKIP {item['id']}: no steps found")
            return None
        return {**item, "predicted_steps": steps}
    except Exception as e:
        print(f"  ERROR {item['id']}: {e}")
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
    global _clients

    proc = None
    if AUTO_VLLM:
        gpus = find_free_gpus(1)
        print(f"Using GPU: {gpus[0]} (port {GENERATOR_PORT})")
        proc = start_vllm(GENERATOR_MODEL_PATH, GENERATOR_PORT, gpus[0])
        _clients = [OpenAI(api_key="EMPTY", base_url=f"http://localhost:{GENERATOR_PORT}/v1")]
    else:
        _clients = [OpenAI(api_key="EMPTY", base_url=url) for url in GENERATOR_BASE_URLS]

    print(f"Using {len(_clients)} generator server(s)")

    try:
        _main_process()
    finally:
        stop_vllm(proc, GENERATOR_PORT)


def _main_process():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"{IN_PATH} not found. Run 01_b_load_hotpotqa.py first.")

    done_ids = load_done_ids(OUT_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    failed = 0
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
                continue
            batch.append(item)

            if len(batch) >= CHUNK_SIZE:
                with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                    futures = {executor.submit(generate_steps, it): it for it in batch}
                    for future in as_completed(futures):
                        result = future.result()
                        if result is not None:
                            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f_out.flush()
                            written += 1
                        else:
                            failed += 1
                total_processed += len(batch)
                print(f"  Progress: {total_processed} (written={written}, failed={failed})")
                batch = []

        # Process the final remaining batch
        if batch:
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {executor.submit(generate_steps, it): it for it in batch}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f_out.flush()
                        written += 1
                    else:
                        failed += 1
            total_processed += len(batch)

    print(f"\nDone. Written: {written}, Failed: {failed}")
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
