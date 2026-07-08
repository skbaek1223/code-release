"""
Shared helpers for talking to the FAISS retriever HTTP server.

Used by launchers (run_all_datasets.py, run_search_o1_wiki.py) to auto-spawn
the server, and by worker processes (run_re_guide.py, run_search_o1_wiki.py
single-process mode) as an HTTP client.
"""
import os
import subprocess
import sys
import time
from typing import List, Dict, Optional, Tuple

import requests


def server_healthy(url: str, timeout: float = 2.0) -> bool:
    try:
        r = requests.get(f"{url.rstrip('/')}/health", timeout=timeout)
        return r.ok
    except Exception:
        return False


def ensure_retriever_server(
    *,
    retriever_gpus: str,
    host: str,
    port: int,
    retrieval_method: str,
    top_k: int,
    startup_timeout: int,
    log_path: str,
    server_script: str,
) -> Tuple[str, Optional[subprocess.Popen]]:
    """Return (retriever_url, spawned_proc_or_None).

    If a server is already reachable at the target URL, reuse it and return
    proc=None (caller will not kill it on exit). Otherwise spawn
    retriever_server.py on the requested GPUs and wait until /health returns.
    """
    url = f"http://{host}:{port}"
    if server_healthy(url):
        print(f"[retriever] reusing existing server at {url}")
        return url, None

    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log_f = open(log_path, "w", buffering=1)

    cmd = [
        sys.executable, server_script,
        "--gpus", retriever_gpus,
        "--host", host,
        "--port", str(port),
        "--retrieval_method", retrieval_method,
        "--top_k", str(top_k),
    ]
    print(f"[retriever] spawning server on GPUs {retriever_gpus} (log: {log_path})")
    proc = subprocess.Popen(cmd, env=os.environ.copy(), stdout=log_f,
                            stderr=subprocess.STDOUT, start_new_session=True)

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            log_f.close()
            raise SystemExit(
                f"Retriever server exited during startup (code {proc.returncode}). "
                f"See {log_path}")
        if server_healthy(url):
            print(f"[retriever] server ready at {url}")
            return url, proc
        time.sleep(2)

    print(f"[retriever] server did not become healthy within {startup_timeout}s; "
          f"terminating")
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    log_f.close()
    raise SystemExit(f"Retriever server startup timeout. See {log_path}")


def stop_retriever_server(proc: Optional[subprocess.Popen]):
    if proc is None or proc.poll() is not None:
        return
    print(f"[retriever] stopping spawned server (pid={proc.pid})")
    try:
        proc.terminate()
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
    except Exception:
        pass


class RemoteRetriever:
    """HTTP client for retriever_server.py."""

    def __init__(self, url: str, max_doc_len: int = 3000, request_timeout: float = 600):
        self.url = url.rstrip("/")
        self.max_doc_len = max_doc_len
        self.timeout = request_timeout
        r = requests.get(f"{self.url}/health", timeout=10)
        r.raise_for_status()
        print(f"[retriever] client connected to {self.url} ({r.json()})")

    def _format_docs(self, docs) -> List[Dict]:
        out = []
        for i, doc in enumerate(docs):
            text = doc.get("text", doc.get("contents", ""))
            out.append({
                "id": i + 1,
                "title": doc.get("title", ""),
                "text": text[:self.max_doc_len],
            })
        return out

    def search(self, query: str, num: int) -> List[Dict]:
        r = requests.post(f"{self.url}/search",
                          json={"query": query, "top_k": num},
                          timeout=self.timeout)
        r.raise_for_status()
        return self._format_docs(r.json()["docs"])

    def batch_search(self, queries: List[str], num: int) -> List[List[Dict]]:
        if not queries:
            return []
        r = requests.post(f"{self.url}/batch_search",
                          json={"queries": queries, "top_k": num},
                          timeout=self.timeout)
        r.raise_for_status()
        return [self._format_docs(docs) for docs in r.json()["results"]]
