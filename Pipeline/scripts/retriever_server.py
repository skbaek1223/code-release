"""
FAISS retriever HTTP server (FastAPI).

Loads the FAISS index and encoder once on dedicated GPUs and serves search
requests so callers share a single retriever process. Decouples FAISS from
vLLM GPUs so vLLM can be loaded once and never swapped.

Usage:
    python retriever_server.py --gpus 0,1 --port 8765
    curl http://127.0.0.1:8765/health
"""
import argparse
import os
from typing import List, Optional


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=str, default="0,1",
                   help="Comma-separated GPU ids for FAISS + encoder (sharded).")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--retrieval_method", type=str, default="e5",
                   choices=["e5", "bm25"])
    p.add_argument("--index_path", type=str,
                   default="/mnt/raid6/skbaek1223/project/FlashRAG/retrieval_corpus/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/e5_flat_inner.index")
    p.add_argument("--corpus_path", type=str,
                   default="/mnt/raid6/skbaek1223/project/FlashRAG/retrieval_corpus/wiki18_100w.jsonl")
    p.add_argument("--retrieval_model_path", type=str, default="intfloat/e5-base-v2")
    p.add_argument("--retrieval_pooling_method", type=str, default="mean")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    from flashrag.retriever.retriever import DenseRetriever, BM25Retriever
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    print(f"[retriever-server] method={args.retrieval_method} gpus={args.gpus} "
          f"index={args.index_path}", flush=True)

    config = {
        "retrieval_method": args.retrieval_method,
        "retrieval_topk": args.top_k,
        "index_path": args.index_path,
        "corpus_path": args.corpus_path,
        "retrieval_model_path": args.retrieval_model_path,
        "retrieval_query_max_length": 128,
        "retrieval_pooling_method": args.retrieval_pooling_method,
        "retrieval_use_fp16": True,
        "retrieval_batch_size": args.batch_size,
        "faiss_gpu": args.retrieval_method != "bm25",
        "save_retrieval_cache": False,
        "use_retrieval_cache": False,
        "retrieval_cache_path": None,
        "silent_retrieval": True,
        "use_reranker": False,
        "instruction": None,
        "use_sentence_transformer": False,
    }

    if args.retrieval_method == "bm25":
        config["bm25_backend"] = "bm25s"
        retriever = BM25Retriever(config)
    else:
        retriever = DenseRetriever(config)
    print("[retriever-server] retriever ready", flush=True)

    app = FastAPI()

    class SearchRequest(BaseModel):
        query: str
        top_k: Optional[int] = None

    class BatchSearchRequest(BaseModel):
        queries: List[str]
        top_k: Optional[int] = None

    @app.get("/health")
    def health():
        return {"status": "ok", "method": args.retrieval_method}

    @app.post("/search")
    def search(req: SearchRequest):
        k = req.top_k or args.top_k
        docs = retriever.search(req.query, num=k)
        return {"docs": docs}

    @app.post("/batch_search")
    def batch_search(req: BatchSearchRequest):
        k = req.top_k or args.top_k
        if not req.queries:
            return {"results": []}
        results = retriever.batch_search(req.queries, num=k)
        return {"results": results}

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
