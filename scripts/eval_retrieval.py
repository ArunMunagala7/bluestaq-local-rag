#!/usr/bin/env python3
"""
Simple retrieval evaluation script comparing BM25, dense (FAISS), and hybrid.

Run from repo root:
    python3 scripts/eval_retrieval.py

Requires the project environment and that `data/index/` contains the built
BM25/FAISS/docs pickles that the repo's `HybridRetriever` expects.
This script is intentionally small and additive (does not change app code).
"""
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"
EVAL_PATH = ROOT / "data/eval/eval_queries.jsonl"

def load_config(p):
    import yaml
    return yaml.safe_load(open(p, "r"))

def load_eval(p):
    qs = []
    with open(p, "r") as f:
        for line in f:
            if line.strip():
                qs.append(json.loads(line))
    return qs

def safe_import_retriever(cfg):
    # Try to import the project's HybridRetriever; otherwise return None
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from app.retriever import HybridRetriever
        return HybridRetriever(cfg)
    except Exception as e:
        print("Could not load HybridRetriever:", e)
        return None

def recall_at_k(retrieved_ids, gold_id_set, k):
    ret = retrieved_ids[:k]
    return int(bool(set(ret) & gold_id_set))

def mrr(retrieved_ids, gold_id_set):
    for i, docid in enumerate(retrieved_ids, start=1):
        if docid in gold_id_set:
            return 1.0 / i
    return 0.0

def run_eval():
    cfg = load_config(CONFIG_PATH)
    queries = load_eval(EVAL_PATH)
    retriever = safe_import_retriever(cfg)
    if retriever is None:
        print("HybridRetriever not available; ensure dependencies and index exist.")
        return

    # We'll use top_k from config and evaluate at k={1,3,5}
    top_k = cfg.get("retrieval", {}).get("top_k", 3)
    eval_ks = [1, 3, 5]

    metrics = {
        "bm25": {k: [] for k in eval_ks},
        "dense": {k: [] for k in eval_ks},
        "hybrid": {k: [] for k in eval_ks},
        "mrr": {"bm25": [], "dense": [], "hybrid": []},
    }

    # For this small eval, we'll treat the index document positions as IDs (0-based)
    for q in queries:
        query_text = q["query"]
        # gold_snippet matching: naive - check which docs contain the gold snippet
        gold = q.get("gold_snippet", "").lower()

        # run the retriever with different modes by varying alpha
        # bm25 only: alpha=0.0 (only sparse), dense only: alpha=1.0, hybrid: config alpha
        bm25_results = retriever.search(query_text, top_k=top_k, rerank_k=cfg["retrieval"].get("rerank_k", 3), alpha=0.0)
        dense_results = retriever.search(query_text, top_k=top_k, rerank_k=cfg["retrieval"].get("rerank_k", 3), alpha=1.0)
        hybrid_results = retriever.search(query_text, top_k=top_k, rerank_k=cfg["retrieval"].get("rerank_k", 3), alpha=cfg["retrieval"].get("hybrid_alpha", 0.65))

        def extract_ids(results):
            ids = []
            for r in results:
                # attempt to map text back to index id via substring match on docs list
                # fallback: use hash of text (not ideal) — but HybridRetriever returns docs list accessibly
                try:
                    idx = retriever.docs.index(r["text"]) if r["text"] in retriever.docs else None
                except Exception:
                    idx = None
                if idx is None:
                    # fallback to hashed pseudo-id
                    ids.append(hash(r["text"]))
                else:
                    ids.append(idx)
            return ids

        bm25_ids = extract_ids(bm25_results)
        dense_ids = extract_ids(dense_results)
        hybrid_ids = extract_ids(hybrid_results)

        # compute gold id set by searching docs for the gold snippet
        gold_ids = set()
        if gold:
            for i, d in enumerate(retriever.docs):
                if gold in d.lower():
                    gold_ids.add(i)

        # If no gold ids found, try fuzzy match on answer string
        if not gold_ids and q.get("answer"):
            a = q["answer"].lower()
            for i, d in enumerate(retriever.docs):
                if a.split()[0] in d.lower():
                    gold_ids.add(i)

        # If still empty, we won't count recall, but will still record MRR=0
        for k in eval_ks:
            metrics["bm25"][k].append(recall_at_k(bm25_ids, gold_ids, k))
            metrics["dense"][k].append(recall_at_k(dense_ids, gold_ids, k))
            metrics["hybrid"][k].append(recall_at_k(hybrid_ids, gold_ids, k))

        metrics["mrr"]["bm25"].append(mrr(bm25_ids, gold_ids))
        metrics["mrr"]["dense"].append(mrr(dense_ids, gold_ids))
        metrics["mrr"]["hybrid"].append(mrr(hybrid_ids, gold_ids))

    # Aggregate and print
    def mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    print("Retrieval evaluation results:")
    for k in eval_ks:
        print(f"Recall@{k} — BM25: {mean(metrics['bm25'][k]):.3f}, Dense: {mean(metrics['dense'][k]):.3f}, Hybrid: {mean(metrics['hybrid'][k]):.3f}")
    print(f"MRR — BM25: {mean(metrics['mrr']['bm25']):.3f}, Dense: {mean(metrics['mrr']['dense']):.3f}, Hybrid: {mean(metrics['mrr']['hybrid']):.3f}")

    # Save detailed outputs
    out_path = ROOT / "data/eval/retrieval_eval_results.json"
    with open(out_path, "w") as f:
        json.dump({"metrics": metrics}, f, indent=2)
    print("Saved results to:", out_path)

if __name__ == '__main__':
    run_eval()
