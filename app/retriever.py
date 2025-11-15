import os, faiss, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import pickle

class HybridRetriever:
    def __init__(self, cfg):
        self.cfg = cfg
        self.index_dir = cfg["paths"]["index_dir"]
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.bm25 = None
        self.faiss_index = None
        self.docs = []
        self.meta = []
        self._load_indexes()

    def _load_indexes(self):
        bm25_path = os.path.join(self.index_dir, "bm25.pkl")
        faiss_path = os.path.join(self.index_dir, "faiss.index")
        docs_path = os.path.join(self.index_dir, "docs.pkl")
        meta_path = os.path.join(self.index_dir, "meta.pkl")
        if os.path.exists(bm25_path):
            self.bm25 = pickle.load(open(bm25_path, "rb"))
        if os.path.exists(faiss_path):
            self.faiss_index = faiss.read_index(faiss_path)
        if os.path.exists(docs_path):
            self.docs = pickle.load(open(docs_path, "rb"))
        if os.path.exists(meta_path):
            self.meta = pickle.load(open(meta_path, "rb"))
        else:
            self.meta = [{"title": "Unknown"} for _ in self.docs]

    def search(self, query, top_k=3, rerank_k=3, alpha=0.65):
        if not self.faiss_index or not self.bm25: 
            return []
        qv = self.embedder.encode([query], normalize_embeddings=True)
        D, I = self.faiss_index.search(qv, top_k)
        dense_idx = I[0]; dense_scores = D[0]
        bm_scores = self.bm25.get_scores(query.split())
        bm_top = np.argpartition(-bm_scores, min(rerank_k, len(bm_scores)-1))[:rerank_k]
        cand = list(set(dense_idx.tolist()) | set(bm_top.tolist()))
        fused = []
        for i in cand:
            dense = float(dense_scores[list(dense_idx).index(i)]) if i in dense_idx else 0
            sparse = float(bm_scores[i]) if i < len(bm_scores) else 0
            score = alpha * dense + (1-alpha) * sparse
            fused.append((i, score))
        fused = sorted(fused, key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for i, score in fused:
            results.append({
                "text": self.docs[i],
                "score": score,
                "title": self.meta[i].get("title", "Unknown") if i < len(self.meta) else "Unknown"
            })
        return results
