import os, faiss, numpy as np, re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
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
        
        # Initialize reranker if enabled
        self.use_reranker = cfg.get("retrieval", {}).get("use_reranker", False)
        self.reranker = None
        if self.use_reranker:
            rerank_model = cfg.get("retrieval", {}).get("rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.reranker = CrossEncoder(rerank_model)
        
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

    def search(self, query, top_k=3, rerank_k=3, alpha=0.65, explain=False):
        if not self.faiss_index or not self.bm25: 
            return []
        qv = self.embedder.encode([query], normalize_embeddings=True)
        
        # Initial retrieval: get more candidates for reranking
        initial_k = top_k * 3 if self.use_reranker else top_k
        D, I = self.faiss_index.search(qv, initial_k)
        dense_idx = I[0]; dense_scores = D[0]
        bm_scores = self.bm25.get_scores(query.split())
        bm_top = np.argpartition(-bm_scores, min(rerank_k * 2, len(bm_scores)-1))[:rerank_k * 2]
        cand = list(set(dense_idx.tolist()) | set(bm_top.tolist()))
        
        # Hybrid fusion
        fused = []
        for i in cand:
            dense = float(dense_scores[list(dense_idx).index(i)]) if i in dense_idx else 0
            sparse = float(bm_scores[i]) if i < len(bm_scores) else 0
            score = alpha * dense + (1-alpha) * sparse
            fused.append((i, score))
        fused = sorted(fused, key=lambda x: x[1], reverse=True)[:initial_k]
        
        # Reranking phase if enabled
        if self.use_reranker and self.reranker:
            # Prepare pairs for cross-encoder
            pairs = [[query, self.docs[i]] for i, _ in fused]
            rerank_scores = self.reranker.predict(pairs)
            # Combine with original score for stability
            fused_reranked = []
            for (i, orig_score), rerank_score in zip(fused, rerank_scores):
                # Weight: 70% reranker, 30% original fusion
                final_score = 0.7 * float(rerank_score) + 0.3 * orig_score
                fused_reranked.append((i, final_score, float(rerank_score), orig_score))
            fused_reranked = sorted(fused_reranked, key=lambda x: x[1], reverse=True)[:top_k]
            final_results = [(i, final_score, rerank_score, orig_score) for i, final_score, rerank_score, orig_score in fused_reranked]
        else:
            final_results = [(i, score, None, score) for i, score in fused[:top_k]]
        
        results = []
        for item_data in final_results:
            i = item_data[0]
            final_score = item_data[1]
            rerank_score = item_data[2]
            orig_score = item_data[3]
            
            item = {
                "text": self.docs[i],
                "score": final_score,
                "title": self.meta[i].get("title", "Unknown") if i < len(self.meta) else "Unknown"
            }

            if explain:
                # Dense explanation: top matching sentences
                try:
                    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", self.docs[i]) if s.strip()]
                except Exception:
                    sentences = [self.docs[i]]
                # Limit sentences to avoid heavy compute
                max_sent = 12
                sent_sample = sentences[:max_sent]
                if sent_sample:
                    sent_emb = self.embedder.encode(sent_sample, normalize_embeddings=True)
                    qvec = qv[0]
                    # cosine since normalized
                    cos = np.dot(sent_emb, qvec)
                    top_idx = np.argsort(-cos)[:2].tolist()
                    top_spans = [{"text": sent_sample[j], "cos": float(cos[j])} for j in top_idx]
                    dense_explain = {"cos": float(dense if i in dense_idx else 0.0), "top_spans": top_spans}
                else:
                    dense_explain = {"cos": float(dense if i in dense_idx else 0.0), "top_spans": []}

                # Sparse explanation: top contributing terms
                terms = list(dict.fromkeys(query.split()))
                idf_map = getattr(self.bm25, "idf", {}) or {}
                # Try to get document tokens from BM25 corpus if available
                try:
                    doc_tokens = self.bm25.corpus[i]
                except Exception:
                    doc_tokens = self.docs[i].split()
                term_explain = []
                for t in terms:
                    tf = doc_tokens.count(t)
                    if tf <= 0:
                        continue
                    idf = float(idf_map.get(t, 0.0))
                    tfidf = tf * idf
                    term_explain.append({"t": t, "tf": tf, "idf": idf, "tfidf": tfidf})
                term_explain = sorted(term_explain, key=lambda x: x["tfidf"], reverse=True)[:3]

                item["explain"] = {
                    "dense": dense_explain,
                    "sparse": {
                        "bm25": float(bm_scores[i]) if i < len(bm_scores) else 0.0,
                        "terms": term_explain,
                    },
                    "fusion": {
                        "alpha": float(alpha),
                        "dense": float(dense_scores[list(dense_idx).index(i)]) if i in dense_idx else 0.0,
                        "bm25": float(bm_scores[i]) if i < len(bm_scores) else 0.0,
                        "fused": float(orig_score),
                    },
                    "rerank": {
                        "enabled": self.use_reranker,
                        "score": float(rerank_score) if rerank_score is not None else None,
                        "final": float(final_score)
                    }
                }

            results.append(item)
        return results
