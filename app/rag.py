from llama_cpp import Llama
from app.retriever import HybridRetriever

class RAGPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.llm = Llama(
            model_path=cfg["model"]["gguf_path"],
            n_ctx=cfg["model"]["ctx_tokens"],
            n_threads=cfg["model"]["n_threads"],
            n_gpu_layers=cfg["model"]["n_gpu_layers"],
        )
        self.retriever = HybridRetriever(cfg)

    def query(self, question: str):
        ctxs = self.retriever.search(question,
                                     self.cfg["retrieval"]["top_k"],
                                     self.cfg["retrieval"]["rerank_k"],
                                     self.cfg["retrieval"]["hybrid_alpha"])
        context_text = "\n\n".join([f"[{i}] {c[:400]}" for i, c in enumerate(ctxs)])
        prompt = (
            "You are a helpful assistant.\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\nAnswer:"
        )

        prompt_tokens = len(prompt.split())
        ctx_limit = self.cfg["model"]["ctx_tokens"]
        safe_max_output = max(64, min(256, ctx_limit - prompt_tokens - 10))
        print(f"🧩 Prompt length: {prompt_tokens} tokens (limit {ctx_limit})")

        try:
            response = self.llm(
                prompt,
                max_tokens=safe_max_output,
                temperature=self.cfg["model"]["temperature"],
                top_p=self.cfg["model"]["top_p"],
                repeat_penalty=self.cfg["model"]["repeat_penalty"],
            )
            return response["choices"][0]["text"].strip()
        except Exception as e:
            return f"❌ LLM error: {e}"
