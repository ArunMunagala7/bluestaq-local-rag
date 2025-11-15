import typer, os, glob
from rich.console import Console
from app.config import CFG
from app.rag import RAGPipeline
from app.ingest import ingest_corpus, process_file

app = typer.Typer()
console = Console()

# ------------------------------------------------------------------
# BASIC: direct question → LLM only
# ------------------------------------------------------------------
@app.command("query-basic")
def query_basic(question: str):
    from llama_cpp import Llama
    llm = Llama(
        model_path=CFG["model"]["gguf_path"],
        n_ctx=CFG["model"]["ctx_tokens"],
        n_threads=CFG["model"]["n_threads"],
        n_gpu_layers=CFG["model"]["n_gpu_layers"],
        verbose=False,
    )
    console.print(f"💬 [cyan]Question:[/cyan] {question}")
    res = llm(
        f"Question: {question}\nAnswer:",
        max_tokens=256,
        temperature=CFG["model"]["temperature"],
        top_p=CFG["model"]["top_p"],
        repeat_penalty=CFG["model"]["repeat_penalty"],
    )
    console.print("\n🤖 [green]Answer:[/green] " + res["choices"][0]["text"].strip())


# ------------------------------------------------------------------
# RAG: question + retrieval
# ------------------------------------------------------------------
@app.command("query-rag")
def query_rag(question: str):
    rag = RAGPipeline(CFG)
    result = rag.query(question)
    ans = result["answer"]
    sources = result["sources"]
    
    console.print("\n🤖 [green]Answer:[/green] " + ans)
    
    if sources:
        console.print("\n📚 [cyan]Retrieved Sources:[/cyan]")
        for i, src in enumerate(sources, 1):
            score_pct = min(100, max(0, int(src["score"] * 100)))
            console.print(f"\n  [{i}] 📄 {src['title']} (relevance: {score_pct}%)")
            console.print(f"      " + "="*70)
            console.print(f"      {src['text_full'][:500]}..." if len(src['text_full']) > 500 else f"      {src['text_full']}")
            console.print(f"      " + "="*70)


# ------------------------------------------------------------------
# Upload one file to corpus
# ------------------------------------------------------------------
@app.command("upload")
def upload_file(path: str):
    process_file(path, CFG)
    console.print("✅ File processed and added to corpus.")


# ------------------------------------------------------------------
# Bulk upload all files in data/uploads
# ------------------------------------------------------------------
@app.command("bulk-upload")
def bulk_upload():
    files = glob.glob("data/uploads/*.*")
    console.print(f"📂 Found {len(files)} file(s) to upload:")
    for f in files:
        console.print(f"• {os.path.basename(f)}")
        try:
            process_file(f, CFG)
        except Exception as e:
            console.print(f"❌ Error processing {f}: {e}")
    console.print("✅ All uploads processed. Rebuilding index...")
    ingest_corpus(CFG)


# ------------------------------------------------------------------
# Chat mode
# ------------------------------------------------------------------
@app.command("chat")
def chat():
    rag = RAGPipeline(CFG)
    console.print("💬 Chat mode started! Type '/exit' to quit.")
    while True:
        q = input("\nYou: ")
        if q.strip().lower() == "/exit":
            break
        result = rag.query(q)
        ans = result["answer"]
        sources = result["sources"]
        console.print(f"Assistant: {ans}")
        if sources:
            console.print("\n  📚 [dim]Sources used:[/dim]")
            for src in sources:
                score_pct = min(100, max(0, int(src["score"] * 100)))
                console.print(f"    • {src['title']} ({score_pct}%)")


if __name__ == "__main__":
    app()
