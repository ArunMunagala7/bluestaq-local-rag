import typer, os, glob
import math
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
def query_rag(
    question: str,
    clarify: bool = typer.Option(False, "--clarify", "-c", help="Force a short clarifying question before answering"),
    justify: bool = typer.Option(False, "--justify", help="Show retrieval justifications (why each source was selected)"),
    style: str = typer.Option(
        "auto",
        "--style",
        "-s",
        help="Answer style: auto|concise|detailed|bullet|code (auto uses interactive clarifier when needed)",
    ),
):
    # determine prompt suffix from style or interactive clarifier
    style_templates = {
        "concise": " Provide a concise 1-2 sentence answer.",
        "detailed": " Provide a detailed answer with examples and explanations.",
        "bullet": " Provide the answer as a short bullet list.",
        "code": " Provide an example code snippet if applicable and explain briefly.",
    }

    prompt_suffix = ""

    # If style is explicitly set and not 'auto', use it directly
    if style != "auto":
        if style not in style_templates:
            console.print(f"⚠️ Unknown style '{style}', defaulting to 'concise'.")
            prompt_suffix = style_templates["concise"]
        else:
            prompt_suffix = style_templates[style]
    else:
        # If style is 'auto', ask user interactively for style choice
        try:
            print("Available answer styles:")
            for key in style_templates:
                print(f"  - {key}")
            choice = input("Which style do you want? [concise/detailed/bullet/code]: ").strip().lower()
            if choice in style_templates:
                prompt_suffix = style_templates[choice]
            else:
                console.print(f"⚠️ Unknown style '{choice}', defaulting to 'concise'.")
                prompt_suffix = style_templates["concise"]
        except Exception:
            # non-interactive environments: fall back to concise
            prompt_suffix = style_templates["concise"]

    followup_suffix = ""
    # Build prompt suffix for generation, but keep retrieval on raw user question
    gen_suffix = prompt_suffix

    rag = RAGPipeline(CFG)
    result = rag.query(question, gen_suffix, explain=justify)

    ans = result["answer"]
    sources = result["sources"]
    evidence_map = result.get("evidence_map", [])
    has_external = result.get("has_external_knowledge", False)
    uncited_warning = result.get("uncited_warning", False)
    followup_questions = result.get("followup_questions", "")
    
    console.print(f"\n🤖 [green]Answer:[/green] {ans}")
    
    # Show follow-up questions if generated
    if followup_questions:
        console.print(f"\n💡 [cyan]Follow-up questions:[/cyan]\n{followup_questions}")
    
    # Show warnings if external knowledge or uncited claims detected
    if has_external:
        console.print("\n⚠️  [yellow]Note:[/yellow] This answer contains information marked as [External Knowledge] - not from provided sources.")
    if uncited_warning:
        console.print("\n⚠️  [yellow]Warning:[/yellow] Some claims in this answer may not be properly cited. Verify against sources below.")
    
    # Display evidence map if citations were found
    if evidence_map:
        console.print("\n  📋 [cyan]Evidence Map:[/cyan]")
        for ev in evidence_map:
            console.print(f"    [Source {ev['id']}] {ev['source_title']}")
            console.print(f"    ↳ \"{ev['span']}\"")
            console.print()
    
    if sources:
        console.print("\n  📚 [dim]Sources used:[/dim]")
        scores = [float(s.get("score", 0.0)) for s in sources]
        if scores:
            m = max(scores)
            exps = [math.exp(s - m) for s in scores]
            ssum = sum(exps) if sum(exps) != 0 else 1.0
            probs = [e / ssum for e in exps]
        else:
            probs = [0.0 for _ in scores]

        for i, (src, p) in enumerate(zip(sources, probs), 1):
            score_pct = int(max(0, min(100, round(p * 100))))
            console.print(f"  [{i}] 📄 {src['title']} (relevance: {score_pct}%)")
            console.print(f"      " + "="*70)
            chunk = src.get('text_full', src.get('text', ''))
            console.print(f"      {chunk[:500]}..." if len(chunk) > 500 else f"      {chunk}")
            console.print(f"      " + "="*70)
            if justify:
                exp = src.get('explain', {}) or {}
                if exp:
                    dense = exp.get('dense', {})
                    fusion = exp.get('fusion', {})
                    sparse = exp.get('sparse', {})
                    # One-line why
                    why = ''
                    spans = dense.get('top_spans', [])
                    if spans:
                        why = spans[0].get('text', '')
                    console.print("      Why selected:")
                    if why:
                        console.print(f"        • {why[:120]}{'...' if len(why)>120 else ''}")
                    # Numbers
                    console.print("      Scores:")
                    rerank_info = exp.get('rerank', {})
                    if rerank_info.get('enabled'):
                        console.print(f"        • Dense: {fusion.get('dense', 0):.3f} | BM25: {fusion.get('bm25', 0):.3f} | α: {fusion.get('alpha', 0):.2f} | Fused: {fusion.get('fused', 0):.3f}")
                        console.print(f"        • Rerank: {rerank_info.get('score', 0):.3f} | Final: {rerank_info.get('final', 0):.3f}")
                    else:
                        console.print(f"        • Dense: {fusion.get('dense', 0):.3f} | BM25: {fusion.get('bm25', 0):.3f} | α: {fusion.get('alpha', 0):.2f} | Fused: {fusion.get('fused', 0):.3f}")
                    # Terms
                    terms = sparse.get('terms', [])
                    if terms:
                        top_terms = ", ".join([f"{t['t']} (tf {t['tf']}, idf {t['idf']:.2f})" for t in terms])
                        console.print(f"      Top terms: {top_terms}")


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
    
    # Answer style templates
    style_templates = {
        "concise": " Provide a concise 1-2 sentence answer.",
        "detailed": " Provide a detailed answer with examples and explanations.",
        "bullet": " Provide the answer as a short bullet list.",
        "code": " Provide an example code snippet if applicable and explain briefly.",
    }
    
    # Ask user for preferred style at the start of chat session
    try:
        print("\nAvailable answer styles:")
        for key in style_templates:
            print(f"  - {key}")
        style_choice = input("Which style do you want for this chat session? [concise/detailed/bullet/code]: ").strip().lower()
        if style_choice in style_templates:
            prompt_suffix = style_templates[style_choice]
        else:
            console.print(f"⚠️ Unknown style '{style_choice}', defaulting to 'concise'.")
            prompt_suffix = style_templates["concise"]
    except Exception:
        # non-interactive environments: fall back to concise
        prompt_suffix = style_templates["concise"]

    # Ask if justifications should be shown in this session
    try:
        j = input("Show retrieval justifications for sources? [y/N]: ").strip().lower()
        justify_session = j == 'y'
    except Exception:
        justify_session = False
    
    
    while True:
        q = input("\nYou: ")
        if q.strip().lower() == "/exit":
            break
        
        # Build prompt suffix for generation; keep retrieval on raw user input
        gen_suffix = prompt_suffix
        result = rag.query(q, gen_suffix, explain=justify_session)
        ans = result["answer"]
        sources = result["sources"]
        evidence_map = result.get("evidence_map", [])
        has_external = result.get("has_external_knowledge", False)
        uncited_warning = result.get("uncited_warning", False)
        followup_questions = result.get("followup_questions", "")
        
        console.print(f"Assistant: {ans}")
        
        # Show follow-up questions if generated
        if followup_questions:
            console.print(f"\n💡 [cyan]Follow-up questions:[/cyan]\n{followup_questions}")
        
        # Show warnings if external knowledge or uncited claims detected
        if has_external:
            console.print("\n⚠️  [yellow]Note:[/yellow] This answer contains information marked as [External Knowledge] - not from provided sources.")
        if uncited_warning:
            console.print("\n⚠️  [yellow]Warning:[/yellow] Some claims may not be properly cited. Verify against sources below.")
        
        # Display evidence map if citations were found
        if evidence_map:
            console.print("\n  📋 [cyan]Evidence Map:[/cyan]")
            for ev in evidence_map:
                console.print(f"    [Source {ev['id']}] {ev['source_title']}")
                console.print(f"    ↳ \"{ev['span']}\"")
                console.print()
        
        if sources:
            console.print("\n  📚 [dim]Sources used:[/dim]")
            # normalize per-query scores for chat mode as well
            scores = [float(s.get("score", 0.0)) for s in sources]
            if scores:
                m = max(scores)
                exps = [math.exp(s - m) for s in scores]
                ssum = sum(exps) if sum(exps) != 0 else 1.0
                probs = [e / ssum for e in exps]
            else:
                probs = [0.0 for _ in scores]

            for src, p in zip(sources, probs):
                score_pct = int(max(0, min(100, round(p * 100))))
                console.print(f"    • {src['title']} ({score_pct}%)")
                console.print(f"      " + "="*70)
                chunk = src.get('text_full', src.get('text', ''))
                console.print(f"      {chunk[:500]}..." if len(chunk) > 500 else f"      {chunk}")
                console.print(f"      " + "="*70)
                if justify_session:
                    exp = src.get('explain', {}) or {}
                    if exp:
                        dense = exp.get('dense', {})
                        fusion = exp.get('fusion', {})
                        sparse = exp.get('sparse', {})
                        why = ''
                        spans = dense.get('top_spans', [])
                        if spans:
                            why = spans[0].get('text', '')
                        console.print("      Why selected:")
                        if why:
                            console.print(f"        • {why[:120]}{'...' if len(why)>120 else ''}")
                        console.print("      Scores:")
                        rerank_info = exp.get('rerank', {})
                        if rerank_info.get('enabled'):
                            console.print(f"        • Dense: {fusion.get('dense', 0):.3f} | BM25: {fusion.get('bm25', 0):.3f} | α: {fusion.get('alpha', 0):.2f} | Fused: {fusion.get('fused', 0):.3f}")
                            console.print(f"        • Rerank: {rerank_info.get('score', 0):.3f} | Final: {rerank_info.get('final', 0):.3f}")
                        else:
                            console.print(f"        • Dense: {fusion.get('dense', 0):.3f} | BM25: {fusion.get('bm25', 0):.3f} | α: {fusion.get('alpha', 0):.2f} | Fused: {fusion.get('fused', 0):.3f}")
                        terms = sparse.get('terms', [])
                        if terms:
                            top_terms = ", ".join([f"{t['t']} (tf {t['tf']}, idf {t['idf']:.2f})" for t in terms])
                            console.print(f"      Top terms: {top_terms}")


if __name__ == "__main__":
    app()
