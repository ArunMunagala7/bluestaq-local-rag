import typer, os, glob
import math
import json
from datetime import datetime
from pathlib import Path
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
    save: bool = typer.Option(False, "--save", help="Save query results to data/query_history.jsonl"),
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
            console.print(f"⚠️ Unknown style '{style}', defaulting to 'detailed'.")
            prompt_suffix = style_templates["detailed"]
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
                console.print(f"⚠️ Unknown style '{choice}', defaulting to 'detailed'.")
                prompt_suffix = style_templates["detailed"]
        except Exception:
            # non-interactive environments: fall back to detailed
            prompt_suffix = style_templates["detailed"]

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
                    
                    # LLM-generated reasoning (if available)
                    llm_reasoning = exp.get('llm_reasoning', '')
                    if llm_reasoning:
                        console.print("      [cyan]LLM Reasoning:[/cyan]")
                        console.print(f"        💡 {llm_reasoning}")
                    
                    # Vector-based why (top matching sentence)
                    why = ''
                    spans = dense.get('top_spans', [])
                    if spans:
                        why = spans[0].get('text', '')
                    console.print("      [dim]Vector Match:[/dim]")
                    if why:
                        console.print(f"        • {why[:120]}{'...' if len(why)>120 else ''}")
                    else:
                        console.print(f"        • (No specific sentence match)")
                    
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
    
    # Save query results if --save flag is used
    if save:
        save_query_result(question, ans, sources, evidence_map, followup_questions, style)
        console.print("\n💾 [green]Query saved to data/query_history.jsonl[/green]")
    
    # Interactive follow-up selection (only in query-rag mode)
    if followup_questions:
        # Parse follow-up questions into a list
        followup_lines = [line.strip() for line in followup_questions.split('\n') if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-'))]
        if followup_lines:
            while True:
                try:
                    choice = input("\n💬 Ask a follow-up? [1-3 to select, or type your own question, Enter to exit]: ").strip()
                    if not choice:
                        # User pressed Enter, exit
                        break
                    elif choice in ['1', '2', '3']:
                        # User selected a numbered follow-up
                        idx = int(choice) - 1
                        if idx < len(followup_lines):
                            # Extract the question text (remove number prefix like "1. ")
                            followup_q = followup_lines[idx]
                            followup_q = followup_q.lstrip('0123456789.-) ').strip()
                            console.print(f"\n[dim]→ Running follow-up:[/dim] {followup_q}\n")
                            # Recursively call query_rag with the follow-up question
                            query_rag(followup_q, clarify=clarify, justify=justify, style=style, save=save)
                            break
                        else:
                            console.print(f"⚠️ Invalid selection. Choose 1-{len(followup_lines)}.")
                    else:
                        # User typed a custom question
                        console.print(f"\n[dim]→ Running custom question:[/dim] {choice}\n")
                        query_rag(choice, clarify=clarify, justify=justify, style=style, save=save)
                        break
                except KeyboardInterrupt:
                    console.print("\n")
                    break
                except Exception:
                    # Non-interactive environment, skip
                    break


def save_query_result(question: str, answer: str, sources: list, evidence_map: list, followup_questions: str, style: str):
    """Save query results to JSONL file for later reference."""
    # Ensure data directory exists
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    # Prepare query record
    record = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "style": style,
        "followup_questions": followup_questions,
        "evidence_map": evidence_map,
        "sources": [
            {
                "id": i + 1,
                "title": src.get("title", "Unknown"),
                "score": float(src.get("score", 0.0)),
                "text_full": src.get("text_full", src.get("text", "")),
                "explain": src.get("explain", {})
            }
            for i, src in enumerate(sources)
        ]
    }
    
    # Append to JSONL file
    output_file = output_dir / "query_history.jsonl"
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
            console.print(f"⚠️ Unknown style '{style_choice}', defaulting to 'detailed'.")
            prompt_suffix = style_templates["detailed"]
    except Exception:
        # non-interactive environments: fall back to detailed
        prompt_suffix = style_templates["detailed"]

    # Ask if justifications should be shown in this session
    try:
        j = input("Show retrieval justifications for sources? [y/N]: ").strip().lower()
        justify_session = j == 'y'
    except Exception:
        justify_session = False
    
    # Ask if queries should be auto-saved in this session
    try:
        s = input("Save each query to history? [y/N]: ").strip().lower()
        save_session = s == 'y'
    except Exception:
        save_session = False
    
    while True:
        q = input("\nYou: ")
        if q.strip().lower() == "/exit":
            break
        
        # Check if user entered a number 1-3 (follow-up selection from previous turn)
        if q.strip() in ['1', '2', '3'] and 'last_followup_lines' in locals() and last_followup_lines:
            idx = int(q.strip()) - 1
            if idx < len(last_followup_lines):
                # Extract the question text
                q = last_followup_lines[idx].lstrip('0123456789.-) ').strip()
                console.print(f"[dim]→ Asking follow-up:[/dim] {q}\n")
        
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
            # Parse and store for next turn
            last_followup_lines = [line.strip() for line in followup_questions.split('\n') if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-'))]
            if last_followup_lines:
                console.print("\n[dim]💬 Type 1-3 to ask a follow-up, or type your own question[/dim]")
        else:
            # Clear stored follow-ups if none generated
            last_followup_lines = []
            last_followup_lines = []
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
            # normalize per-query scores for chat mode as well
            scores = [float(s.get("score", 0.0)) for s in sources]
            if scores:
                m = max(scores)
                exps = [math.exp(s - m) for s in scores]
                ssum = sum(exps) if sum(exps) != 0 else 1.0
                probs = [e / ssum for e in exps]
            else:
                probs = [0.0 for _ in sources]

            for i, (src, p) in enumerate(zip(sources, probs), 1):
                score_pct = int(max(0, min(100, round(p * 100))))
                console.print(f"  [{i}] 📄 {src['title']} (relevance: {score_pct}%)")
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
                        
                        # LLM-generated reasoning (if available)
                        llm_reasoning = exp.get('llm_reasoning', '')
                        if llm_reasoning:
                            console.print("      [cyan]LLM Reasoning:[/cyan]")
                            console.print(f"        💡 {llm_reasoning}")
                        
                        # Vector-based why (top matching sentence)
                        why = ''
                        spans = dense.get('top_spans', [])
                        if spans:
                            why = spans[0].get('text', '')
                        console.print("      [dim]Vector Match:[/dim]")
                        if why:
                            console.print(f"        • {why[:120]}{'...' if len(why)>120 else ''}")
                        else:
                            console.print(f"        • (No specific sentence match)")
                        
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
        
        # Save query if auto-save is enabled for this session
        if save_session:
            save_query_result(q, ans, sources, evidence_map, followup_questions, style_choice if 'style_choice' in locals() else 'detailed')
            console.print("\n💾 [green]Query saved to data/query_history.jsonl[/green]")


if __name__ == "__main__":
    app()
