#!/usr/bin/env python3
"""
View and search saved query results from data/query_history.jsonl
"""

import json
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from datetime import datetime

console = Console()

def load_query_history():
    """Load all saved queries from JSONL file."""
    history_file = Path("data/query_history.jsonl")
    if not history_file.exists():
        console.print("❌ No query history found. Use --save flag when running queries.")
        return []
    
    queries = []
    with open(history_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def list_queries(queries, limit=None):
    """Display a table of all saved queries."""
    if not queries:
        console.print("No queries found.")
        return
    
    table = Table(title="📚 Saved Query History", show_lines=True)
    table.add_column("#", style="cyan", width=4)
    table.add_column("Timestamp", style="dim", width=20)
    table.add_column("Question", style="green", width=60)
    table.add_column("Style", width=10)
    table.add_column("Sources", width=8)
    
    display_queries = queries[-limit:] if limit else queries
    
    for i, q in enumerate(display_queries, 1):
        timestamp = datetime.fromisoformat(q["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        question = q["question"][:57] + "..." if len(q["question"]) > 60 else q["question"]
        style = q.get("style", "auto")
        num_sources = len(q.get("sources", []))
        
        table.add_row(str(i), timestamp, question, style, str(num_sources))
    
    console.print(table)
    console.print(f"\nTotal queries: {len(queries)}")


def view_query(queries, index):
    """Display full details of a specific query."""
    if index < 1 or index > len(queries):
        console.print(f"❌ Invalid query index. Valid range: 1-{len(queries)}")
        return
    
    q = queries[index - 1]
    
    console.print(f"\n{'='*80}")
    console.print(f"📅 [cyan]Timestamp:[/cyan] {q['timestamp']}")
    console.print(f"❓ [cyan]Question:[/cyan] {q['question']}")
    console.print(f"🎨 [cyan]Style:[/cyan] {q.get('style', 'auto')}")
    console.print(f"\n🤖 [green]Answer:[/green]\n{q['answer']}")
    
    if q.get("followup_questions"):
        console.print(f"\n💡 [cyan]Follow-up Questions:[/cyan]\n{q['followup_questions']}")
    
    if q.get("evidence_map"):
        console.print("\n📋 [cyan]Evidence Map:[/cyan]")
        for ev in q["evidence_map"]:
            console.print(f"  [Source {ev['id']}] {ev['source_title']}")
            console.print(f"  ↳ \"{ev['span'][:100]}...\"")
    
    console.print(f"\n📚 [cyan]Sources ({len(q['sources'])}):[/cyan]")
    for src in q["sources"]:
        console.print(f"\n  [{src['id']}] {src['title']} (score: {src['score']:.3f})")
        console.print(f"  {'='*76}")
        text = src['text_full']
        console.print(f"  {text[:500]}..." if len(text) > 500 else f"  {text}")
        console.print(f"  {'='*76}")


def search_queries(queries, keyword):
    """Search queries by keyword in question or answer."""
    keyword_lower = keyword.lower()
    matches = [
        q for q in queries
        if keyword_lower in q["question"].lower() or keyword_lower in q["answer"].lower()
    ]
    
    if not matches:
        console.print(f"No queries found matching '{keyword}'")
        return
    
    console.print(f"\n🔍 Found {len(matches)} matching queries:\n")
    for i, q in enumerate(matches, 1):
        console.print(f"{i}. {q['question']}")
        console.print(f"   {q['timestamp']}")
        console.print()


def export_to_markdown(queries, output_file="data/query_history.md"):
    """Export query history to Markdown format."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Query History\n\n")
        f.write(f"Total queries: {len(queries)}\n\n")
        f.write("---\n\n")
        
        for i, q in enumerate(queries, 1):
            f.write(f"## Query {i}: {q['question']}\n\n")
            f.write(f"**Timestamp:** {q['timestamp']}  \n")
            f.write(f"**Style:** {q.get('style', 'auto')}  \n\n")
            
            f.write("### Answer\n\n")
            f.write(f"{q['answer']}\n\n")
            
            if q.get("followup_questions"):
                f.write("### Follow-up Questions\n\n")
                f.write(f"{q['followup_questions']}\n\n")
            
            if q.get("evidence_map"):
                f.write("### Evidence Map\n\n")
                for ev in q["evidence_map"]:
                    f.write(f"- **[Source {ev['id']}]** {ev['source_title']}\n")
                    f.write(f"  > {ev['span']}\n\n")
            
            f.write("### Sources\n\n")
            for src in q["sources"]:
                f.write(f"#### Source {src['id']}: {src['title']}\n\n")
                f.write(f"**Score:** {src['score']:.3f}\n\n")
                f.write(f"```\n{src['text_full']}\n```\n\n")
            
            f.write("---\n\n")
    
    console.print(f"✅ Exported to {output_file}")


def main():
    """Main CLI for viewing saved queries."""
    queries = load_query_history()
    
    if not queries:
        return
    
    if len(sys.argv) == 1:
        # No arguments - show list of recent queries
        list_queries(queries, limit=20)
        console.print("\n[dim]Usage:[/dim]")
        console.print("  python scripts/view_saved_queries.py list       # Show all queries")
        console.print("  python scripts/view_saved_queries.py view N     # View query #N in detail")
        console.print("  python scripts/view_saved_queries.py search KEYWORD")
        console.print("  python scripts/view_saved_queries.py export     # Export to Markdown")
    
    elif sys.argv[1] == "list":
        list_queries(queries)
    
    elif sys.argv[1] == "view" and len(sys.argv) > 2:
        try:
            index = int(sys.argv[2])
            view_query(queries, index)
        except ValueError:
            console.print("❌ Invalid query number")
    
    elif sys.argv[1] == "search" and len(sys.argv) > 2:
        keyword = sys.argv[2]
        search_queries(queries, keyword)
    
    elif sys.argv[1] == "export":
        export_to_markdown(queries)
    
    else:
        console.print("❌ Unknown command. Use: list, view N, search KEYWORD, or export")


if __name__ == "__main__":
    main()
