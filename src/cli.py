"""
Rich-powered interactive CLI for the arXiv Digest Agent.

Provides an engaging terminal experience with:
  - Colour-coded status messages and spinners
  - Formatted markdown briefing output
  - Interactive QA loop with styled prompts
  - Session management with table display
"""
from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich import print as rprint

from .graph import ArxivDigestGraph
from .models import AgentState
from . import session
from .config import has_groq_key

console = Console()


# ── Helpers ───────────────────────────────────────────────────────────────

def _show_briefing(state: AgentState) -> None:
    """Render the briefing as a rich markdown panel."""
    md = state.briefing.to_markdown()
    console.print()
    console.print(Panel(
        Markdown(md),
        title="[bold cyan]📄 Executive Briefing[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    if state.errors:
        console.print()
        console.print("[yellow]⚠  Notes / warnings during ingestion:[/yellow]")
        for e in state.errors:
            console.print(f"  [dim]•[/dim] {e}")
    console.print()


def _qa_loop(graph: ArxivDigestGraph, state: AgentState) -> None:
    """Interactive QA loop — ask questions about the paper."""
    title = state.selected_paper.title
    if len(title) > 60:
        title = title[:57] + "..."

    console.print(Panel(
        f"[bold]Ask questions about:[/bold] [italic]{title}[/italic]\n"
        "[dim]Type [bold]exit[/bold] or [bold]quit[/bold] to end the session.[/dim]",
        border_style="green",
    ))
    console.print()

    while True:
        try:
            question = Prompt.ask("[bold green]You[/bold green]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        question = question.strip()
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break

        with console.status("[cyan]Thinking...[/cyan]", spinner="dots"):
            answer = graph.ask(state, question)

        console.print()
        console.print(Panel(
            answer,
            title="[bold blue]🤖 Agent[/bold blue]",
            border_style="blue",
            padding=(0, 2),
        ))
        console.print()

    # Save session on exit
    session.save_session(state)
    arxiv_id = state.selected_paper.arxiv_id
    console.print(
        f"[green]✓[/green] Session saved. Resume later with: "
        f"[bold]python main.py qa {arxiv_id}[/bold]"
    )


# ── Commands ──────────────────────────────────────────────────────────────

def cmd_digest(query: str, skip_qa: bool = False) -> None:
    """Main command: fetch a paper, summarize, and enter QA."""
    console.print()
    console.print(Panel(
        "[bold]arXiv Digest Agent[/bold]\n"
        f"[dim]Query:[/dim] {query}",
        border_style="magenta",
    ))
    console.print()

    graph = ArxivDigestGraph()

    # Node 1: Query Understanding
    with console.status("[cyan]🔍 Understanding query...[/cyan]", spinner="dots"):
        state = AgentState(input_query=query)
        graph.query_understanding(state)

    intent_label = "Direct paper ID" if state.query_intent == "direct_id" else "Topic search"
    console.print(f"  [dim]Intent:[/dim] {intent_label}")

    # Node 2: arXiv Retrieval
    with console.status("[cyan]📡 Searching arXiv...[/cyan]", spinner="dots"):
        graph.arxiv_retrieval(state)

    if state.status == "no_results":
        console.print("[red]✗ No matching papers found.[/red]")
        for e in state.errors:
            console.print(f"  [dim]•[/dim] {e}")
        sys.exit(1)

    console.print(f"  [dim]Found:[/dim] {len(state.candidate_papers)} candidate(s)")

    # Node 3: Selection
    with console.status("[cyan]🏆 Selecting best paper...[/cyan]", spinner="dots"):
        graph.selection_ranking(state)

    console.print(f"  [dim]Selected:[/dim] [bold]{state.selected_paper.title}[/bold]")

    # Node 4: Fetch & Parse
    with console.status("[cyan]📥 Downloading & parsing PDF...[/cyan]", spinner="dots"):
        graph.fetch_and_parse(state)

    console.print(f"  [dim]Parse method:[/dim] {state.parse_method}")

    # Node 5: Chunk & Embed
    with console.status("[cyan]🧩 Chunking & embedding (local model → ChromaDB)...[/cyan]", spinner="dots"):
        graph.chunk_and_embed(state)

    console.print(f"  [dim]Chunks:[/dim] {len(state.chunks)} chunks indexed")

    # Node 6: Summarize
    label = "Groq" if has_groq_key() else "extractive (no API key)"
    with console.status(f"[cyan]📝 Generating briefing ({label})...[/cyan]", spinner="dots"):
        graph.summarize(state)

    # Show the briefing
    _show_briefing(state)

    # Save session
    session.save_session(state)

    # Node 7: QA Loop
    if not skip_qa:
        _qa_loop(graph, state)


def cmd_qa(arxiv_id: str) -> None:
    """Resume the QA loop for a previously ingested paper."""
    console.print()
    with console.status("[cyan]Loading session...[/cyan]", spinner="dots"):
        try:
            state = session.load_session(arxiv_id)
        except FileNotFoundError as e:
            console.print(f"[red]✗ {e}[/red]")
            sys.exit(1)

    graph = ArxivDigestGraph()

    _show_briefing(state)
    _qa_loop(graph, state)


def cmd_sessions() -> None:
    """List all saved sessions in a table."""
    console.print()
    sessions = session.list_sessions()

    if not sessions:
        console.print(
            "[yellow]No saved sessions yet.[/yellow]\n"
            "Run [bold]python main.py digest <query>[/bold] to get started."
        )
        return

    table = Table(title="📚 Saved Sessions", border_style="cyan")
    table.add_column("arXiv ID", style="bold")
    table.add_column("Title", max_width=50)
    table.add_column("Published", style="dim")
    table.add_column("QA Turns", justify="right")

    for s in sessions:
        title = s["title"]
        if len(title) > 50:
            title = title[:47] + "..."
        table.add_row(s["arxiv_id"], title, s["published"], str(s["qa_turns"]))

    console.print(table)
    console.print(
        "\n[dim]Resume a session:[/dim] [bold]python main.py qa <arxiv_id>[/bold]"
    )


def cmd_demo() -> None:
    """
    Demo mode using the bundled synthetic paper.
    Exercises nodes 5–7 without needing internet for arXiv.

    Works with or without API keys:
      - With GROQ_API_KEY: full LLM-powered briefing + QA
      - Without GROQ_API_KEY: extractive briefing + extractive QA
    Embeddings are always local (sentence-transformers), no key needed.
    """
    from .models import PaperMetadata

    console.print()
    console.print(Panel(
        "[bold]arXiv Digest Agent — Demo Mode[/bold]\n"
        "[dim]Using bundled synthetic paper (no arXiv fetch needed)[/dim]",
        border_style="magenta",
    ))
    console.print()

    if not has_groq_key():
        console.print(
            "[yellow]ℹ  GROQ_API_KEY not set — running in zero-config mode.[/yellow]\n"
            "[dim]   Briefing: extractive heuristics (no LLM)\n"
            "   QA: returns top retrieved passage directly\n"
            "   Set GROQ_API_KEY in .env for full LLM-powered output.[/dim]"
        )
        console.print()

    sample_path = Path(__file__).resolve().parent.parent / "sample_data" / "sample_paper.txt"
    if not sample_path.exists():
        console.print("[red]✗ sample_data/sample_paper.txt not found.[/red]")
        sys.exit(1)

    raw_text = sample_path.read_text()

    paper = PaperMetadata(
        arxiv_id="0000.00000",
        title="Sparse Gradient Checkpointing for KV-Cache Compression in Long-Context Transformers (demo)",
        authors=["A. Researcher", "B. Researcher"],
        abstract=raw_text.split("\n\n")[1] if "\n\n" in raw_text else raw_text[:500],
        published="2026-01-01",
        pdf_url="(none — offline demo)",
        abs_url="(none — offline demo)",
        categories=["cs.LG"],
    )

    state = AgentState(input_query="demo")
    state.query_intent = "direct_id"
    state.candidate_papers = [paper]
    state.selected_paper = paper
    state.raw_text = raw_text
    state.parse_method = "bundled_demo_text"
    state.parse_healthy = True

    graph = ArxivDigestGraph()

    # Node 5
    with console.status("[cyan]🧩 Chunking & embedding (local model)...[/cyan]", spinner="dots"):
        graph.chunk_and_embed(state)

    console.print(f"  [dim]Chunks:[/dim] {len(state.chunks)} chunks indexed")

    # Node 6
    label = "Groq" if has_groq_key() else "extractive (no API key)"
    with console.status(f"[cyan]📝 Generating briefing ({label})...[/cyan]", spinner="dots"):
        graph.summarize(state)

    _show_briefing(state)
    session.save_session(state)
    _qa_loop(graph, state)
