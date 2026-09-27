"""
Data models / schemas for the agent pipeline.

Every piece of data that flows through the state graph is defined here:
PaperMetadata, Chunk, Briefing, and AgentState.  Nodes read from an
AgentState instance, mutate it, and pass it along.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class PaperMetadata:
    """Metadata for a single arXiv paper."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published: str
    pdf_url: str
    abs_url: str
    categories: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    """A text chunk produced by the chunking stage."""
    id: int
    text: str
    section_hint: Optional[str] = None


@dataclass
class Briefing:
    """Structured executive briefing generated in Node 6."""
    title: str
    authors: List[str]
    arxiv_id: str
    published: str
    link: str
    summary: str
    problem_statement: str
    method: List[str]
    key_results: List[str]
    limitations: List[str]
    followup_questions: List[str]
    degraded: bool = False  # True when generated from abstract only

    def to_markdown(self) -> str:
        lines = [
            f"# {self.title}",
            "",
            f"**arXiv ID:** {self.arxiv_id}  ",
            f"**Authors:** {', '.join(self.authors)}  ",
            f"**Published:** {self.published}  ",
            f"**Link:** {self.link}",
            "",
        ]
        if self.degraded:
            lines += [
                "> ⚠️  Full-text parsing was unavailable. This briefing is based on "
                "the abstract only and may be less detailed than usual.",
                "",
            ]
        lines += [
            "## Why This Paper Matters",
            self.summary,
            "",
            "## Problem Statement",
            self.problem_statement,
            "",
            "## Method / Approach",
        ]
        lines += [f"- {m}" for m in self.method]
        lines += ["", "## Key Results / Claims"]
        lines += [f"- {r}" for r in self.key_results]
        lines += ["", "## Limitations"]
        lines += [f"- {lim}" for lim in self.limitations]
        lines += ["", "## Suggested Follow-up Questions"]
        lines += [f"- {q}" for q in self.followup_questions]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentState:
    """
    Shared mutable state threaded through every node in the graph.

    Fields are grouped by pipeline stage:
      input/routing → retrieval → parsing → chunk/embed → briefing → QA → control
    """
    # ── input / routing ──
    input_query: str
    query_intent: Optional[str] = None       # "direct_id" | "topic_search"
    extracted_id: Optional[str] = None
    search_keywords: Optional[str] = None
    expansion_attempts: int = 0

    # ── retrieval ──
    candidate_papers: List[PaperMetadata] = field(default_factory=list)
    selected_paper: Optional[PaperMetadata] = None

    # ── parsing ──
    raw_text: Optional[str] = None
    parse_method: Optional[str] = None       # "pymupdf" | "pypdf" | "ocr" | "abstract_only"
    parse_healthy: bool = False

    # ── chunk / embed ──
    chunks: List[Chunk] = field(default_factory=list)
    collection_name: Optional[str] = None    # ChromaDB collection name

    # ── briefing ──
    briefing: Optional[Briefing] = None

    # ── QA conversation ──
    messages: List[Dict[str, str]] = field(default_factory=list)

    # ── control / diagnostics ──
    status: str = "running"   # "running" | "awaiting_qa" | "no_results" | "error"
    errors: List[str] = field(default_factory=list)

    def log_error(self, msg: str) -> None:
        self.errors.append(msg)
