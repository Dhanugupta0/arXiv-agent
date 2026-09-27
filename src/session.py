"""
Session persistence.

Sessions are JSON files (one per paper, keyed by arXiv ID) stored in
sessions/.  They save just enough state to resume QA without re-processing:
  - Paper metadata
  - Briefing
  - ChromaDB collection name (embeddings live in ChromaDB, not the session file)
  - Conversation history

Why JSON instead of pickle?
  - Human-readable, inspectable, safe to load
  - Embeddings are already persisted in ChromaDB — no need to serialise them
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from .config import SESSIONS_DIR
from .models import AgentState, Briefing, PaperMetadata


def save_session(state: AgentState) -> Path:
    """Save an agent state to a JSON session file."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESSIONS_DIR / f"{state.selected_paper.arxiv_id}.json"

    session_data = {
        "paper": asdict(state.selected_paper),
        "briefing": asdict(state.briefing) if state.briefing else None,
        "collection_name": state.collection_name,
        "messages": state.messages,
        "parse_method": state.parse_method,
        "parse_healthy": state.parse_healthy,
        "status": state.status,
        "errors": state.errors,
    }

    with open(path, "w") as f:
        json.dump(session_data, f, indent=2)

    return path


def load_session(arxiv_id: str) -> AgentState:
    """Load a saved session and reconstruct the AgentState."""
    path = SESSIONS_DIR / f"{arxiv_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"No saved session for '{arxiv_id}' in {SESSIONS_DIR}")

    with open(path, "r") as f:
        data = json.load(f)

    paper = PaperMetadata(**data["paper"])
    briefing = Briefing(**data["briefing"]) if data.get("briefing") else None

    state = AgentState(input_query="(resumed session)")
    state.selected_paper = paper
    state.briefing = briefing
    state.collection_name = data.get("collection_name")
    state.messages = data.get("messages", [])
    state.parse_method = data.get("parse_method")
    state.parse_healthy = data.get("parse_healthy", False)
    state.status = data.get("status", "awaiting_qa")
    state.errors = data.get("errors", [])

    return state


def list_sessions() -> List[Dict]:
    """List all saved sessions with basic metadata."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    sessions = []
    for path in sorted(SESSIONS_DIR.glob("*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            sessions.append({
                "arxiv_id": data["paper"]["arxiv_id"],
                "title": data["paper"]["title"],
                "published": data["paper"]["published"],
                "qa_turns": len(data.get("messages", [])) // 2,
            })
        except Exception:
            continue
    return sessions
