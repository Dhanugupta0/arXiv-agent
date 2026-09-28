"""
Centralised configuration.

All environment variables and tuneable constants live here so they're easy
to find and override without touching business logic.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ── API Keys ──────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")

# ── Groq LLM settings ────────────────────────────────────────────────────
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_TEMPERATURE: float = float(os.environ.get("GROQ_TEMPERATURE", "0.2"))
GROQ_MAX_TOKENS: int = int(os.environ.get("GROQ_MAX_TOKENS", "4096"))

# ── Local embedding model ────────────────────────────────────────────────
# BAAI/bge-small-en-v1.5: 33M params, 384-dim, CPU-friendly.
# Downloads ~130MB from HuggingFace on first run, then works fully offline.
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM: int = int(os.environ.get("EMBEDDING_DIM", "384"))

# ── Chunking ──────────────────────────────────────────────────────────────
CHUNK_WORDS: int = 800
OVERLAP_WORDS: int = 150

# ── ChromaDB persistence ────────────────────────────────────────────────
CHROMA_PERSIST_DIR: Path = _PROJECT_ROOT / "chroma_store"

# ── Session persistence ──────────────────────────────────────────────────
SESSIONS_DIR: Path = _PROJECT_ROOT / "sessions"

# ── arXiv settings ────────────────────────────────────────────────────────
MAX_SEARCH_RESULTS: int = 5
MAX_EXPANSION_ATTEMPTS: int = 1

# ── RAG QA ────────────────────────────────────────────────────────────────
QA_TOP_K: int = 3

# ── Helpers ───────────────────────────────────────────────────────────────

def has_groq_key() -> bool:
    """Check whether a Groq API key is configured."""
    return bool(GROQ_API_KEY and GROQ_API_KEY.strip())
