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
JINA_API_KEY: str = os.environ.get("JINA_API_KEY", "")

# ── Groq LLM settings ────────────────────────────────────────────────────
GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
GROQ_TEMPERATURE: float = float(os.environ.get("GROQ_TEMPERATURE", "0.2"))
GROQ_MAX_TOKENS: int = int(os.environ.get("GROQ_MAX_TOKENS", "800"))

# ── Jina AI embeddings ───────────────────────────────────────────────────
JINA_EMBEDDING_MODEL: str = os.environ.get("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")
JINA_EMBEDDING_DIM: int = int(os.environ.get("JINA_EMBEDDING_DIM", "1024"))

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
