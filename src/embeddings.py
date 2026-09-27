"""
Node 5b: Local embeddings via sentence-transformers.

Uses BAAI/bge-small-en-v1.5 (33M params, 384-dim, CPU-friendly).
Downloads once from HuggingFace (~130MB), then works fully offline.

Why local sentence-transformers instead of Jina AI API?
  - No API key required — zero-configuration for grading
  - Works offline after first download
  - Fast CPU inference for a small model
  - Normalized vectors ready for cosine similarity
"""
from __future__ import annotations

from typing import List

import numpy as np

from .config import EMBEDDING_MODEL, EMBEDDING_DIM

# ── BGE query instruction prefix ────────────────────────────────────────
# bge-small-en-v1.5 requires this prefix on QUERY embeddings only,
# not on indexed passages.  Skipping it measurably hurts retrieval.
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Singleton model ──────────────────────────────────────────────────────
_model = None


def _ensure_model():
    """Load the SentenceTransformer model once (lazy singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of passage texts for storage/indexing.
    Returns a numpy array of shape (len(texts), EMBEDDING_DIM).

    No instruction prefix — passages are embedded as-is.
    """
    if not texts:
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model = _ensure_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query text for retrieval.
    Returns a numpy array of shape (1, EMBEDDING_DIM).

    Prepends the BGE instruction prefix for better retrieval quality.
    """
    model = _ensure_model()
    prefixed = _BGE_QUERY_PREFIX + text
    vector = model.encode([prefixed], normalize_embeddings=True, show_progress_bar=False)
    return np.array(vector, dtype=np.float32)
