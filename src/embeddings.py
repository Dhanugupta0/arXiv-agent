"""
Node 5b: Jina AI Embeddings via HTTP API.

Uses Jina AI's embedding API (https://api.jina.ai/v1/embeddings) to
generate dense vector embeddings for paper chunks and queries.

Why Jina AI?
  - High-quality multilingual embeddings
  - Simple HTTP API — no heavy SDK dependency
  - Free tier available for development
  - Returns normalised vectors ready for cosine similarity
"""
from __future__ import annotations

from typing import List

import numpy as np
import requests

from .config import JINA_API_KEY, JINA_EMBEDDING_MODEL, JINA_EMBEDDING_DIM

_JINA_API_URL = "https://api.jina.ai/v1/embeddings"


def _get_headers() -> dict:
    if not JINA_API_KEY:
        raise RuntimeError(
            "JINA_API_KEY is not set. "
            "Add it to your .env file — get a free key at https://jina.ai/embeddings"
        )
    return {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of texts using Jina AI.
    Returns a numpy array of shape (len(texts), embedding_dim).
    """
    if not texts:
        return np.empty((0, JINA_EMBEDDING_DIM), dtype=np.float32)

    # Jina API accepts batches — send all at once for efficiency
    payload = {
        "model": JINA_EMBEDDING_MODEL,
        "input": texts,
        "task": "retrieval.passage",
        "dimensions": JINA_EMBEDDING_DIM,
    }

    resp = requests.post(_JINA_API_URL, json=payload, headers=_get_headers(), timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Sort by index to maintain order (API may return out of order)
    embeddings_data = sorted(data["data"], key=lambda x: x["index"])
    vectors = [item["embedding"] for item in embeddings_data]

    return np.array(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query text using Jina AI.
    Returns a numpy array of shape (1, embedding_dim).
    """
    payload = {
        "model": JINA_EMBEDDING_MODEL,
        "input": [text],
        "task": "retrieval.query",
        "dimensions": JINA_EMBEDDING_DIM,
    }

    resp = requests.post(_JINA_API_URL, json=payload, headers=_get_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()

    vector = data["data"][0]["embedding"]
    return np.array([vector], dtype=np.float32)
