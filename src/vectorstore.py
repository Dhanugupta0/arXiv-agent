"""
ChromaDB-backed vector store for RAG retrieval.

Embeddings are generated locally by sentence-transformers and stored in
ChromaDB for persistent, queryable storage.  Each paper gets its own
ChromaDB collection (keyed by sanitised arXiv ID + embedding model tag),
so:
  - Multiple papers can coexist without interference
  - Sessions persist across CLI invocations
  - Embeddings are never recomputed for already-indexed papers
  - Changing embedding models won't collide with old collections
"""
from __future__ import annotations

import re
from typing import List, Tuple

import chromadb
import numpy as np

from .config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL
from .embeddings import embed_texts, embed_query
from .models import Chunk


def _embedding_tag() -> str:
    """
    Short tag derived from the embedding model name.
    e.g. "BAAI/bge-small-en-v1.5" → "bge_small_en_v1_5"

    Embedded in collection names so that switching embedding models
    doesn't silently reuse incompatible vectors.
    """
    tag = EMBEDDING_MODEL.split("/")[-1]
    tag = re.sub(r"[^a-zA-Z0-9]", "_", tag).lower()
    return tag


def _sanitize_collection_name(arxiv_id: str) -> str:
    """
    ChromaDB collection names must be 3-63 chars, start/end with alnum,
    and contain only alnum, underscores, hyphens.
    """
    tag = _embedding_tag()
    name = f"paper_{re.sub(r'[^a-zA-Z0-9_-]', '_', arxiv_id)}_{tag}"
    # Ensure length constraints
    if len(name) < 3:
        name = name + "_col"
    return name[:63]


def _get_client() -> chromadb.PersistentClient:
    """Get or create a persistent ChromaDB client."""
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))


def index_chunks(chunks: List[Chunk], arxiv_id: str) -> str:
    """
    Embed chunks via sentence-transformers and store them in ChromaDB.

    Returns the collection name (used later for querying).
    If the collection already exists with the same number of documents,
    skip re-indexing (embeddings are already stored).
    """
    collection_name = _sanitize_collection_name(arxiv_id)
    client = _get_client()

    # Get or create the collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Skip re-indexing if already populated
    if collection.count() >= len(chunks):
        return collection_name

    # Generate embeddings via sentence-transformers (local)
    texts = [c.text for c in chunks]
    vectors = embed_texts(texts)

    # Add to ChromaDB
    collection.add(
        ids=[f"chunk_{c.id}" for c in chunks],
        embeddings=vectors.tolist(),
        documents=texts,
        metadatas=[{"chunk_id": c.id, "section_hint": c.section_hint or ""} for c in chunks],
    )

    return collection_name


def search(collection_name: str, query: str, top_k: int = 4) -> List[Tuple[str, float]]:
    """
    Search the vector store for chunks relevant to a query.

    Returns a list of (chunk_text, similarity_score) tuples,
    sorted by relevance (highest first).
    """
    client = _get_client()

    try:
        collection = client.get_collection(name=collection_name)
    except Exception:
        return []

    if collection.count() == 0:
        return []

    # Embed the query via sentence-transformers (with BGE instruction prefix)
    query_vector = embed_query(query)

    results = collection.query(
        query_embeddings=query_vector.tolist(),
        n_results=min(top_k, collection.count()),
        include=["documents", "distances"],
    )

    if not results or not results["documents"] or not results["documents"][0]:
        return []

    pairs: List[Tuple[str, float]] = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        # ChromaDB cosine distance = 1 - cosine_similarity
        similarity = 1.0 - dist
        pairs.append((doc, similarity))

    return pairs
