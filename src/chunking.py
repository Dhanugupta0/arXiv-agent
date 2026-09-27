"""
Node 5a: Text chunking.

- Strip bibliography/references (pollutes embedding space)
- Split into ~800-word chunks with ~150-word overlap
- Word-based sizing avoids a tokenizer dependency
"""
from __future__ import annotations

import re
from typing import List

from .models import Chunk
from .config import CHUNK_WORDS, OVERLAP_WORDS

_REFERENCES_HEADER_RE = re.compile(
    r"\n\s*(references|bibliography|works cited)\s*\n", re.IGNORECASE
)


def strip_references(text: str) -> str:
    """Remove everything after a References / Bibliography header."""
    match = _REFERENCES_HEADER_RE.search(text)
    if match and match.start() > len(text) * 0.4:
        return text[: match.start()]
    return text


def chunk_text(
    text: str,
    chunk_words: int = CHUNK_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> List[Chunk]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []

    chunks: List[Chunk] = []
    step = max(chunk_words - overlap_words, 1)
    start = 0
    idx = 0

    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk_str = " ".join(words[start:end])
        chunks.append(Chunk(id=idx, text=chunk_str))
        idx += 1
        if end == len(words):
            break
        start += step

    return chunks


def prepare_chunks(raw_text: str) -> List[Chunk]:
    """Full pipeline: strip references, then chunk."""
    cleaned = strip_references(raw_text)
    return chunk_text(cleaned)
