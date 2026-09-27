"""
Node 1 (query understanding) + Node 2 (arXiv retrieval).

Intent classification is deterministic (regex), not LLM-based — an arXiv
ID/URL has a rigid shape, so burning an LLM call to detect it would be
wasteful.  Only genuinely ambiguous free-text queries fall through to
"topic_search".
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import arxiv

from .models import PaperMetadata
from .config import MAX_SEARCH_RESULTS

# ── Regex patterns for arXiv IDs / URLs ──────────────────────────────────
_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?")
_ARXIV_URL_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?")


def parse_query_intent(query: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Returns (intent, extracted_id, search_keywords).
    intent is either "direct_id" or "topic_search".
    """
    q = query.strip()

    # Check for a full arXiv URL first
    url_match = _ARXIV_URL_RE.search(q)
    if url_match:
        return "direct_id", url_match.group(1), None

    # Exact arXiv ID
    id_match = _ARXIV_ID_RE.fullmatch(q)
    if id_match:
        return "direct_id", id_match.group(1), None

    # Loose ID embedded in a short sentence ("summarize 2401.12345 for me")
    loose_id_match = _ARXIV_ID_RE.search(q)
    if loose_id_match and len(q) < 40:
        return "direct_id", loose_id_match.group(1), None

    return "topic_search", None, q


def _to_metadata(result) -> PaperMetadata:
    """Convert an arxiv.Result to our PaperMetadata schema."""
    return PaperMetadata(
        arxiv_id=result.get_short_id(),
        title=result.title.strip().replace("\n", " "),
        authors=[a.name for a in result.authors],
        abstract=result.summary.strip().replace("\n", " "),
        published=result.published.strftime("%Y-%m-%d") if result.published else "unknown",
        pdf_url=result.pdf_url,
        abs_url=result.entry_id,
        categories=list(result.categories) if result.categories else [],
    )


def fetch_by_id(arxiv_id: str) -> Optional[PaperMetadata]:
    """Fetch a single paper by its arXiv ID."""
    search = arxiv.Search(id_list=[arxiv_id])
    client = arxiv.Client()
    try:
        result = next(client.results(search))
    except StopIteration:
        return None
    return _to_metadata(result)


def search_by_topic(keywords: str, max_results: int | None = None) -> List[PaperMetadata]:
    """Search arXiv by topic keywords, sorted by relevance."""
    # Use 'all:' field prefix for better matching across title+abstract+fulltext
    query = f"all:{keywords}"
    search = arxiv.Search(
        query=query,
        max_results=max_results or MAX_SEARCH_RESULTS,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    client = arxiv.Client()
    return [_to_metadata(r) for r in client.results(search)]


def expand_query(keywords: str) -> str:
    """
    Heuristic query broadening for the zero-results case (no LLM needed):
    - Drop quoted exact-phrase constraints
    - Drop parenthetical qualifiers
    - Drop the last (most-specific) keyword
    """
    q = re.sub(r'"[^"]*"', "", keywords)
    q = re.sub(r"\([^)]*\)", "", q)
    q = q.strip()
    words = q.split()
    if len(words) > 3:
        words = words[:-1]
    return " ".join(words) if words else keywords
