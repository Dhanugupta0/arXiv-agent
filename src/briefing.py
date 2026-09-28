"""
Node 6: Executive Briefing generation.

Uses Groq LLM to produce a structured JSON briefing from the paper's text.
Includes an extractive fallback if the LLM call fails for any reason,
or if GROQ_API_KEY is not configured (zero-config demo mode).
"""
from __future__ import annotations

import json
import re
from typing import List

from .config import has_groq_key
from .models import Briefing, PaperMetadata

_JSON_SCHEMA_HINT = """Respond with ONLY a single JSON object (no markdown fences, no prose
before or after) with exactly these keys:
{
  "summary": "<one paragraph, plain English, on why this paper matters>",
  "problem_statement": "<the specific gap/problem the paper addresses>",
  "method": ["<bullet point>", "..."],
  "key_results": ["<bullet point>", "..."],
  "limitations": ["<bullet point>", "..."],
  "followup_questions": ["<question a reader might ask>", "..."]
}
The "limitations" list must never be empty — if the paper does not state
limitations explicitly, infer plausible ones from the scope of the method
and say so (e.g. "Not explicitly discussed by the authors; likely
limitation: ...")."""


def _extract_json(text: str) -> dict:
    """Robustly extract a JSON object from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(text[start : end + 1])


def _llm_briefing_fields(paper: PaperMetadata, full_text: str) -> dict:
    """Generate briefing fields via Groq LLM."""
    from . import llm  # deferred import to avoid crash when key is absent

    system = (
        "You are an expert research assistant producing a structured executive "
        "briefing for a busy engineer deciding whether to read a paper in full. "
        "Be precise and concrete; do not pad with generic statements."
    )
    user = (
        f"Paper title: {paper.title}\n"
        f"Abstract: {paper.abstract}\n\n"
        f"Full text of the paper (intro/method/results/conclusion, "
        f"references stripped):\n{full_text}\n\n"
        f"{_JSON_SCHEMA_HINT}"
    )
    raw = llm.complete(system, user)
    return _extract_json(raw)


def _extractive_fields(paper: PaperMetadata, text: str) -> dict:
    """
    No-LLM fallback: cheap heuristics over the abstract + raw text.
    Used if the LLM call fails for any reason, or if no API key is set.
    """
    sentences = re.split(r"(?<=[.!?])\s+", paper.abstract)
    summary = " ".join(sentences[:3]) if sentences else paper.abstract
    problem_statement = sentences[0] if sentences else "Not available."

    method_bullets: List[str] = []
    for header in ("method", "approach", "model", "architecture"):
        m = re.search(rf"\n\s*\d*\.?\s*{header}s?\b.*?\n", text, re.IGNORECASE)
        if m:
            snippet = text[m.end(): m.end() + 400].strip().replace("\n", " ")
            if snippet:
                method_bullets.append(snippet[:280])
    if not method_bullets:
        method_bullets = ["See abstract for a high-level description of the method."]

    results_bullets: List[str] = []
    m = re.search(r"\n\s*\d*\.?\s*(results|experiments|evaluation)\b.*?\n", text, re.IGNORECASE)
    if m:
        snippet = text[m.end(): m.end() + 400].strip().replace("\n", " ")
        if snippet:
            results_bullets.append(snippet[:280])
    if not results_bullets:
        results_bullets = [s for s in sentences[3:6]] or ["Not available from the excerpt processed."]

    limitations: List[str] = []
    m = re.search(r"\n\s*\d*\.?\s*limitations?\b.*?\n", text, re.IGNORECASE)
    if m:
        snippet = text[m.end(): m.end() + 400].strip().replace("\n", " ")
        if snippet:
            limitations.append(snippet[:280])
    if not limitations:
        limitations = ["Limitations not explicitly stated in extracted text."]

    return {
        "summary": summary,
        "problem_statement": problem_statement,
        "method": method_bullets,
        "key_results": results_bullets,
        "limitations": limitations,
        "followup_questions": [
            "What dataset(s) or benchmarks were used to validate the claims?",
            "How does this compare against the strongest prior baseline?",
            "What would break if scaled up by 10x?",
        ],
    }


def strip_references(text: str) -> str:
    """Strip references section at the end of a paper if present."""
    pattern = r"\n\s*(?:\d*\.?\s*)?(?:References|BIBLIOGRAPHY|References and Notes)\s*\n"
    match = list(re.finditer(pattern, text, re.IGNORECASE))
    if match:
        last_match = match[-1]
        if last_match.start() > len(text) * 0.5:
            return text[:last_match.start()].strip()
    return text


def generate_briefing(paper: PaperMetadata, full_text: str, degraded: bool) -> Briefing:
    """
    Generate a structured executive briefing.

    Uses full reference-stripped paper text with LLM or extractive fallback.
    """
    cleaned_text = strip_references(full_text)
    fields = None

    try:
        fields = _llm_briefing_fields(paper, cleaned_text)
    except Exception:
        pass  # fall through to extractive

    if fields is None:
        fields = _extractive_fields(paper, cleaned_text)
        if not has_groq_key():
            degraded = True  # mark as degraded when running without LLM

    return Briefing(
        title=paper.title,
        authors=paper.authors,
        arxiv_id=paper.arxiv_id,
        published=paper.published,
        link=paper.abs_url,
        summary=fields.get("summary", ""),
        problem_statement=fields.get("problem_statement", ""),
        method=fields.get("method", []),
        key_results=fields.get("key_results", []),
        limitations=fields.get("limitations", []),
        followup_questions=fields.get("followup_questions", []),
        degraded=degraded,
    )
