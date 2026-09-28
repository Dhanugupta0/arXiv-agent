"""
Node 7: Grounded QA with anti-hallucination prompt.

Retrieves top-k chunks from ChromaDB, then instructs Groq to answer
using ONLY those chunks.  If the answer isn't in the paper, the model
is told to refuse rather than hallucinate.

When GROQ_API_KEY is not set, falls back to returning the most relevant
retrieved chunk directly (extractive QA).
"""
from __future__ import annotations

from typing import List, Tuple

from .config import has_groq_key, QA_TOP_K
from . import vectorstore
from .models import AgentState

GROUNDED_SYSTEM_PROMPT = (
    "You are an expert academic research assistant answering questions about "
    "a specific paper. You are provided with the paper's general info (Title, Summary/Abstract) "
    "and specific excerpts retrieved via search.\n\n"
    "Instructions:\n"
    "1. Answer the user's question using ONLY the provided Paper Info and excerpts.\n"
    "2. If the user asks for a general explanation, overview, or broader context, use the Paper Info to provide a comprehensive answer.\n"
    "3. If the user asks for specific details not found in the provided text, respond EXACTLY with: "
    '"The provided text does not contain sufficient information to answer this question."\n'
    "4. Do not use outside/pretrained knowledge. Keep answers clear and cite excerpts (e.g., [Excerpt 1]) when they are used."
)

# Max words safeguard per excerpt (comfortably exceeds the 800-word chunk size to preserve full text including limitations & conclusions)
_MAX_EXCERPT_WORDS = 1500


def _truncate(text: str, max_words: int = _MAX_EXCERPT_WORDS) -> str:
    """Truncate text to max_words, adding ellipsis if cut."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def _format_context(retrieved: List[Tuple[str, float]]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, (text, score) in enumerate(retrieved, start=1):
        cleaned = _truncate(text.strip())
        parts.append(f"[Excerpt {i}, relevance={score:.2f}]\n{cleaned}")
    return "\n\n".join(parts)


def _extractive_answer(state: AgentState, question: str, retrieved: List[Tuple[str, float]]) -> str:
    """
    No-LLM fallback: return the most relevant chunk directly.
    Provides a useful answer even without an API key or when the LLM is unavailable.
    """
    if not retrieved:
        return (
            "No indexed content is available for this paper, "
            "so I can't answer questions about it."
        )

    best_text, best_score = retrieved[0]
    cleaned = _truncate(best_text.strip())

    return (
        f"**[Extractive answer — offline/mock fallback]**\n\n"
        f"Most relevant passage (relevance: {best_score:.2f}):\n\n"
        f"> {cleaned}\n\n"
        f"_Note: Returning top-retrieved chunk directly because no LLM API key is configured or the LLM call was unavailable._"
    )


def answer_question(state: AgentState, question: str, top_k: int = QA_TOP_K) -> str:
    """
    Answer a user question grounded in the paper's chunks and summary.

    1. Retrieve top-k relevant chunks from ChromaDB
    2. Pass to LLM provider (AutoLLMProvider tries Groq with retries & backoff,
       falling back to MockProvider on missing key or runtime errors)
    3. Return synthesized or extractive answer
    """
    retrieved = vectorstore.search(state.collection_name, question, top_k=top_k)

    if not retrieved:
        return (
            "No indexed content is available for this paper, "
            "so I can't answer questions about it."
        )

    context = _format_context(retrieved)

    paper_context = f"Title: {state.selected_paper.title}\n"
    if state.briefing:
        paper_context += f"Summary: {state.briefing.summary}\n"
    else:
        paper_context += f"Abstract: {state.selected_paper.abstract}\n"

    user = (
        f"Paper Info:\n{paper_context}\n\n"
        f"Retrieved excerpts:\n{context}\n\n"
        f"Question: {question}"
    )

    try:
        from . import llm
        provider = llm.get_llm("auto")
        ans = provider.complete(GROUNDED_SYSTEM_PROMPT, user)
        if ans:
            return ans
    except Exception:
        pass

    return _extractive_answer(state, question, retrieved)
