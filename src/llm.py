"""
LLM provider module supporting Groq API with automatic MockProvider fallback.

- GroqProvider: Uses Groq API (default: qwen/qwen3.8-27b) with retries & backoff.
- MockProvider: Extractive zero-key, zero-network fallback provider.
- get_llm("auto"): Returns AutoLLMProvider which attempts Groq first and falls back to MockProvider on missing key or runtime errors.
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

from groq import Groq

from .config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS, has_groq_key

# Maximum retries for transient Groq errors
_MAX_RETRIES = 2
_RETRY_BACKOFF_BASE = 1.5  # seconds


def _is_transient(exc: Exception) -> bool:
    """Check if an exception is a transient/retryable error (429, timeouts, server errors)."""
    exc_str = str(exc).lower()
    if any(code in exc_str for code in ("429", "rate_limit", "rate limit")):
        return True
    if any(code in exc_str for code in ("timeout", "timed out", "503", "502", "500", "connection")):
        return True
    return False


class MockProvider:
    """Zero-key, zero-network fallback provider."""

    def complete(self, system: str, user: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        # Check if prompt requests JSON briefing
        if "summary" in user.lower() and "problem_statement" in user.lower() or "json" in system.lower() or "json" in user.lower():
            return self._mock_briefing(user)
        # Check if prompt requests candidate ranking
        if "candidates:" in user.lower() or "which number" in user.lower():
            return "1"
        # Standard QA or other prompt
        return self._mock_qa(user)

    def _mock_briefing(self, user: str) -> str:
        abstract_m = re.search(r"Abstract:\s*(.*?)(?=\n\n|\n[A-Z]|$)", user, re.DOTALL)
        abstract = abstract_m.group(1).strip() if abstract_m else user[:500]

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if s.strip()]
        summary = " ".join(sentences[:3]) if sentences else abstract[:300]
        problem = sentences[0] if sentences else "Problem statement extracted from paper text."

        return json.dumps({
            "summary": summary,
            "problem_statement": problem,
            "method": ["Method extracted from paper text."],
            "key_results": ["Key results extracted from paper text."],
            "limitations": ["Limitations not explicitly detailed in short excerpt."],
            "followup_questions": [
                "What dataset(s) or benchmarks were used?",
                "How does this compare against baseline methods?",
                "What are the main compute requirements?"
            ]
        })

    def _mock_qa(self, user: str) -> str:
        excerpts = re.findall(r"\[Excerpt \d+.*?\]\n(.*?)(?=\n\n\[Excerpt|\n\nQuestion:|$)", user, re.DOTALL)
        if excerpts:
            best = excerpts[0].strip()
            return f"**[Extractive answer — offline/mock fallback]**\n\n> {best}"
        return "**[Offline/mock fallback]** Extracted paper information."


class GroqProvider:
    """Groq API provider with retries."""

    def __init__(self):
        self._client: Optional[Groq] = None

    def _get_client(self) -> Groq:
        if self._client is None:
            if not GROQ_API_KEY:
                raise RuntimeError("GROQ_API_KEY is not set.")
            self._client = Groq(api_key=GROQ_API_KEY)
        return self._client

    def complete(self, system: str, user: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        client = self._get_client()
        last_exc = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                resp = client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=temperature if temperature is not None else GROQ_TEMPERATURE,
                    max_tokens=max_tokens if max_tokens is not None else GROQ_MAX_TOKENS,
                )
                content = resp.choices[0].message.content
                if content:
                    return content
                raise ValueError("Empty response from Groq API")
            except Exception as e:
                last_exc = e
                if attempt < _MAX_RETRIES and _is_transient(e):
                    time.sleep(_RETRY_BACKOFF_BASE * (2 ** attempt))
                    continue
                raise last_exc
        raise last_exc


class AutoLLMProvider:
    """Automatically attempts Groq first, falling back to MockProvider on missing key or runtime errors."""

    def __init__(self):
        self.mock = MockProvider()
        self.groq = GroqProvider()

    def complete(self, system: str, user: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        if has_groq_key():
            try:
                return self.groq.complete(system, user, temperature=temperature, max_tokens=max_tokens)
            except Exception:
                # Runtime failure (e.g. rate limit, connection drop, invalid model) -> fall back to mock
                pass
        return self.mock.complete(system, user, temperature=temperature, max_tokens=max_tokens)


def get_llm(mode: str = "auto"):
    """
    LLM Factory function.
    Returns an LLM provider based on mode ('auto', 'groq', 'mock').
    """
    if mode == "groq":
        return GroqProvider()
    if mode == "mock":
        return MockProvider()
    return AutoLLMProvider()


def complete(system: str, user: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
    """Convenience function using get_llm('auto')."""
    return get_llm("auto").complete(system, user, temperature=temperature, max_tokens=max_tokens)

