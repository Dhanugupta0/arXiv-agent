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
from typing import List, Optional

from groq import Groq

from .config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS, has_groq_key

# Maximum retries for transient Groq errors
_MAX_RETRIES = 1
_RETRY_BACKOFF = 1.5  # seconds


def _is_transient(exc: Exception) -> bool:
    """Check if an exception is a transient/retryable error (429, timeouts, server errors)."""
    exc_name = type(exc).__name__
    if any(k in exc_name for k in ("RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError")):
        return True
    if hasattr(exc, "status_code") and getattr(exc, "status_code") in (429, 500, 502, 503, 504):
        return True
    exc_str = str(exc).lower()
    return any(
        code in exc_str
        for code in ("429", "rate_limit", "rate limit", "ratelimit", "timeout", "timed out", "503", "502", "500", "connection")
    )


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
        # Extract abstract
        abstract_m = re.search(r"Abstract:\s*(.*?)(?=\n\nFull text|\n\n[A-Z]|\n\n\{|$)", user, re.DOTALL)
        abstract = abstract_m.group(1).strip() if abstract_m else user[:500]

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if s.strip()]
        summary = " ".join(sentences[:3]) if sentences else abstract[:300]
        problem = sentences[0] if sentences else "Problem statement extracted from paper text."

        # Extract method from regex-detected section headers
        method_bullets: List[str] = []
        method_m = re.search(r"\n\s*(?:\d+\.?)?\s*(?:methods?|approach|model|architecture)\b[^\n]*\n", user, re.IGNORECASE)
        if method_m:
            snippet = user[method_m.end(): method_m.end() + 600].strip().replace("\n", " ")
            s_list = [s.strip() for s in re.split(r"(?<=[.!?])\s+", snippet) if s.strip()]
            if s_list:
                method_bullets.append(" ".join(s_list[:2])[:280])
        if not method_bullets:
            method_bullets = ["See abstract and paper introduction for a high-level description of the method."]

        # Extract results from regex-detected section headers
        results_bullets: List[str] = []
        results_m = re.search(r"\n\s*(?:\d+\.?)?\s*(?:results?|experiments?|evaluations?)\b[^\n]*\n", user, re.IGNORECASE)
        if results_m:
            snippet = user[results_m.end(): results_m.end() + 600].strip().replace("\n", " ")
            s_list = [s.strip() for s in re.split(r"(?<=[.!?])\s+", snippet) if s.strip()]
            if s_list:
                results_bullets.append(" ".join(s_list[:2])[:280])
        if not results_bullets:
            results_bullets = [s for s in sentences[3:6]] or ["Key experimental results are detailed in the full paper text."]

        # Extract limitations from regex-detected section headers
        limitations: List[str] = []
        lim_m = re.search(r"\n\s*(?:\d+\.?)?\s*limitations?\b[^\n]*\n", user, re.IGNORECASE)
        if lim_m:
            snippet = user[lim_m.end(): lim_m.end() + 600].strip().replace("\n", " ")
            s_list = [s.strip() for s in re.split(r"(?<=[.!?])\s+", snippet) if s.strip()]
            if s_list:
                limitations.append(" ".join(s_list[:2])[:280])
        if not limitations:
            limitations = ["Limitations not explicitly discussed by the authors; inferred from experimental scope."]

        return json.dumps({
            "summary": summary,
            "problem_statement": problem,
            "method": method_bullets,
            "key_results": results_bullets,
            "limitations": limitations,
            "followup_questions": [
                "What dataset(s) or benchmarks were used to validate the claims?",
                "How does this approach compare against baseline methods?",
                "What are the main compute and memory requirements?"
            ]
        })

    def _mock_qa(self, user: str) -> str:
        # Match chunks formatted as [Excerpt N, ...] ... or [Excerpt N] ...
        excerpts = re.findall(r"\[Excerpt \d+.*?\]\n(.*?)(?=\n\n\[Excerpt|\n\nQuestion:|$)", user, re.DOTALL)
        if excerpts:
            best = excerpts[0].strip()
            return (
                f"**[Extractive answer — offline/mock fallback]**\n\n"
                f"> {best}\n\n"
                f"_Note: Returning top-retrieved chunk directly because no LLM API key is configured or the LLM call was unavailable._"
            )
        return (
            "**[Extractive answer — offline/mock fallback]**\n\n"
            "The provided text does not contain sufficient information to answer this question."
        )


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
                    time.sleep(_RETRY_BACKOFF)
                    continue
                raise last_exc
        raise last_exc


class AutoLLMProvider:
    """Automatically attempts Groq first, falling back to MockProvider on missing key or runtime errors."""

    def __init__(self):
        self.mock = MockProvider()
        self.groq = GroqProvider()
        self.last_provider_used: str = "none"

    def complete(self, system: str, user: str, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        if has_groq_key():
            try:
                content = self.groq.complete(system, user, temperature=temperature, max_tokens=max_tokens)
                self.last_provider_used = "groq"
                return content
            except Exception:
                # Groq failed at runtime (after retries) -> fall back to mock
                self.last_provider_used = "mock"
                return self.mock.complete(system, user, temperature=temperature, max_tokens=max_tokens)
        self.last_provider_used = "mock"
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


