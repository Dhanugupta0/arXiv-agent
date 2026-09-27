"""
Groq LLM provider.

Single provider — Groq API (default: llama-3.3-70b-versatile).
Exposes one function: complete(system, user) -> str

Includes retry with exponential backoff for transient errors
(429 rate limits, timeouts) since Groq is the only external
dependency now that embeddings are local.
"""
from __future__ import annotations

import time

from groq import Groq

from .config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS, has_groq_key

# Maximum retries for transient Groq errors
_MAX_RETRIES = 2
_RETRY_BACKOFF_BASE = 2.0  # seconds


def _get_client() -> Groq:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file — get a free key at https://console.groq.com"
        )
    return Groq(api_key=GROQ_API_KEY)


_client: Groq | None = None


def _ensure_client() -> Groq:
    global _client
    if _client is None:
        _client = _get_client()
    return _client


def _is_transient(exc: Exception) -> bool:
    """Check if an exception is a transient/retryable error."""
    exc_str = str(exc).lower()
    # 429 rate limit, 500/502/503 server errors, timeouts
    if any(code in exc_str for code in ("429", "rate_limit", "rate limit")):
        return True
    if any(code in exc_str for code in ("timeout", "timed out", "503", "502", "500")):
        return True
    return False


def complete(system: str, user: str, temperature: float | None = None,
             max_tokens: int | None = None) -> str:
    """
    Send a system + user message pair to Groq and return the assistant's text.

    Retries up to _MAX_RETRIES times on transient errors (429/timeout)
    with exponential backoff.
    """
    client = _ensure_client()
    last_exc = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature or GROQ_TEMPERATURE,
                max_tokens=max_tokens or GROQ_MAX_TOKENS,
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_exc = e
            if attempt < _MAX_RETRIES and _is_transient(e):
                wait = _RETRY_BACKOFF_BASE * (2 ** attempt)
                time.sleep(wait)
                continue
            raise

    raise last_exc  # should not reach here, but satisfy type checker
