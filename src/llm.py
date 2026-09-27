"""
Groq LLM provider.

Single provider — Groq API (default: qwen/qwen3.8-27b).
Exposes one function: complete(system, user) -> str
"""
from __future__ import annotations

from groq import Groq

from .config import GROQ_API_KEY, GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_TOKENS


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


def complete(system: str, user: str, temperature: float | None = None,
             max_tokens: int | None = None) -> str:
    """
    Send a system + user message pair to Groq and return the assistant's text.
    """
    client = _ensure_client()
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
