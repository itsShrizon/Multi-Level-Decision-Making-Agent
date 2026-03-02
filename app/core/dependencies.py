"""Lightweight DI helpers. Most LM access now goes through app.core.llm.

Legacy shims for the in-flight refactor. The OpenAI/Gemini direct
clients will go away once chat/insights/outbound have been ported
to DSPy modules.
"""

from __future__ import annotations

import time
from functools import lru_cache

import google.generativeai as genai
from openai import AsyncOpenAI

from app.core.config import get_settings


@lru_cache
def get_openai_client() -> AsyncOpenAI:
    """Legacy: direct OpenAI client. Prefer app.core.llm.get_lm()."""
    settings = get_settings()
    return AsyncOpenAI(api_key=settings.OPENAI_API_KEY, timeout=30.0, max_retries=3)


@lru_cache
def get_gemini_client():
    """Legacy: raw Gemini model. Prefer app.core.llm.get_lm('report')."""
    settings = get_settings()
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model_name = settings.LM_REPORT.removeprefix("gemini/")
    return genai.GenerativeModel(
        model_name=model_name,
        generation_config={"temperature": 0.3, "max_output_tokens": 4000},
    )


# in-memory rate limiter — placeholder until slowapi lands in sprint 5
class RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = {}

    def is_allowed(self, key: str, limit: int = 100, window: int = 60) -> bool:
        now = time.time()
        bucket = [t for t in self._hits.get(key, []) if now - t < window]
        if len(bucket) >= limit:
            self._hits[key] = bucket
            return False
        bucket.append(now)
        self._hits[key] = bucket
        return True


_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter
