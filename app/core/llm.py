"""DSPy LM provider — one place to wire up the language models.

Tiers:
  main    — chat triage, risk, sentiment, response generation
  fast    — micro insights, classification, anything cheap
  summary — chat summarization
  report  — long-form firm-wide reports (Gemini Pro)

Callers ask for a tier; we hand back a configured dspy.LM. The default tier
is also installed onto dspy globally so any signature picks it up unless
overridden via dspy.context(lm=...).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import dspy

from app.core.config import get_settings

Tier = Literal["main", "fast", "summary", "report"]


def _api_key_for(model: str) -> str | None:
    settings = get_settings()
    if model.startswith("openai/"):
        return settings.OPENAI_API_KEY
    if model.startswith("gemini/"):
        return settings.GEMINI_API_KEY
    return None


def _model_for(tier: Tier) -> str:
    settings = get_settings()
    return {
        "main": settings.LM_MAIN,
        "fast": settings.LM_FAST,
        "summary": settings.LM_SUMMARY,
        "report": settings.LM_REPORT,
    }[tier]


@lru_cache(maxsize=8)
def get_lm(tier: Tier = "main") -> dspy.LM:
    settings = get_settings()
    model = _model_for(tier)
    api_key = _api_key_for(model)
    if api_key is None and model.startswith(("openai/", "gemini/")):
        raise RuntimeError(f"no API key configured for {model}")

    return dspy.LM(
        model=model,
        api_key=api_key,
        max_tokens=settings.LM_MAX_TOKENS,
        temperature=settings.LM_TEMPERATURE,
        timeout=settings.LM_TIMEOUT_S,
    )


def configure_default_lm() -> None:
    """Install the main-tier LM as DSPy's global default. Call once at startup."""
    dspy.configure(lm=get_lm("main"))
