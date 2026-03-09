"""Light text utilities: concise rewrite, keyword pull, urgency tag."""

from __future__ import annotations

import asyncio
from typing import Literal

import dspy

from app.core.llm import get_lm
from app.core.logging import get_logger

logger = get_logger(__name__)


class _Concise(dspy.Signature):
    """Rewrite text concisely while keeping meaning. Cap at 4 words."""

    text: str = dspy.InputField()
    concise: str = dspy.OutputField()


class _Keywords(dspy.Signature):
    """Extract up to N key terms or phrases as a comma-separated list."""

    text: str = dspy.InputField()
    max_keywords: int = dspy.InputField()
    keywords: str = dspy.OutputField(desc="Comma-separated, no extra commentary")


class _Urgency(dspy.Signature):
    """Classify message urgency. Consider intensity, time pressure, emotion."""

    text: str = dspy.InputField()
    urgency: Literal["Low", "Medium", "High"] = dspy.OutputField()


class TextProcessor:
    def __init__(self) -> None:
        self._lm = get_lm("main")
        self._concise = dspy.Predict(_Concise)
        self._keywords = dspy.Predict(_Keywords)
        self._urgency = dspy.Predict(_Urgency)

    async def make_concise(self, text: str) -> str:
        if not text or not text.strip():
            raise ValueError("text is empty")

        def _run() -> str:
            with dspy.context(lm=self._lm):
                return self._concise(text=text).concise.strip()

        out = await asyncio.to_thread(_run)
        logger.info("concise_done", original=len(text), shrunk=len(out))
        return out

    async def extract_keywords(self, text: str, max_keywords: int = 10) -> list[str]:
        if not text or not text.strip():
            raise ValueError("text is empty")

        def _run() -> str:
            with dspy.context(lm=self._lm):
                return self._keywords(text=text, max_keywords=max_keywords).keywords

        raw = await asyncio.to_thread(_run)
        kws = [k.strip() for k in raw.split(",") if k.strip()][:max_keywords]
        logger.info("keywords_done", count=len(kws))
        return kws

    async def classify_urgency(self, text: str) -> str:
        if not text or not text.strip():
            raise ValueError("text is empty")

        def _run() -> str:
            with dspy.context(lm=self._lm):
                return self._urgency(text=text).urgency

        out = await asyncio.to_thread(_run)
        logger.info("urgency_done", urgency=out)
        return out
