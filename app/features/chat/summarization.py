"""Chat summarization via DSPy. Cheap-tier LM, deterministic temperature."""

from __future__ import annotations

import asyncio
from typing import Any

import dspy

from app.core.llm import get_lm
from app.core.logging import get_logger

logger = get_logger(__name__)


class _Summarize(dspy.Signature):
    """Summarize a chat conversation in the exact format below.

    Output ONLY this format, no extra prose:

    Chat Summary:
    Summary: The conversation started around "[first 5-7 words of the very first message]..." and the latest point discussed was "[brief summary of the last user message]...". The client seems generally [positive, neutral, or concerned] based on the interaction. Overall, [N] key topics were covered.
    """

    chat_log: str = dspy.InputField(desc="One '<sender>: <text>' per line")
    summary: str = dspy.OutputField()


class ChatSummarizer:
    def __init__(self) -> None:
        self._lm = get_lm("summary")
        self._predict = dspy.Predict(_Summarize)

    async def summarize_chat(self, chat: dict[str, Any]) -> str:
        messages = chat.get("messages") or []
        if not messages:
            raise ValueError("chat conversation is empty")

        chat_log = "\n".join(f"{m['sender']}: {m['text']}" for m in messages)

        def _run() -> str:
            with dspy.context(lm=self._lm):
                return self._predict(chat_log=chat_log).summary.strip()

        summary = await asyncio.to_thread(_run)
        logger.info("chat_summary_done", message_count=len(messages), summary_chars=len(summary))
        return summary
