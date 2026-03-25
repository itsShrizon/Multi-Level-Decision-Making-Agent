"""Small request-handling helpers used by the API routes."""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


def sanitize_text(text: str | None, max_length: int = 10_000) -> str:
    if not text or not isinstance(text, str):
        return ""
    out = text.strip()
    if len(out) > max_length:
        logger.warning("text_truncated", original=len(out), cap=max_length)
        out = out[:max_length] + "..."
    return out


def truncate_conversation_history(
    messages: list[dict[str, Any]],
    max_length: int = 500,
) -> list[dict[str, Any]]:
    if len(messages) <= max_length:
        return messages
    return messages[-max_length:]


def extract_client_context(
    client_info: dict[str, Any],
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "client_id": client_info.get("client_id", "unknown"),
        "client_name": client_info.get("name", ""),
        "message_count": len(messages),
        "last_message": messages[-1] if messages else None,
        "conversation_span": None,
    }
    if len(messages) > 1:
        first = messages[0].get("timestamp")
        last = messages[-1].get("timestamp")
        if first and last:
            ctx["conversation_span"] = {"start": first, "end": last}
    return ctx
