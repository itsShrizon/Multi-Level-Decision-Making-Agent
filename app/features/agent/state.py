"""LangGraph state schema for the chat-analysis DAG."""

from __future__ import annotations

from typing import Any, TypedDict


class ChatGraphState(TypedDict, total=False):
    # inputs
    message: str
    history: list[dict[str, Any]]
    client_info: dict[str, Any]

    # outputs from the parallel branch nodes
    triage: dict[str, Any]
    sentiment: dict[str, Any]
    event: dict[str, Any]
    risk: dict[str, Any]

    # final reply (None if we deliberately skipped it)
    reply: str | None
