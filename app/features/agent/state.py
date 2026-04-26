"""LangGraph state schema for the chat-analysis DAG."""

from __future__ import annotations

from typing import Any, TypedDict


class ChatGraphState(TypedDict, total=False):
    # inputs
    message: str
    history: list[dict[str, Any]]
    client_info: dict[str, Any]

    # context fetched at the start of the run
    prior_insights: list[str]
    open_case_data: dict[str, Any]
    last_sentiment: str | None
    last_risk: str | None

    # outputs from the parallel branch nodes
    triage: dict[str, Any]
    sentiment: dict[str, Any]
    event: dict[str, Any]
    risk: dict[str, Any]

    # reply pipeline
    reply: str | None
    critic_score: int | None
    critic_notes: str
    refine_count: int

    # human-in-the-loop result (set when graph resumes from await_human)
    # shape: {"action": "send" | "skip", "reply": str | None, "reviewer": str}
    human_decision: dict[str, Any] | None
