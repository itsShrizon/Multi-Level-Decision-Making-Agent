"""LangGraph state schemas.

Two graphs live in this package:
  AgentState     — the high-level tool-calling agent (graph.py)
  ChatGraphState — the chat-analysis fan-out DAG (chat_graph.py)
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Sequence, TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


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
