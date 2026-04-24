"""End-to-end tests for the chat-analysis LangGraph.

We monkeypatch each module's `forward` so the graph runs without an LM.
Coverage targets the full BSP topology and the conditional routing.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import dspy
import pytest


@pytest.fixture
def graph(monkeypatch):
    # Import lazily so monkeypatch can land before module-level singletons read settings
    from app.features.agent import chat_graph as cg

    def _triage(message, **_):
        return SimpleNamespace(
            should_flag=False,
            should_respond=True,
            should_ignore=False,
            reasoning="auto",
        )

    def _sentiment(message, **_):
        return SimpleNamespace(sentiment="Positive", sentiment_score=20)

    def _event(message, **_):
        return SimpleNamespace(
            has_event=False,
            event_details=None,
            suggested_reminder=None,
            internal_note=None,
        )

    def _risk(message, triage_actions, **_):
        return SimpleNamespace(risk_update="Low", risk_score=10)

    def _respond(**_):
        return "thanks for reaching out"

    monkeypatch.setattr(cg._triage, "__call__", _triage)
    monkeypatch.setattr(cg._sentiment, "__call__", _sentiment)
    monkeypatch.setattr(cg._event, "__call__", _event)
    monkeypatch.setattr(cg._risk, "__call__", _risk)
    monkeypatch.setattr(cg._respond, "__call__", _respond)
    return cg.chat_graph


def test_graph_full_pipeline_returns_reply(graph):
    state = asyncio.run(graph.ainvoke({"message": "hi", "history": [], "client_info": {}}))
    assert state["triage"]["should_respond"] is True
    assert state["sentiment"]["sentiment"] == "Positive"
    assert state["risk"]["risk_update"] == "Low"
    assert state["event"]["has_event"] is False
    assert state["reply"] == "thanks for reaching out"


def test_graph_skips_reply_on_ignore(monkeypatch):
    from app.features.agent import chat_graph as cg

    monkeypatch.setattr(cg._triage, "__call__", lambda message, **_: SimpleNamespace(
        should_flag=False, should_respond=False, should_ignore=True, reasoning="closer",
    ))
    monkeypatch.setattr(cg._sentiment, "__call__", lambda message, **_: SimpleNamespace(
        sentiment="Neutral", sentiment_score=50,
    ))
    monkeypatch.setattr(cg._event, "__call__", lambda message, **_: SimpleNamespace(
        has_event=False, event_details=None, suggested_reminder=None, internal_note=None,
    ))
    monkeypatch.setattr(cg._risk, "__call__", lambda message, triage_actions, **_: SimpleNamespace(
        risk_update="Low", risk_score=5,
    ))

    state = asyncio.run(cg.chat_graph.ainvoke({"message": "ok", "history": [], "client_info": {}}))
    assert state["reply"] is None


def test_graph_skips_reply_on_high_risk_flag(monkeypatch):
    from app.features.agent import chat_graph as cg

    monkeypatch.setattr(cg._triage, "__call__", lambda message, **_: SimpleNamespace(
        should_flag=True, should_respond=False, should_ignore=False, reasoning="urgent",
    ))
    monkeypatch.setattr(cg._sentiment, "__call__", lambda message, **_: SimpleNamespace(
        sentiment="Negative", sentiment_score=85,
    ))
    monkeypatch.setattr(cg._event, "__call__", lambda message, **_: SimpleNamespace(
        has_event=False, event_details=None, suggested_reminder=None, internal_note=None,
    ))
    monkeypatch.setattr(cg._risk, "__call__", lambda message, triage_actions, **_: SimpleNamespace(
        risk_update="High", risk_score=85,
    ))

    state = asyncio.run(cg.chat_graph.ainvoke({"message": "I want out", "history": [], "client_info": {}}))
    assert state["reply"] is None
