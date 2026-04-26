"""End-to-end tests for the chat-analysis LangGraph.

The graph nodes call DSPy modules under the hood; we monkeypatch each
module's __call__ so the tests run without an LM.

The graph is compiled with a checkpointer, so every invoke needs a
config with a thread_id.
"""

from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace

import pytest
from langgraph.types import Command


def _ns(**kwargs):
    return SimpleNamespace(**kwargs)


def _config():
    return {"configurable": {"thread_id": f"t-{uuid.uuid4()}"}}


@pytest.fixture
def patch_modules(monkeypatch):
    """Patch the module-level singletons in chat_graph with sensible defaults.
    Individual tests can override via monkeypatch as needed.
    """
    from app.features.agent import chat_graph as cg

    monkeypatch.setattr(cg._triage, "__call__", lambda m, **_: _ns(
        should_flag=False, should_respond=True, should_ignore=False, reasoning="auto",
    ))
    monkeypatch.setattr(cg._sentiment, "__call__", lambda m, **_: _ns(
        sentiment="Positive", sentiment_score=20,
    ))
    monkeypatch.setattr(cg._event, "__call__", lambda m, **_: _ns(
        has_event=False, event_details=None, suggested_reminder=None, internal_note=None,
    ))
    monkeypatch.setattr(cg._risk, "__call__", lambda m, ta, **_: _ns(
        risk_update="Low", risk_score=10,
    ))
    monkeypatch.setattr(cg._respond, "__call__", lambda **_: "thanks for reaching out")
    # critic accepts the draft; default to a passing score so no refine loop
    monkeypatch.setattr(cg._critic, "__call__", lambda **_: _ns(score=90, notes=""))
    return cg


def test_graph_full_pipeline_returns_reply(patch_modules):
    cg = patch_modules
    state = asyncio.run(cg.chat_graph.ainvoke(
        {"message": "hi", "history": [], "client_info": {}}, config=_config(),
    ))
    assert state["reply"] == "thanks for reaching out"
    assert state["critic_score"] == 90


def test_graph_skips_on_ignore(patch_modules, monkeypatch):
    cg = patch_modules
    monkeypatch.setattr(cg._triage, "__call__", lambda m, **_: _ns(
        should_flag=False, should_respond=False, should_ignore=True, reasoning="closer",
    ))
    state = asyncio.run(cg.chat_graph.ainvoke(
        {"message": "ok", "history": [], "client_info": {}}, config=_config(),
    ))
    assert state["reply"] is None


def test_graph_loops_on_low_critic_score(patch_modules, monkeypatch):
    cg = patch_modules
    calls = {"respond": 0, "critic": 0}

    def _respond(**_):
        calls["respond"] += 1
        return f"draft {calls['respond']}"

    # First two critic calls fail, third passes — should trigger 2 refines
    def _critic(**_):
        calls["critic"] += 1
        if calls["critic"] < 3:
            return _ns(score=50, notes="too short")
        return _ns(score=90, notes="")

    monkeypatch.setattr(cg._respond, "__call__", _respond)
    monkeypatch.setattr(cg._critic, "__call__", _critic)

    state = asyncio.run(cg.chat_graph.ainvoke(
        {"message": "hi", "history": [], "client_info": {}}, config=_config(),
    ))
    assert calls["respond"] == 3
    assert calls["critic"] == 3
    assert state["reply"] == "draft 3"


def test_graph_bails_after_max_refine(patch_modules, monkeypatch):
    cg = patch_modules
    monkeypatch.setattr(cg._critic, "__call__", lambda **_: _ns(score=10, notes="bad"))
    state = asyncio.run(cg.chat_graph.ainvoke(
        {"message": "hi", "history": [], "client_info": {}}, config=_config(),
    ))
    # MAX_REFINE = 2, so respond runs at most 1 + 2 = 3 times. After that we exit.
    assert state["refine_count"] >= cg.MAX_REFINE
    assert state["reply"] is not None  # we keep the latest draft


def test_graph_pauses_on_high_risk_flag(patch_modules, monkeypatch):
    cg = patch_modules
    monkeypatch.setattr(cg._triage, "__call__", lambda m, **_: _ns(
        should_flag=True, should_respond=True, should_ignore=False, reasoning="urgent",
    ))
    monkeypatch.setattr(cg._risk, "__call__", lambda m, ta, **_: _ns(
        risk_update="High", risk_score=85,
    ))

    config = _config()
    result = asyncio.run(cg.chat_graph.ainvoke(
        {"message": "I want out", "history": [], "client_info": {}}, config=config,
    ))
    # The first invocation pauses at await_human; the snapshot should show next=("await_human",)
    snapshot = asyncio.run(cg.chat_graph.aget_state(config))
    assert "await_human" in snapshot.next

    # Resume with a reviewer's decision; reply should land on state
    final = asyncio.run(cg.chat_graph.ainvoke(
        Command(resume={"action": "send", "reply": "we are on it", "reviewer": "alice"}),
        config=config,
    ))
    assert final["reply"] == "we are on it"
    assert final["human_decision"]["reviewer"] == "alice"
