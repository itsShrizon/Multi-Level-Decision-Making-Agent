"""Smoke tests for the DSPy-backed chat modules.

We swap dspy's LM for a deterministic dummy so the tests don't need an
API key and don't make network calls.
"""

from __future__ import annotations

import asyncio
import json

import dspy
import pytest

from app.features.chat.services import (
    ChatOrchestrator,
    EventModule,
    RiskModule,
    SentimentModule,
    TriageModule,
)


class _DummyLM(dspy.LM):
    """Returns a fixed JSON-shaped response for any prompt."""

    def __init__(self, response: dict) -> None:
        super().__init__(model="dummy/dummy")
        self._response = response

    def __call__(self, *args, **kwargs):
        # DSPy's adapter parses the string back into the typed signature
        return [json.dumps(self._response)]


@pytest.fixture(autouse=True)
def _reset_dspy_lm():
    yield
    dspy.settings.configure(lm=None)


def test_triage_validates_mutually_exclusive_actions():
    lm = _DummyLM({
        "should_flag": False,
        "should_respond": True,
        "should_ignore": True,
        "reasoning": "model returned both",
    })
    dspy.configure(lm=lm)
    out = TriageModule()(message="hi")
    assert not (out.should_respond and out.should_ignore)


def test_triage_forces_at_least_one_action_true():
    lm = _DummyLM({
        "should_flag": False,
        "should_respond": False,
        "should_ignore": False,
        "reasoning": "all false from model",
    })
    dspy.configure(lm=lm)
    out = TriageModule()(message="hi")
    assert out.should_flag or out.should_respond or out.should_ignore


def test_risk_score_clamped():
    lm = _DummyLM({"risk_update": "Medium", "risk_score": 999})
    dspy.configure(lm=lm)
    out = RiskModule()(message="anything", triage_actions="FLAG")
    assert 0 <= out.risk_score <= 100


def test_sentiment_score_clamped():
    lm = _DummyLM({"sentiment": "Neutral", "sentiment_score": -50})
    dspy.configure(lm=lm)
    out = SentimentModule()(message="meh")
    assert 0 <= out.sentiment_score <= 100


def test_event_passthrough_when_no_event():
    lm = _DummyLM({
        "has_event": False,
        "event_details": None,
        "suggested_reminder": None,
        "internal_note": None,
    })
    dspy.configure(lm=lm)
    out = EventModule()(message="just checking in")
    assert out.has_event is False


def test_orchestrator_rejects_empty_history():
    orch = ChatOrchestrator()
    with pytest.raises(ValueError):
        asyncio.run(orch.analyze_message({"client_id": "c1"}, []))
