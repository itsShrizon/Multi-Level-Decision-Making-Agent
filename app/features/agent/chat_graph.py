"""LangGraph DAG for chat analysis.

Topology:

      START
        │
      triage
       /│\\
      / │ \\
sentiment event risk     <-- run in parallel (BSP step)
      \\ │ /
       \\│/
       respond           <-- merges all branches, decides reply
        │
       END
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.features.agent.state import ChatGraphState
from app.features.chat.services import (
    EventModule,
    ResponseModule,
    RiskModule,
    SentimentModule,
    TriageModule,
)
from app.features.chat.signatures import EventDetails

# module singletons — DSPy modules are cheap but not free to instantiate
_triage = TriageModule()
_risk = RiskModule()
_sentiment = SentimentModule()
_event = EventModule()
_respond = ResponseModule()


def _actions(triage: dict[str, Any]) -> list[str]:
    out = []
    if triage["should_flag"]:
        out.append("FLAG")
    if triage["should_respond"]:
        out.append("RESPOND")
    if triage["should_ignore"]:
        out.append("IGNORE")
    return out


def triage_node(state: ChatGraphState) -> dict[str, Any]:
    out = _triage(state["message"])
    return {
        "triage": {
            "should_flag": out.should_flag,
            "should_respond": out.should_respond,
            "should_ignore": out.should_ignore,
            "reasoning": out.reasoning,
        }
    }


def sentiment_node(state: ChatGraphState) -> dict[str, Any]:
    out = _sentiment(state["message"])
    return {
        "sentiment": {
            "sentiment": out.sentiment,
            "sentiment_score": out.sentiment_score,
        }
    }


def event_node(state: ChatGraphState) -> dict[str, Any]:
    out = _event(state["message"])
    details = out.event_details
    if isinstance(details, EventDetails):
        details = details.model_dump()
    return {
        "event": {
            "has_event": bool(out.has_event),
            "event_details": details,
            "suggested_reminder": out.suggested_reminder,
            "internal_note": out.internal_note,
        }
    }


def risk_node(state: ChatGraphState) -> dict[str, Any]:
    triage = state["triage"]
    actions = ", ".join(_actions(triage)) or "NONE"
    out = _risk(state["message"], actions)
    return {"risk": {"risk_update": out.risk_update, "risk_score": out.risk_score}}


def respond_node(state: ChatGraphState) -> dict[str, Any]:
    """Generate a reply if the triage + risk combo calls for one."""
    triage = state["triage"]
    risk = state["risk"]
    sentiment = state["sentiment"]

    wants_reply = triage["should_respond"] and not triage["should_ignore"]
    high_flag_only = triage["should_flag"] and risk["risk_update"] == "High" and not triage["should_respond"]
    if not wants_reply or high_flag_only:
        return {"reply": None}

    reply = _respond(
        last_message=state["message"],
        triage_reasoning=triage["reasoning"],
        sentiment=sentiment["sentiment"],
        sentiment_score=sentiment["sentiment_score"],
        is_flagged=triage["should_flag"],
    )
    return {"reply": reply}


def build_chat_graph():
    g = StateGraph(ChatGraphState)
    g.add_node("triage", triage_node)
    g.add_node("sentiment", sentiment_node)
    g.add_node("event", event_node)
    g.add_node("risk", risk_node)
    g.add_node("respond", respond_node)

    g.add_edge(START, "triage")
    # fan out — these three run in the same BSP step
    g.add_edge("triage", "sentiment")
    g.add_edge("triage", "event")
    g.add_edge("triage", "risk")
    # join — respond runs once, after all three above land
    g.add_edge("sentiment", "respond")
    g.add_edge("event", "respond")
    g.add_edge("risk", "respond")
    g.add_edge("respond", END)

    return g.compile()


chat_graph = build_chat_graph()
