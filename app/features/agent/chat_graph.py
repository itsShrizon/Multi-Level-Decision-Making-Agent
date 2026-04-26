"""LangGraph DAG for chat analysis.

Topology:

      START
        |
   retrieve_context     <-- pulls prior insights / case state into state
        |
      triage
       /|\\
      / | \\
sentiment event risk    <-- parallel BSP step
      \\ | /
       \\|/
       decide           <-- no-op join, conditional edge picks the next step
        |
   +----+----+
respond     skip
   |         |
   +----+----+
        v
       END
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from app.features.agent.state import ChatGraphState
from app.features.chat.context import default_repo
from app.features.chat.services import (
    EventModule,
    ResponseModule,
    RiskModule,
    SentimentModule,
    TriageModule,
)
from app.features.chat.signatures import EventDetails

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


async def retrieve_context_node(state: ChatGraphState) -> dict[str, Any]:
    """Look up prior insights / sentiment / case data before triage runs.

    Empty by default; ContextRepo is the swap point for real persistence.
    """
    client_id = (state.get("client_info") or {}).get("client_id", "")
    if not client_id:
        return {"prior_insights": [], "open_case_data": {}, "last_sentiment": None, "last_risk": None}
    ctx = await default_repo.fetch_for_client(client_id)
    return {
        "prior_insights": ctx.get("prior_insights", []),
        "open_case_data": ctx.get("open_case_data", {}),
        "last_sentiment": ctx.get("last_sentiment"),
        "last_risk": ctx.get("last_risk"),
    }


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


def decide_node(state: ChatGraphState) -> dict[str, Any]:
    return {}


def route_after_decide(state: ChatGraphState) -> Literal["respond", "skip"]:
    triage = state["triage"]
    risk = state["risk"]

    if triage["should_ignore"]:
        return "skip"
    if not triage["should_respond"]:
        return "skip"
    if triage["should_flag"] and risk["risk_update"] == "High":
        return "skip"
    return "respond"


def respond_node(state: ChatGraphState) -> dict[str, Any]:
    triage = state["triage"]
    sentiment = state["sentiment"]
    reply = _respond(
        last_message=state["message"],
        triage_reasoning=triage["reasoning"],
        sentiment=sentiment["sentiment"],
        sentiment_score=sentiment["sentiment_score"],
        is_flagged=triage["should_flag"],
    )
    return {"reply": reply}


def skip_node(state: ChatGraphState) -> dict[str, Any]:
    return {"reply": None}


def build_chat_graph():
    g = StateGraph(ChatGraphState)
    g.add_node("retrieve_context", retrieve_context_node)
    g.add_node("triage", triage_node)
    g.add_node("sentiment", sentiment_node)
    g.add_node("event", event_node)
    g.add_node("risk", risk_node)
    g.add_node("decide", decide_node)
    g.add_node("respond", respond_node)
    g.add_node("skip", skip_node)

    g.add_edge(START, "retrieve_context")
    g.add_edge("retrieve_context", "triage")
    g.add_edge("triage", "sentiment")
    g.add_edge("triage", "event")
    g.add_edge("triage", "risk")
    g.add_edge("sentiment", "decide")
    g.add_edge("event", "decide")
    g.add_edge("risk", "decide")

    g.add_conditional_edges(
        "decide",
        route_after_decide,
        {"respond": "respond", "skip": "skip"},
    )
    g.add_edge("respond", END)
    g.add_edge("skip", END)

    return g.compile()


chat_graph = build_chat_graph()
