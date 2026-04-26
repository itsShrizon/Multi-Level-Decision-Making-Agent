"""LangGraph DAG for chat analysis.

Topology:

      START
        |
   retrieve_context     <-- prior insights / case state
        |
      triage
       /|\\
      / | \\
sentiment event risk    <-- parallel BSP step
      \\ | /
       \\|/
       decide           <-- conditional routing
        |
   +----+----+----------+
respond     skip     await_human   <-- FLAG+High pauses here via interrupt()
   |         |         |
 critic      |         | (resume with Command(resume={action, reply, reviewer}))
  /\\         |         |
 /  \\        |         |
refine done  |         |
 |    \\     |         |
respond \\   |         |
(loop)   \\  |         |
          \\ |         |
           v v         v
            END
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.core.logging import get_logger
from app.features.agent.state import ChatGraphState
from app.features.chat.context import default_repo
from app.features.chat.services import (
    CriticModule,
    EventModule,
    ResponseModule,
    RiskModule,
    SentimentModule,
    TriageModule,
)
from app.features.chat.signatures import EventDetails

logger = get_logger(__name__)

CRITIC_THRESHOLD = 75
MAX_REFINE = 2

_triage = TriageModule()
_risk = RiskModule()
_sentiment = SentimentModule()
_event = EventModule()
_respond = ResponseModule()
_critic = CriticModule()


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
    return {"sentiment": {"sentiment": out.sentiment, "sentiment_score": out.sentiment_score}}


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


def route_after_decide(state: ChatGraphState) -> Literal["respond", "skip", "await_human"]:
    triage = state["triage"]
    risk = state["risk"]

    if triage["should_ignore"]:
        return "skip"
    if not triage["should_respond"] and not triage["should_flag"]:
        return "skip"
    if triage["should_flag"] and risk["risk_update"] == "High":
        # don't auto-reply; pause for a real human
        return "await_human"
    if not triage["should_respond"]:
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
        critic_notes=state.get("critic_notes", ""),
    )
    return {"reply": reply}


def critic_node(state: ChatGraphState) -> dict[str, Any]:
    triage = state["triage"]
    sentiment = state["sentiment"]
    out = _critic(
        client_message=state["message"],
        sentiment=sentiment["sentiment"],
        is_flagged=triage["should_flag"],
        draft_reply=state.get("reply") or "",
    )
    refine_count = state.get("refine_count", 0)
    logger.info("critic_done", score=out.score, refine_count=refine_count)
    return {
        "critic_score": out.score,
        "critic_notes": out.notes or "",
        "refine_count": refine_count + (1 if out.score < CRITIC_THRESHOLD else 0),
    }


def route_after_critic(state: ChatGraphState) -> Literal["refine", "done"]:
    score = state.get("critic_score") or 0
    if score >= CRITIC_THRESHOLD:
        return "done"
    if state.get("refine_count", 0) > MAX_REFINE:
        return "done"
    return "refine"


def await_human_node(state: ChatGraphState) -> dict[str, Any]:
    """Pause execution until a human resumes the graph.

    The first call lifts a GraphInterrupt with the payload below; the caller
    persists the run by thread_id and shows the reviewer the context. When
    the reviewer is done, resume with:

        graph.invoke(Command(resume={"action": "send", "reply": "...",
                                       "reviewer": "alice@firm.com"}),
                       config={"configurable": {"thread_id": ...}})

    The second invocation re-enters this node and interrupt() returns the
    resume value, which we land into state.
    """
    payload = {
        "reason": "FLAG + High risk — needs human review",
        "client_message": state["message"],
        "triage_reasoning": state["triage"]["reasoning"],
        "risk": state["risk"],
        "sentiment": state["sentiment"],
        "prior_insights": state.get("prior_insights", []),
    }
    decision = interrupt(payload) or {}
    action = decision.get("action") or "skip"
    if action == "send":
        return {"reply": decision.get("reply"), "human_decision": decision}
    return {"reply": None, "human_decision": decision}


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
    g.add_node("critic", critic_node)
    g.add_node("await_human", await_human_node)
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
        {"respond": "respond", "skip": "skip", "await_human": "await_human"},
    )
    g.add_edge("respond", "critic")
    g.add_conditional_edges(
        "critic",
        route_after_critic,
        {"refine": "respond", "done": END},
    )
    g.add_edge("await_human", END)
    g.add_edge("skip", END)

    return g.compile()


chat_graph = build_chat_graph()
