"""Chat analysis services — DSPy modules + a LangGraph orchestrator.

The individual modules (TriageModule, RiskModule, ...) live here so the
graph can import and instantiate them. ChatOrchestrator is a thin wrapper
that runs app.features.agent.chat_graph and reshapes the state into the
dict the API returns.
"""

from __future__ import annotations

from typing import Any

import dspy

from app.core.llm import get_lm
from app.core.logging import get_logger
from app.features.chat.signatures import (
    EventDetect,
    EventDetails,
    GenerateResponse,
    Risk,
    Sentiment,
    Triage,
)

logger = get_logger(__name__)


def _validate_triage(decision: Any) -> None:
    if decision.should_respond and decision.should_ignore:
        decision.should_ignore = False
    if not (decision.should_flag or decision.should_respond or decision.should_ignore):
        decision.should_flag = True


class TriageModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(Triage)

    def forward(self, message: str):
        out = self.predict(message=message)
        _validate_triage(out)
        return out


class RiskModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(Risk)

    def forward(self, message: str, triage_actions: str):
        out = self.predict(message=message, triage_actions=triage_actions)
        out.risk_score = max(0, min(100, int(out.risk_score)))
        return out


class SentimentModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(Sentiment)

    def forward(self, message: str):
        out = self.predict(message=message)
        out.sentiment_score = max(0, min(100, int(out.sentiment_score)))
        return out


class EventModule(dspy.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predict = dspy.Predict(EventDetect)

    def forward(self, message: str):
        return self.predict(message=message)


class ResponseModule(dspy.Module):
    """Hotter LM for natural-sounding replies."""

    def __init__(self) -> None:
        super().__init__()
        base = get_lm("main")
        self._lm_warm = dspy.LM(
            model=base.model,
            api_key=base.kwargs.get("api_key"),
            temperature=0.7,
            max_tokens=512,
        )
        self.predict = dspy.Predict(GenerateResponse)

    def forward(self, **kwargs):
        with dspy.context(lm=self._lm_warm):
            out = self.predict(**kwargs)
        return out.reply.strip().strip('"').strip("'")


class ChatOrchestrator:
    """Drives the chat-analysis LangGraph and shapes the result for the API."""

    def __init__(self) -> None:
        # imported here to avoid an import cycle: chat_graph imports the modules above
        from app.features.agent.chat_graph import chat_graph
        self._graph = chat_graph

    async def analyze_message(
        self,
        client_info: dict[str, Any],
        conversation_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not conversation_history:
            raise ValueError("conversation history is empty")
        current = conversation_history[-1].get("content", "")
        if not current:
            raise ValueError("current message is empty")

        final = await self._graph.ainvoke(
            {
                "message": current,
                "history": conversation_history,
                "client_info": client_info,
            }
        )
        return _shape(final, client_info)


def _shape(state: dict[str, Any], client_info: dict[str, Any]) -> dict[str, Any]:
    triage = state["triage"]
    risk = state["risk"]
    sentiment = state["sentiment"]
    event = state["event"]
    reply = state.get("reply")

    actions: list[str] = []
    if triage["should_flag"]:
        actions.append("FLAG")
    if triage["should_respond"]:
        actions.append("RESPOND")
    if triage["should_ignore"]:
        actions.append("IGNORE")

    event_payload = _coerce_event_details(event)

    logger.info(
        "chat_analysis_done",
        client_id=client_info.get("client_id", "unknown"),
        actions=actions,
        risk=risk["risk_update"],
        sentiment=sentiment["sentiment"],
        has_event=event_payload["has_event"],
    )

    return {
        "actions": actions,
        "triage_reasoning": triage["reasoning"],
        "risk_update": risk["risk_update"],
        "risk_score": risk["risk_score"],
        "sentiment": sentiment["sentiment"],
        "sentiment_score": sentiment["sentiment_score"],
        "response_to_send": reply,
        "event_detection": event_payload,
        "full_analysis": {
            "should_flag": triage["should_flag"],
            "should_respond": triage["should_respond"],
            "should_ignore": triage["should_ignore"],
            "risk_update": risk["risk_update"],
            "sentiment": sentiment["sentiment"],
            "event_detection": event_payload,
        },
    }


def _coerce_event_details(event: dict[str, Any]) -> dict[str, Any]:
    details = event.get("event_details")
    if isinstance(details, EventDetails):
        details = details.model_dump()
    return {
        "has_event": bool(event.get("has_event")),
        "event_details": details,
        "suggested_reminder": event.get("suggested_reminder"),
        "internal_note": event.get("internal_note"),
    }
