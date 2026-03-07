"""Chat analysis services — DSPy-powered.

Each agent is a tiny dspy.Module wrapping one signature. ChatOrchestrator
runs the independent ones in parallel via asyncio.to_thread (DSPy modules
are sync), then chains risk and reply on top.
"""

from __future__ import annotations

import asyncio
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


def _actions_str(decision: Any) -> str:
    actions = []
    if decision.should_flag:
        actions.append("FLAG")
    if decision.should_respond:
        actions.append("RESPOND")
    if decision.should_ignore:
        actions.append("IGNORE")
    return ", ".join(actions) or "NONE"


def _validate_triage(decision: Any) -> None:
    if decision.should_respond and decision.should_ignore:
        # don't crash — coerce to a single sane choice
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
    """Runs the four analyses, then risk + (optional) reply on top."""

    def __init__(self) -> None:
        self.triage = TriageModule()
        self.risk = RiskModule()
        self.sentiment = SentimentModule()
        self.event = EventModule()
        self.respond = ResponseModule()

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

        triage, sentiment, event = await asyncio.gather(
            asyncio.to_thread(self.triage, current),
            asyncio.to_thread(self.sentiment, current),
            asyncio.to_thread(self.event, current),
        )

        risk = await asyncio.to_thread(self.risk, current, _actions_str(triage))

        reply: str | None = None
        wants_reply = triage.should_respond and not triage.should_ignore
        if wants_reply and not (triage.should_flag and risk.risk_update == "High"):
            reply = await asyncio.to_thread(
                self.respond,
                last_message=current,
                triage_reasoning=triage.reasoning,
                sentiment=sentiment.sentiment,
                sentiment_score=sentiment.sentiment_score,
                is_flagged=triage.should_flag,
            )

        actions = _actions_str(triage).split(", ")
        event_payload = self._event_payload(event)

        logger.info(
            "chat_analysis_done",
            client_id=client_info.get("client_id", "unknown"),
            actions=actions,
            risk=risk.risk_update,
            sentiment=sentiment.sentiment,
            has_event=event_payload["has_event"],
        )

        return {
            "actions": actions,
            "triage_reasoning": triage.reasoning,
            "risk_update": risk.risk_update,
            "risk_score": risk.risk_score,
            "sentiment": sentiment.sentiment,
            "sentiment_score": sentiment.sentiment_score,
            "response_to_send": reply,
            "event_detection": event_payload,
            "full_analysis": {
                "should_flag": triage.should_flag,
                "should_respond": triage.should_respond,
                "should_ignore": triage.should_ignore,
                "risk_update": risk.risk_update,
                "sentiment": sentiment.sentiment,
                "event_detection": event_payload,
            },
        }

    @staticmethod
    def _event_payload(event: Any) -> dict[str, Any]:
        details = event.event_details
        if isinstance(details, EventDetails):
            details = details.model_dump()
        return {
            "has_event": bool(event.has_event),
            "event_details": details,
            "suggested_reminder": event.suggested_reminder,
            "internal_note": event.internal_note,
        }
