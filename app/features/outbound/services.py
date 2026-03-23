"""Outbound message services — DSPy-powered, scheduler with DI."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import dspy

from app.core.llm import get_lm
from app.core.logging import get_logger
from app.features.outbound.signatures import (
    AppointmentReminder,
    CaseUpdate,
    FollowUp,
    OutboundCheckIn,
)

logger = get_logger(__name__)


def _strip_quotes(text: str) -> str:
    return text.strip().strip('"').strip("'")


class OutboundMessageGenerator:
    """Drafts outbound messages: weekly check-in, follow-up, appointment, case update."""

    def __init__(self) -> None:
        base = get_lm("main")
        # warmer-than-default temperature for natural prose
        self._lm = dspy.LM(
            model=base.model,
            api_key=base.kwargs.get("api_key"),
            temperature=0.5,
            max_tokens=512,
        )
        self._checkin = dspy.Predict(OutboundCheckIn)
        self._followup = dspy.Predict(FollowUp)
        self._appointment = dspy.Predict(AppointmentReminder)
        self._case_update = dspy.Predict(CaseUpdate)

    async def generate_outbound_message(
        self,
        information: str,
        messages: list[dict[str, Any]],
    ) -> str:
        if not information or not information.strip():
            raise ValueError("information is required")
        if not messages:
            raise ValueError("message history is required")

        history = [
            {
                "timestamp": m.get("timestamp"),
                "sender": m.get("sender", "unknown"),
                "content": m.get("content") or m.get("body") or "",
            }
            for m in messages
        ]

        def _run() -> str:
            with dspy.context(lm=self._lm):
                out = self._checkin(
                    objective_and_timing=information,
                    history_json=json.dumps(history, ensure_ascii=False),
                )
            return _strip_quotes(out.message)

        msg = await asyncio.to_thread(_run)
        logger.info("outbound_checkin_done", chars=len(msg), history_n=len(messages))
        return msg

    async def generate_follow_up_message(
        self,
        original_message: str,
        client_response: str | None = None,
        follow_up_type: str = "standard",
    ) -> str:
        def _run() -> str:
            with dspy.context(lm=self._lm):
                out = self._followup(
                    original_message=original_message,
                    client_response=client_response or "",
                    follow_up_type=follow_up_type,
                )
            return _strip_quotes(out.message)

        msg = await asyncio.to_thread(_run)
        logger.info("outbound_followup_done", kind=follow_up_type, had_response=bool(client_response))
        return msg

    async def generate_appointment_reminder(
        self,
        appointment_details: dict[str, Any],
        client_name: str | None = None,
        reminder_type: str = "standard",
    ) -> str:
        def _run() -> str:
            with dspy.context(lm=self._lm):
                out = self._appointment(
                    appointment_json=json.dumps(appointment_details, ensure_ascii=False),
                    client_name=client_name or "",
                    reminder_type=reminder_type,
                )
            return _strip_quotes(out.message)

        msg = await asyncio.to_thread(_run)
        logger.info("outbound_appointment_done", kind=reminder_type, type=appointment_details.get("type", "unknown"))
        return msg

    async def generate_case_update_message(
        self,
        case_info: dict[str, Any],
        update_type: str,
        client_context: dict[str, Any] | None = None,
    ) -> str:
        def _run() -> str:
            with dspy.context(lm=self._lm):
                out = self._case_update(
                    case_json=json.dumps(case_info, ensure_ascii=False),
                    update_type=update_type,
                    client_context_json=json.dumps(client_context or {}, ensure_ascii=False),
                )
            return _strip_quotes(out.message)

        msg = await asyncio.to_thread(_run)
        logger.info("outbound_case_update_done", kind=update_type, case_id=case_info.get("case_id", "unknown"))
        return msg


class MessageScheduler:
    """Lightweight scheduling helpers. The generator is injected so tests
    can stub it without monkeypatching."""

    def __init__(self, generator: OutboundMessageGenerator | None = None) -> None:
        self.generator = generator or OutboundMessageGenerator()

    async def schedule_weekly_checkin(
        self,
        client_id: str,
        message_history: list[dict[str, Any]],
        preferences: dict[str, Any],
    ) -> dict[str, Any]:
        day = preferences.get("preferred_day", "Monday")
        time = preferences.get("preferred_time", "10:00 AM")
        ctx = f"Weekly check-in scheduled for {day} at {time}"

        message = await self.generator.generate_outbound_message(
            information=ctx, messages=message_history
        )
        return {
            "client_id": client_id,
            "message_type": "weekly_checkin",
            "message_content": message,
            "scheduled_day": day,
            "scheduled_time": time,
            "status": "scheduled",
        }

    async def schedule_appointment_reminders(
        self,
        client_id: str,
        appointment_details: dict[str, Any],
        reminder_schedule: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        schedule = reminder_schedule or ["advance", "day_before", "same_day"]
        client_name = appointment_details.get("client_name")

        results = []
        for kind in schedule:
            text = await self.generator.generate_appointment_reminder(
                appointment_details=appointment_details,
                client_name=client_name,
                reminder_type=kind,
            )
            results.append(
                {
                    "client_id": client_id,
                    "message_type": f"appointment_reminder_{kind}",
                    "message_content": text,
                    "appointment_id": appointment_details.get("appointment_id"),
                    "reminder_type": kind,
                    "status": "scheduled",
                }
            )
        return results
