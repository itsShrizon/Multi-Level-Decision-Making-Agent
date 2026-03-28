"""Outbound message routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.features.outbound.services import MessageScheduler, OutboundMessageGenerator
from app.shared.http import ok
from app.shared.schemas import OutboundMessageRequest, OutboundMessageResult
from app.shared.utils import sanitize_text

logger = get_logger(__name__)
router = APIRouter()


# Endpoint-specific request models stay co-located with the routes that use them.
class FollowUpRequest(BaseModel):
    original_message: str
    client_response: str | None = None
    follow_up_type: str = "standard"


class AppointmentReminderRequest(BaseModel):
    appointment_details: dict[str, Any]
    client_name: str | None = None
    reminder_type: str = "standard"


class CaseUpdateRequest(BaseModel):
    case_info: dict[str, Any]
    update_type: str
    client_context: dict[str, Any] | None = None


class WeeklyCheckinRequest(BaseModel):
    client_id: str
    message_history: list[dict[str, Any]]
    preferences: dict[str, Any]


class AppointmentReminderScheduleRequest(BaseModel):
    client_id: str
    appointment_details: dict[str, Any]
    reminder_schedule: list[str] | None = None


def get_generator() -> OutboundMessageGenerator:
    return OutboundMessageGenerator()


def get_scheduler() -> MessageScheduler:
    return MessageScheduler()


def _clean_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        text = sanitize_text(m.get("content") or m.get("body", ""))
        if text:
            out.append(
                {
                    "timestamp": m.get("timestamp"),
                    "sender": m.get("sender", "unknown"),
                    "content": text,
                }
            )
    return out


@router.post("/generate", response_model=OutboundMessageResult)
async def generate_outbound_message(
    request: OutboundMessageRequest,
    generator: OutboundMessageGenerator = Depends(get_generator),
):
    info = sanitize_text(request.information)
    if not info:
        raise ValidationError("information is required")
    history = _clean_history(request.messages)
    if not history:
        raise ValidationError("no valid messages after sanitization")

    message = await generator.generate_outbound_message(information=info, messages=history)
    return ok({"message": message}, processed_messages=len(history), message_length=len(message))


@router.post("/follow-up")
async def generate_follow_up(
    request: FollowUpRequest,
    generator: OutboundMessageGenerator = Depends(get_generator),
):
    original = sanitize_text(request.original_message)
    if not original:
        raise ValidationError("original_message is required")

    follow_up = await generator.generate_follow_up_message(
        original_message=original,
        client_response=sanitize_text(request.client_response) if request.client_response else None,
        follow_up_type=request.follow_up_type,
    )
    return ok({"message": follow_up}, follow_up_type=request.follow_up_type)


@router.post("/appointment-reminder")
async def generate_appointment_reminder(
    request: AppointmentReminderRequest,
    generator: OutboundMessageGenerator = Depends(get_generator),
):
    if not request.appointment_details:
        raise ValidationError("appointment_details is required")

    reminder = await generator.generate_appointment_reminder(
        appointment_details=request.appointment_details,
        client_name=request.client_name,
        reminder_type=request.reminder_type,
    )
    return ok({"message": reminder}, reminder_type=request.reminder_type)


@router.post("/case-update")
async def generate_case_update(
    request: CaseUpdateRequest,
    generator: OutboundMessageGenerator = Depends(get_generator),
):
    if not request.case_info:
        raise ValidationError("case_info is required")
    if not request.update_type:
        raise ValidationError("update_type is required")

    message = await generator.generate_case_update_message(
        case_info=request.case_info,
        update_type=request.update_type,
        client_context=request.client_context,
    )
    return ok({"message": message}, update_type=request.update_type)


@router.post("/schedule/weekly-checkin")
async def schedule_weekly_checkin(
    request: WeeklyCheckinRequest,
    scheduler: MessageScheduler = Depends(get_scheduler),
):
    if not request.client_id:
        raise ValidationError("client_id is required")
    if not request.message_history:
        raise ValidationError("message_history is required")

    info = await scheduler.schedule_weekly_checkin(
        client_id=request.client_id,
        message_history=request.message_history,
        preferences=request.preferences,
    )
    return ok(info, client_id=request.client_id, schedule_created=True)


@router.post("/schedule/appointment-reminders")
async def schedule_appointment_reminders(
    request: AppointmentReminderScheduleRequest,
    scheduler: MessageScheduler = Depends(get_scheduler),
):
    if not request.client_id:
        raise ValidationError("client_id is required")
    if not request.appointment_details:
        raise ValidationError("appointment_details is required")

    reminders = await scheduler.schedule_appointment_reminders(
        client_id=request.client_id,
        appointment_details=request.appointment_details,
        reminder_schedule=request.reminder_schedule,
    )
    return ok({"scheduled_reminders": reminders}, reminders_scheduled=len(reminders))


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "outbound_messaging"}
