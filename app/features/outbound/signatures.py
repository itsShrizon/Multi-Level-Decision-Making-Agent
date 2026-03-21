"""DSPy signatures for outbound message flows."""

from __future__ import annotations

from typing import Literal

import dspy

FollowUpType = Literal["standard", "urgent", "reminder"]
ReminderType = Literal["standard", "advance", "day_before", "same_day"]
UpdateType = Literal["standard", "progress", "milestone", "requirement"]


class OutboundCheckIn(dspy.Signature):
    """Draft one professional outbound message for a law firm client.

    Silently read the conversation history to gauge mood and tone, then
    write an empathetic, concise weekly check-in that acknowledges any
    stated preferences and the timing context. Output the message text
    only — no quotes, labels, or analysis.
    """

    objective_and_timing: str = dspy.InputField()
    history_json: str = dspy.InputField()
    message: str = dspy.OutputField()


class FollowUp(dspy.Signature):
    """Write a follow-up to an earlier outbound message.

    Keep it brief, professional, and contextually relevant. Don't be pushy.
    If the follow_up_type is 'urgent', convey appropriate urgency without
    losing professionalism. If 'reminder', stay gentle.
    """

    original_message: str = dspy.InputField()
    client_response: str = dspy.InputField(desc="Empty string if no response yet")
    follow_up_type: FollowUpType = dspy.InputField()
    message: str = dspy.OutputField()


class AppointmentReminder(dspy.Signature):
    """Write an appointment reminder.

    Use the timing_context to set the right tone (advance / day-before /
    same-day / standard). Be clear about what the client should do or
    bring, and include contact info for changes.
    """

    appointment_json: str = dspy.InputField()
    client_name: str = dspy.InputField(desc="Empty string if not provided")
    reminder_type: ReminderType = dspy.InputField()
    message: str = dspy.OutputField()


class CaseUpdate(dspy.Signature):
    """Write a case update message.

    Be clear, professional, reassuring. Avoid legal jargon. If action is
    required from the client, state it plainly with the deadline.
    The update_type tunes the framing (progress, milestone, requirement,
    or generic).
    """

    case_json: str = dspy.InputField()
    update_type: UpdateType = dspy.InputField()
    client_context_json: str = dspy.InputField()
    message: str = dspy.OutputField()
