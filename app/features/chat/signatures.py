"""DSPy signatures for chat analysis.

Each signature is a typed I/O contract — the docstring becomes the
instruction the LM sees, fields define the structured payload. The actual
modules that call these signatures live in services.py.
"""

from __future__ import annotations

from typing import Literal

import dspy
from pydantic import BaseModel, Field


class EventDetails(BaseModel):
    date: str | None = None
    time: str | None = None
    location: str | None = None
    event_type: str | None = None
    additional_info: str | None = None


class Triage(dspy.Signature):
    """Triage a client message for a law firm.

    Decide which actions apply. should_respond and should_ignore are
    mutually exclusive. should_flag may combine with either.
    At least one action must be true.

    Flag for: legal/medical advice questions, extreme distress, new
    injuries, threats to leave, requests to speak with a person.
    Ignore only for content-free closers: "ok", "thanks", etc.
    Respond when an immediate auto-reply would help.
    """

    message: str = dspy.InputField(desc="The client's most recent message")

    should_flag: bool = dspy.OutputField(desc="Needs human attention")
    should_respond: bool = dspy.OutputField(desc="Send an automated reply")
    should_ignore: bool = dspy.OutputField(desc="Conversation closer with no new info")
    reasoning: str = dspy.OutputField(desc="One sentence on why")


class Risk(dspy.Signature):
    """Assess client retention risk from the message and triage outcome.

    High (70-100): direct threats to leave, malpractice accusations,
        frantic urgency, financial-aid requests, suicidal ideation.
    Medium (40-69): frustration, negative sentiment, vague dissatisfaction,
        or anything flagged but not High.
    Low (0-39): neutral or positive.
    """

    message: str = dspy.InputField()
    triage_actions: str = dspy.InputField(desc="Comma-separated subset of FLAG, RESPOND, IGNORE")

    risk_update: Literal["Low", "Medium", "High"] = dspy.OutputField()
    risk_score: int = dspy.OutputField(desc="0-100 inside the band for risk_update")


class Sentiment(dspy.Signature):
    """Classify message sentiment with an intensity score.

    Positive: 0-30, Neutral: 31-60, Negative: 61-100.
    The score should track intensity within its band.
    """

    message: str = dspy.InputField()
    sentiment: Literal["Positive", "Neutral", "Negative"] = dspy.OutputField()
    sentiment_score: int = dspy.OutputField(desc="0-100, see band rules")


class EventDetect(dspy.Signature):
    """Find any future event or appointment mention.

    If found, extract details and draft a short reminder + an internal note.
    If nothing event-like, set has_event=false and leave the rest empty.
    """

    message: str = dspy.InputField()
    has_event: bool = dspy.OutputField()
    event_details: EventDetails | None = dspy.OutputField()
    suggested_reminder: str | None = dspy.OutputField()
    internal_note: str | None = dspy.OutputField()


class GenerateResponse(dspy.Signature):
    """Write one short, human-sounding reply that matches the client's tone.

    Mirror the client's register (formal, casual, distressed, happy).
    For Negative sentiment, lead with empathy. For Positive, mirror warmth.
    For flagged messages with serious content, do NOT use casual filler like
    "great question" — match the seriousness. If flagged, also let them know
    a team member is being looped in.
    Output only the reply text — no quotes, labels, or commentary.
    """

    last_message: str = dspy.InputField()
    triage_reasoning: str = dspy.InputField()
    sentiment: str = dspy.InputField()
    sentiment_score: int = dspy.InputField()
    is_flagged: bool = dspy.InputField()

    reply: str = dspy.OutputField()
