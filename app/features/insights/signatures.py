"""DSPy signatures for the insights features."""

from __future__ import annotations

from typing import Literal

import dspy

Sentiment = Literal["Positive", "Neutral", "Negative"]


class ClassifyAggregateSentiment(dspy.Signature):
    """Classify the aggregate sentiment of provided text."""

    text: str = dspy.InputField()
    sentiment: Sentiment = dspy.OutputField()


class AdjustSentiment(dspy.Signature):
    """Decide whether sentiment should change.

    If the new observation is unclear or marginal, keep the previous value.
    """

    previous: Sentiment = dspy.InputField()
    observed: Sentiment = dspy.InputField()
    sentiment: Sentiment = dspy.OutputField()


class MicroInsight(dspy.Signature):
    """Write one sentence a case manager can read to instantly understand
    what's going on with the client right now.

    Embed the current sentiment naturally in the sentence (do not just
    prefix with 'Sentiment: X'). Focus on tone, preferences, and the most
    actionable cue. Don't repeat the previous insight verbatim.
    """

    client_profile_json: str = dspy.InputField()
    recent_messages_json: str = dspy.InputField()
    previous_insight: str = dspy.InputField()
    current_sentiment: Sentiment = dspy.InputField()

    insight: str = dspy.OutputField()


class HighLevelReport(dspy.Signature):
    """Write a structured business-intelligence report for firm leadership.

    Follow the exact email-body structure in the user prompt. Each finding
    section answers three questions in this order:
      'What I'm seeing', 'Why it matters', 'How to fix it'.
    Close with a 'Summary of Action Items' prioritized list.
    Output the email body text only, no commentary.
    """

    template: str = dspy.InputField(desc="The required email body template, pre-filled with metadata")
    firm_wide_data_json: str = dspy.InputField()
    user_performance_json: str = dspy.InputField()

    report: str = dspy.OutputField()


class SummaryInsights(dspy.Signature):
    """Produce 3-5 dashboard-ready insights as JSON.

    Format: { "insights": [ { "title": "...", "description": "..." }, ... ] }
    Output ONLY valid JSON, no surrounding prose.
    """

    firm_data_json: str = dspy.InputField()
    time_period: str = dspy.InputField()

    json_payload: str = dspy.OutputField()
