"""Insights services — micro (per-client) and high-level (firm-wide).

Micro insights run on the cheap tier. The high-level report runs on the
report tier (defaults to Gemini Pro) but the wiring is identical thanks
to dspy.context().
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Literal

import dspy

from app.core.llm import get_lm
from app.core.logging import get_logger
from app.features.insights.signatures import (
    AdjustSentiment,
    ClassifyAggregateSentiment,
    HighLevelReport,
    MicroInsight,
    SummaryInsights,
)

logger = get_logger(__name__)
Sentiment = Literal["Positive", "Neutral", "Negative"]


class MicroInsightEngine:
    """Per-client single-sentence insight + sentiment update."""

    def __init__(self) -> None:
        self._lm = get_lm("fast")
        self._classify = dspy.Predict(ClassifyAggregateSentiment)
        self._adjust = dspy.Predict(AdjustSentiment)
        self._insight = dspy.Predict(MicroInsight)

    async def classify_sentiment(self, text: str) -> Sentiment:
        def _run() -> Sentiment:
            with dspy.context(lm=self._lm):
                return self._classify(text=text).sentiment

        return await asyncio.to_thread(_run)

    async def adjust_sentiment(
        self,
        previous: Sentiment | None,
        messages: list[dict[str, Any]],
    ) -> Sentiment:
        text = "\n".join(str(m.get("body") or m.get("content") or "") for m in messages[-500:])
        if not text.strip():
            return previous or "Neutral"

        observed = await self.classify_sentiment(text)
        if previous is None:
            return observed

        def _run() -> Sentiment:
            with dspy.context(lm=self._lm):
                return self._adjust(previous=previous, observed=observed).sentiment

        try:
            return await asyncio.to_thread(_run)
        except Exception as e:  # noqa: BLE001
            logger.warning("adjust_sentiment_failed", error=str(e))
            return previous

    async def generate_insight(
        self,
        client_info: dict[str, Any],
        messages: list[dict[str, Any]],
        previous_insight: str | None,
        current_sentiment: Sentiment,
    ) -> str:
        # only keep the last 200 messages; trim each to the fields we actually use
        recent = [
            {k: v for k, v in m.items() if k in ("timestamp", "sender", "body", "content")}
            for m in messages[-200:]
        ]

        def _run() -> str:
            with dspy.context(lm=self._lm):
                return self._insight(
                    client_profile_json=json.dumps(client_info or {}, ensure_ascii=False),
                    recent_messages_json=json.dumps(recent, ensure_ascii=False),
                    previous_insight=previous_insight or "",
                    current_sentiment=current_sentiment,
                ).insight.strip()

        insight = await asyncio.to_thread(_run)
        if insight and insight[-1] not in ".!?":
            insight += "."
        if current_sentiment not in insight:
            insight = f"Sentiment: {current_sentiment} — {insight}"
        return insight

    async def run_micro_insight_engine(
        self,
        client_id: str,
        client_profile: dict[str, Any],
        messages: list[dict[str, Any]],
        previous_insight: str | None = None,
        previous_sentiment: Sentiment | None = None,
    ) -> str:
        try:
            sentiment = await self.adjust_sentiment(previous_sentiment, messages)
            insight = await self.generate_insight(
                client_info={"client_id": client_id, **client_profile},
                messages=messages,
                previous_insight=previous_insight,
                current_sentiment=sentiment,
            )
            logger.info(
                "micro_insight_done",
                client_id=client_id,
                sentiment=sentiment,
                changed=sentiment != previous_sentiment,
            )
            return insight
        except Exception as e:  # noqa: BLE001
            logger.error("micro_insight_failed", client_id=client_id, error=str(e))
            return f"Sentiment: {previous_sentiment or 'Neutral'} — Recent client interaction requires review."


class HighLevelInsightEngine:
    """Firm-wide leadership reports. Defaults to Gemini Pro via the report tier."""

    def __init__(self) -> None:
        self._lm = get_lm("report")
        self._report = dspy.Predict(HighLevelReport)
        self._summary = dspy.Predict(SummaryInsights)

    async def generate_high_level_insights(
        self,
        firm_name: str,
        admin_names: list[str],
        report_period: str,
        analysis_date: str,
        firm_wide_data: dict[str, Any],
        user_performance_data: list[dict[str, Any]],
    ) -> str:
        admin_list = ", ".join(admin_names)
        recipients = ", ".join(f"{n} <email@example.com>" for n in admin_names)
        template = _REPORT_TEMPLATE.format(
            firm_name=firm_name,
            report_period=report_period,
            analysis_date=analysis_date,
            admin_list=admin_list,
            recipients=recipients,
        )

        def _run() -> str:
            with dspy.context(lm=self._lm):
                return self._report(
                    template=template,
                    firm_wide_data_json=json.dumps(firm_wide_data),
                    user_performance_json=json.dumps(user_performance_data),
                ).report.strip()

        report = await asyncio.to_thread(_run)
        logger.info(
            "high_level_report_done",
            firm=firm_name,
            period=report_period,
            chars=len(report),
        )
        return report

    async def generate_summary_insights(
        self,
        firm_data: dict[str, Any],
        time_period: str = "monthly",
    ) -> dict[str, Any]:
        def _run() -> str:
            with dspy.context(lm=self._lm):
                return self._summary(
                    firm_data_json=json.dumps(firm_data),
                    time_period=time_period,
                ).json_payload

        raw = await asyncio.to_thread(_run)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise
            return json.loads(match.group())


_REPORT_TEMPLATE = """\
This monthly insight report for {firm_name} is ready for your review. If approved, please forward to the designated recipients:
{recipients}

**Arviso High-Level Insight: {firm_name}**
**Report for Period:** {report_period}
**Date of Analysis:** {analysis_date}
**Prepared For:** {admin_list}, {firm_name}

---

**Executive Summary:**
[2-4 sentence summary of all key findings and their implications]

---

**1 - [Compelling Title for Finding #1]**

**What I'm seeing:** [data pattern]
**Why it matters:** [business impact]
**How to fix it:** [actionable recommendation]

---

(More numbered sections for each significant insight)

---

**Summary of Action Items:**

**Priority 1:** [most urgent recommendation]
**Priority 2:** [second-most urgent recommendation]
"""
