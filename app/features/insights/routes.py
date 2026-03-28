"""Insights routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends

from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.features.insights.services import HighLevelInsightEngine, MicroInsightEngine
from app.shared.http import ok
from app.shared.schemas import (
    HighLevelInsightRequest,
    InsightRequest,
    MicroInsightResult,
)
from app.shared.utils import sanitize_text, truncate_conversation_history

logger = get_logger(__name__)
router = APIRouter()


def get_micro() -> MicroInsightEngine:
    return MicroInsightEngine()


def get_high_level() -> HighLevelInsightEngine:
    return HighLevelInsightEngine()


def _normalize(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        text = sanitize_text(m.get("content") or m.get("body", ""))
        if text:
            out.append(
                {
                    "timestamp": m.get("timestamp"),
                    "sender": m.get("sender", "unknown"),
                    "body": text,
                    "content": text,
                }
            )
    return out


@router.post("/micro", response_model=MicroInsightResult)
async def generate_micro_insight(
    request: InsightRequest,
    background_tasks: BackgroundTasks,
    engine: MicroInsightEngine = Depends(get_micro),
):
    if not request.client_id:
        raise ValidationError("client_id is required")
    if not request.messages:
        raise ValidationError("messages list cannot be empty")

    cleaned = _normalize(request.messages)
    if not cleaned:
        raise ValidationError("no valid messages after sanitization")

    recent = truncate_conversation_history(cleaned, max_length=200)

    insight = await engine.run_micro_insight_engine(
        client_id=request.client_id,
        client_profile=request.client_profile,
        messages=recent,
        previous_insight=request.previous_insight,
        previous_sentiment=request.previous_sentiment,
    )

    sentiment = "Neutral"
    if "Positive" in insight:
        sentiment = "Positive"
    elif "Negative" in insight:
        sentiment = "Negative"

    background_tasks.add_task(
        logger.info,
        "micro_insight_logged",
        client_id=request.client_id,
        sentiment=sentiment,
    )
    return ok(
        {"insight": insight, "sentiment": sentiment},
        processed_messages=len(recent),
        previous_sentiment=request.previous_sentiment,
        sentiment_changed=sentiment != request.previous_sentiment,
    )


@router.post("/high-level")
async def generate_high_level_insights(
    request: HighLevelInsightRequest,
    engine: HighLevelInsightEngine = Depends(get_high_level),
):
    if not request.firm_name:
        raise ValidationError("firm_name is required")
    if not request.admin_names:
        raise ValidationError("admin_names is required")
    if not request.report_period:
        raise ValidationError("report_period is required")

    report = await engine.generate_high_level_insights(
        firm_name=request.firm_name,
        admin_names=request.admin_names,
        report_period=request.report_period,
        analysis_date=request.analysis_date,
        firm_wide_data=request.firm_wide_data,
        user_performance_data=request.user_performance_data,
    )
    return ok(
        {
            "insights_report": report,
            "report_metadata": {
                "firm_name": request.firm_name,
                "report_period": request.report_period,
                "analysis_date": request.analysis_date,
                "admin_recipients": request.admin_names,
            },
        },
        report_length=len(report),
        data_points_analyzed=len(request.user_performance_data),
    )


@router.post("/summary")
async def generate_summary_insights(
    firm_data: dict[str, Any],
    time_period: str = "monthly",
    engine: HighLevelInsightEngine = Depends(get_high_level),
):
    if not firm_data:
        raise ValidationError("firm_data is required")

    summary = await engine.generate_summary_insights(firm_data=firm_data, time_period=time_period)
    return ok(summary, time_period=time_period, data_points=len(firm_data))


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "insights"}
