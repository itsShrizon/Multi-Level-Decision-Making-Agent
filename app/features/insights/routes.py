"""
Insights API routes for micro and high-level analytics.
"""

from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.core.exceptions import ValidationError, ServiceError
from app.shared.schemas import (
    InsightRequest,
    MicroInsightResult,
    HighLevelInsightRequest,
)
from app.shared.utils import sanitize_text, truncate_conversation_history
from app.features.insights.services import MicroInsightEngine, HighLevelInsightEngine

logger = get_logger(__name__)
router = APIRouter()


# Dependencies
def get_micro_insight_engine() -> MicroInsightEngine:
    """Get micro insight engine instance."""
    return MicroInsightEngine()


def get_high_level_insight_engine() -> HighLevelInsightEngine:
    """Get high-level insight engine instance."""
    return HighLevelInsightEngine()


@router.post("/micro", response_model=MicroInsightResult)
async def generate_micro_insight(
    request: InsightRequest,
    background_tasks: BackgroundTasks,
    engine: MicroInsightEngine = Depends(get_micro_insight_engine)
) -> JSONResponse:
    """
    Generate a micro-level insight for a specific client.
    
    This endpoint analyzes recent client interactions to provide a single-sentence
    insight that helps case managers quickly understand the client's current state,
    including sentiment and actionable cues.
    """
    try:
        # Validate input
        if not request.client_id:
            raise ValidationError("Client ID is required")
        
        if not request.messages:
            raise ValidationError("Messages list cannot be empty")
        
        # Sanitize and prepare messages
        sanitized_messages = []
        for msg in request.messages:
            content = msg.get("content") or msg.get("body", "")
            sanitized_content = sanitize_text(content)
            if sanitized_content:
                sanitized_messages.append({
                    "timestamp": msg.get("timestamp"),
                    "sender": msg.get("sender", "unknown"),
                    "body": sanitized_content,
                    "content": sanitized_content
                })
        
        if not sanitized_messages:
            raise ValidationError("No valid messages found after sanitization")
        
        # Truncate to recent messages for performance
        recent_messages = truncate_conversation_history(sanitized_messages, max_length=200)
        
        logger.info(
            "Starting micro insight generation",
            extra={
                "client_id": request.client_id,
                "message_count": len(recent_messages),
                "has_previous_insight": bool(request.previous_insight),
                "previous_sentiment": request.previous_sentiment
            }
        )
        
        # Generate micro insight
        insight = await engine.run_micro_insight_engine(
            client_id=request.client_id,
            client_profile=request.client_profile,
            messages=recent_messages,
            previous_insight=request.previous_insight,
            previous_sentiment=request.previous_sentiment
        )
        
        # Extract sentiment from insight (it should be embedded)
        sentiment = "Neutral"  # Default
        if "Positive" in insight:
            sentiment = "Positive"
        elif "Negative" in insight:
            sentiment = "Negative"
        
        # Log for analytics
        background_tasks.add_task(
            log_micro_insight,
            client_id=request.client_id,
            insight=insight,
            sentiment=sentiment,
            message_count=len(recent_messages)
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "insight": insight,
                    "sentiment": sentiment
                },
                "metadata": {
                    "processed_messages": len(recent_messages),
                    "previous_sentiment": request.previous_sentiment,
                    "sentiment_changed": sentiment != request.previous_sentiment
                }
            }
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error in micro insight generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        logger.error(f"Service error in micro insight generation: {e}")
        raise HTTPException(status_code=503, detail="Insight generation service temporarily unavailable")
    except Exception as e:
        logger.error(f"Unexpected error in micro insight generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/high-level")
async def generate_high_level_insights(
    request: HighLevelInsightRequest,
    background_tasks: BackgroundTasks,
    engine: HighLevelInsightEngine = Depends(get_high_level_insight_engine)
) -> JSONResponse:
    """
    Generate high-level insights for firm leadership.
    
    This endpoint analyzes firm-wide data and user performance to generate
    comprehensive business intelligence reports with actionable recommendations.
    """
    try:
        # Validate input
        if not request.firm_name:
            raise ValidationError("Firm name is required")
        
        if not request.admin_names:
            raise ValidationError("Administrator names are required")
        
        if not request.report_period:
            raise ValidationError("Report period is required")
        
        logger.info(
            "Starting high-level insight generation",
            extra={
                "firm_name": request.firm_name,
                "admin_count": len(request.admin_names),
                "report_period": request.report_period,
                "user_performance_records": len(request.user_performance_data)
            }
        )
        
        # Generate high-level insights
        insights_report = await engine.generate_high_level_insights(
            firm_name=request.firm_name,
            admin_names=request.admin_names,
            report_period=request.report_period,
            analysis_date=request.analysis_date,
            firm_wide_data=request.firm_wide_data,
            user_performance_data=request.user_performance_data
        )
        
        # Log for analytics and compliance
        background_tasks.add_task(
            log_high_level_insights,
            firm_name=request.firm_name,
            report_period=request.report_period,
            report_length=len(insights_report),
            admin_names=request.admin_names
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {
                    "insights_report": insights_report,
                    "report_metadata": {
                        "firm_name": request.firm_name,
                        "report_period": request.report_period,
                        "analysis_date": request.analysis_date,
                        "admin_recipients": request.admin_names
                    }
                },
                "metadata": {
                    "report_length": len(insights_report),
                    "data_points_analyzed": len(request.user_performance_data)
                }
            }
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error in high-level insight generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        logger.error(f"Service error in high-level insight generation: {e}")
        raise HTTPException(status_code=503, detail="Insight generation service temporarily unavailable")
    except Exception as e:
        logger.error(f"Unexpected error in high-level insight generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/summary")
async def generate_summary_insights(
    firm_data: Dict[str, Any],
    time_period: str = "monthly",
    engine: HighLevelInsightEngine = Depends(get_high_level_insight_engine)
) -> JSONResponse:
    """
    Generate quick summary insights for dashboard display.
    
    This endpoint provides a condensed view of the most important insights
    for quick consumption in dashboards or summary reports.
    """
    try:
        # Validate input
        if not firm_data:
            raise ValidationError("Firm data is required")
        
        logger.info(
            "Starting summary insight generation",
            extra={
                "time_period": time_period,
                "data_keys": list(firm_data.keys())
            }
        )
        
        # Generate summary insights
        summary_insights = await engine.generate_summary_insights(
            firm_data=firm_data,
            time_period=time_period
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": summary_insights,
                "metadata": {
                    "time_period": time_period,
                    "data_points": len(firm_data)
                }
            }
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error in summary insight generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        logger.error(f"Service error in summary insight generation: {e}")
        raise HTTPException(status_code=503, detail="Insight generation service temporarily unavailable")
    except Exception as e:
        logger.error(f"Unexpected error in summary insight generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def health_check():
    """Health check endpoint for the insights service."""
    return {
        "status": "healthy",
        "service": "insights",
        "features": [
            "micro_insights",
            "high_level_insights",
            "summary_insights"
        ]
    }


# Background task functions
async def log_micro_insight(
    client_id: str,
    insight: str,
    sentiment: str,
    message_count: int
) -> None:
    """Log micro insight generation for analytics."""
    try:
        logger.info(
            "Micro insight generated and logged",
            extra={
                "client_id": client_id,
                "sentiment": sentiment,
                "insight_length": len(insight),
                "message_count": message_count,
                "timestamp": "utc_now"
            }
        )
        
        # Here you could send data to analytics service, database, etc.
        # await analytics_service.log_micro_insight(client_id, insight, sentiment)
        
    except Exception as e:
        logger.error(f"Failed to log micro insight: {e}")


async def log_high_level_insights(
    firm_name: str,
    report_period: str,
    report_length: int,
    admin_names: List[str]
) -> None:
    """Log high-level insight generation for compliance and analytics."""
    try:
        logger.info(
            "High-level insights generated and logged",
            extra={
                "firm_name": firm_name,
                "report_period": report_period,
                "report_length": report_length,
                "admin_count": len(admin_names),
                "timestamp": "utc_now"
            }
        )
        
        # Here you could send data to analytics service, compliance logging, etc.
        # await compliance_service.log_insight_report(firm_name, report_period, admin_names)
        
    except Exception as e:
        logger.error(f"Failed to log high-level insights: {e}")
