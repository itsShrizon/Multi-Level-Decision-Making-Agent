"""
Chat analysis API routes.
"""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from app.core.logging import get_logger
from app.core.exceptions import ValidationError, ServiceError
from app.shared.schemas import (
    ConversationHistory,
    MessageAnalysisResult,
    ChatSummarizationRequest,
    ChatSummarizationResult,
    ConciseRequest,
    ConciseResult,
)
from app.shared.utils import sanitize_text, truncate_conversation_history, extract_client_context
from app.features.chat.services import ChatOrchestrator
from app.features.chat.summarization import ChatSummarizer
from app.features.chat.text_processing import TextProcessor

logger = get_logger(__name__)
router = APIRouter()


# Dependency to get chat orchestrator
def get_chat_orchestrator() -> ChatOrchestrator:
    """Get chat orchestrator instance."""
    return ChatOrchestrator()


def get_chat_summarizer() -> ChatSummarizer:
    """Get chat summarizer instance."""
    return ChatSummarizer()


def get_text_processor() -> TextProcessor:
    """Get text processor instance."""
    return TextProcessor()


@router.post("/analyze", response_model=MessageAnalysisResult)
async def analyze_message(
    conversation: ConversationHistory,
    background_tasks: BackgroundTasks,
    orchestrator: ChatOrchestrator = Depends(get_chat_orchestrator)
) -> JSONResponse:
    """
    Analyze a client message using AI agents for triage, risk, sentiment, and event detection.
    
    This endpoint performs comprehensive analysis including:
    - Message triage (FLAG, IGNORE, RESPOND)
    - Risk assessment for client retention
    - Sentiment analysis
    - Event and appointment detection
    - Contextual response generation
    """
    try:
        # Validate input
        if not conversation.messages:
            raise ValidationError("Messages list cannot be empty")
        
        # Sanitize and truncate conversation history
        sanitized_messages = []
        for msg in conversation.messages:
            sanitized_content = sanitize_text(msg.content)
            if sanitized_content:  # Only include non-empty messages
                sanitized_messages.append({
                    "sender": msg.sender,
                    "content": sanitized_content,
                    "timestamp": msg.timestamp
                })
        
        if not sanitized_messages:
            raise ValidationError("No valid messages found after sanitization")
        
        # Truncate conversation history for performance
        truncated_messages = truncate_conversation_history(sanitized_messages)
        
        # Extract client context
        client_context = extract_client_context(
            conversation.client_info.model_dump(),
            truncated_messages
        )
        
        logger.info(
            "Starting message analysis",
            extra={
                "client_id": conversation.client_info.client_id,
                "message_count": len(truncated_messages),
                "last_message_preview": truncated_messages[-1]["content"][:100] + "..." 
                if len(truncated_messages[-1]["content"]) > 100 else truncated_messages[-1]["content"]
            }
        )
        
        # Perform analysis
        analysis_result = await orchestrator.analyze_message(
            client_info=conversation.client_info.model_dump(),
            conversation_history=truncated_messages
        )
        
        # Log analysis for background processing (analytics, etc.)
        background_tasks.add_task(
            log_analysis_result,
            client_id=conversation.client_info.client_id,
            analysis=analysis_result,
            context=client_context
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": analysis_result,
                "metadata": {
                    "processed_messages": len(truncated_messages),
                    "client_context": client_context
                }
            }
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error in message analysis: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        logger.error(f"Service error in message analysis: {e}")
        raise HTTPException(status_code=503, detail="Analysis service temporarily unavailable")
    except Exception as e:
        logger.error(f"Unexpected error in message analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/summarize", response_model=ChatSummarizationResult)
async def summarize_conversation(
    request: ChatSummarizationRequest,
    summarizer: ChatSummarizer = Depends(get_chat_summarizer)
) -> JSONResponse:
    """
    Generate a concise summary of a chat conversation.
    
    This endpoint analyzes the conversation flow and generates a structured summary
    including key topics, client sentiment, and conversation progression.
    """
    try:
        # Validate input
        if not request.messages:
            raise ValidationError("Messages list cannot be empty")
        
        # Convert messages to dict format
        messages_dict = [
            {
                "sender": msg.sender,
                "text": sanitize_text(msg.content),
                "timestamp": msg.timestamp
            }
            for msg in request.messages
            if sanitize_text(msg.content)  # Only include non-empty messages
        ]
        
        if not messages_dict:
            raise ValidationError("No valid messages found after sanitization")
        
        logger.info(
            "Starting conversation summarization",
            extra={"message_count": len(messages_dict)}
        )
        
        # Generate summary
        summary = await summarizer.summarize_chat({"messages": messages_dict})
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {"summary": summary},
                "metadata": {
                    "processed_messages": len(messages_dict)
                }
            }
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error in summarization: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        logger.error(f"Service error in summarization: {e}")
        raise HTTPException(status_code=503, detail="Summarization service temporarily unavailable")
    except Exception as e:
        logger.error(f"Unexpected error in summarization: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/make-concise", response_model=ConciseResult)
async def make_text_concise(
    request: ConciseRequest,
    processor: TextProcessor = Depends(get_text_processor)
) -> JSONResponse:
    """
    Make a given text more concise while preserving meaning.
    
    This endpoint uses AI to reduce text length while maintaining key information,
    typically reducing to 3-4 words maximum.
    """
    try:
        # Validate and sanitize input
        text = sanitize_text(request.text)
        if not text:
            raise ValidationError("Text cannot be empty")
        
        logger.info(
            "Starting text concisification",
            extra={
                "original_length": len(text),
                "text_preview": text[:100] + "..." if len(text) > 100 else text
            }
        )
        
        # Process text
        concise_text = await processor.make_concise(text)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {"concise_text": concise_text},
                "metadata": {
                    "original_length": len(text),
                    "concise_length": len(concise_text),
                    "reduction_ratio": round((len(text) - len(concise_text)) / len(text) * 100, 2)
                }
            }
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error in text processing: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        logger.error(f"Service error in text processing: {e}")
        raise HTTPException(status_code=503, detail="Text processing service temporarily unavailable")
    except Exception as e:
        logger.error(f"Unexpected error in text processing: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def health_check():
    """Health check endpoint for the chat analysis service."""
    return {
        "status": "healthy",
        "service": "chat_analysis",
        "features": [
            "message_analysis",
            "conversation_summarization", 
            "text_processing"
        ]
    }


# Background task functions
async def log_analysis_result(
    client_id: str,
    analysis: Dict[str, Any],
    context: Dict[str, Any]
) -> None:
    """
    Log analysis results for analytics and monitoring.
    
    This function runs in the background to avoid impacting response time.
    """
    try:
        logger.info(
            "Analysis result logged",
            extra={
                "client_id": client_id,
                "action": analysis.get("action"),
                "risk_level": analysis.get("risk_update"),
                "sentiment": analysis.get("sentiment"),
                "has_response": analysis.get("response_to_send") is not None,
                "has_event": analysis.get("event_detection", {}).get("has_event", False),
                "context": context
            }
        )
        
        # Here you could send data to analytics service, database, etc.
        # await analytics_service.log_analysis(client_id, analysis, context)
        
    except Exception as e:
        logger.error(f"Failed to log analysis result: {e}")
