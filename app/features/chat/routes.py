"""Chat analysis routes."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends

from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.features.chat.services import ChatOrchestrator
from app.features.chat.summarization import ChatSummarizer
from app.features.chat.text_processing import TextProcessor
from app.shared.http import ok
from app.shared.schemas import (
    ChatSummarizationRequest,
    ChatSummarizationResult,
    ConciseRequest,
    ConciseResult,
    ConversationHistory,
    MessageAnalysisResult,
)
from app.shared.utils import (
    extract_client_context,
    sanitize_text,
    truncate_conversation_history,
)

logger = get_logger(__name__)
router = APIRouter()


def get_orchestrator() -> ChatOrchestrator:
    return ChatOrchestrator()


def get_summarizer() -> ChatSummarizer:
    return ChatSummarizer()


def get_text_processor() -> TextProcessor:
    return TextProcessor()


def _clean_messages(messages) -> list[dict]:
    out = []
    for m in messages:
        content = sanitize_text(m.content)
        if content:
            out.append({"sender": m.sender, "content": content, "timestamp": m.timestamp})
    return out


@router.post("/analyze", response_model=MessageAnalysisResult)
async def analyze_message(
    conversation: ConversationHistory,
    background_tasks: BackgroundTasks,
    orchestrator: ChatOrchestrator = Depends(get_orchestrator),
):
    if not conversation.messages:
        raise ValidationError("messages list cannot be empty")

    cleaned = _clean_messages(conversation.messages)
    if not cleaned:
        raise ValidationError("no valid messages after sanitization")

    truncated = truncate_conversation_history(cleaned)
    ctx = extract_client_context(conversation.client_info.model_dump(), truncated)

    logger.info(
        "analyze_message_start",
        client_id=conversation.client_info.client_id,
        n=len(truncated),
    )

    result = await orchestrator.analyze_message(
        client_info=conversation.client_info.model_dump(),
        conversation_history=truncated,
    )

    background_tasks.add_task(
        _log_analysis,
        client_id=conversation.client_info.client_id,
        analysis=result,
    )
    return ok(result, processed_messages=len(truncated), client_context=ctx)


@router.post("/summarize", response_model=ChatSummarizationResult)
async def summarize_conversation(
    request: ChatSummarizationRequest,
    summarizer: ChatSummarizer = Depends(get_summarizer),
):
    if not request.messages:
        raise ValidationError("messages list cannot be empty")

    messages = [
        {"sender": m.sender, "text": sanitize_text(m.content), "timestamp": m.timestamp}
        for m in request.messages
        if sanitize_text(m.content)
    ]
    if not messages:
        raise ValidationError("no valid messages after sanitization")

    summary = await summarizer.summarize_chat({"messages": messages})
    return ok({"summary": summary}, processed_messages=len(messages))


@router.post("/make-concise", response_model=ConciseResult)
async def make_text_concise(
    request: ConciseRequest,
    processor: TextProcessor = Depends(get_text_processor),
):
    text = sanitize_text(request.text)
    if not text:
        raise ValidationError("text cannot be empty")

    concise = await processor.make_concise(text)
    return ok(
        {"concise_text": concise},
        original_length=len(text),
        concise_length=len(concise),
    )


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "chat_analysis"}


async def _log_analysis(client_id: str, analysis: dict) -> None:
    logger.info(
        "analysis_logged",
        client_id=client_id,
        actions=analysis.get("actions"),
        risk=analysis.get("risk_update"),
        sentiment=analysis.get("sentiment"),
        has_event=analysis.get("event_detection", {}).get("has_event", False),
    )
