"""Shared schemas + small request helpers."""

from app.shared.schemas import (
    BaseResponse,
    ChatSummarizationRequest,
    ChatSummarizationResult,
    ClientInfo,
    ConciseRequest,
    ConciseResult,
    ConversationHistory,
    ErrorResponse,
    HighLevelInsightRequest,
    InsightRequest,
    Message,
    MessageAnalysisResult,
    MicroInsightResult,
    OutboundMessageRequest,
    OutboundMessageResult,
)
from app.shared.utils import (
    extract_client_context,
    sanitize_text,
    truncate_conversation_history,
)

__all__ = [
    "BaseResponse",
    "ChatSummarizationRequest",
    "ChatSummarizationResult",
    "ClientInfo",
    "ConciseRequest",
    "ConciseResult",
    "ConversationHistory",
    "ErrorResponse",
    "HighLevelInsightRequest",
    "InsightRequest",
    "Message",
    "MessageAnalysisResult",
    "MicroInsightResult",
    "OutboundMessageRequest",
    "OutboundMessageResult",
    "extract_client_context",
    "sanitize_text",
    "truncate_conversation_history",
]
