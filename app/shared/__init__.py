"""
Shared components and utilities initialization.
"""

from app.shared.schemas import (
    BaseResponse,
    ErrorResponse,
    Message,
    ClientInfo,
    ConversationHistory,
    MessageAnalysisResult,
    InsightRequest,
    MicroInsightResult,
    HighLevelInsightRequest,
    OutboundMessageRequest,
    OutboundMessageResult,
    ChatSummarizationRequest,
    ChatSummarizationResult,
    ConciseRequest,
    ConciseResult,
)

from app.shared.utils import (
    retry_with_backoff,
    format_messages_for_openai,
    truncate_conversation_history,
    sanitize_text,
    extract_client_context,
    validate_openai_response,
    Timer,
)

__all__ = [
    # Schemas
    "BaseResponse",
    "ErrorResponse", 
    "Message",
    "ClientInfo",
    "ConversationHistory",
    "MessageAnalysisResult",
    "InsightRequest",
    "MicroInsightResult",
    "HighLevelInsightRequest",
    "OutboundMessageRequest",
    "OutboundMessageResult",
    "ChatSummarizationRequest",
    "ChatSummarizationResult",
    "ConciseRequest",
    "ConciseResult",
    # Utils
    "retry_with_backoff",
    "format_messages_for_openai",
    "truncate_conversation_history",
    "sanitize_text",
    "extract_client_context",
    "validate_openai_response",
    "Timer",
]
