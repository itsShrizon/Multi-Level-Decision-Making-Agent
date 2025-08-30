"""
Shared Pydantic models for request/response schemas.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field


# Base models
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    
    success: bool = Field(default=True, description="Whether the request was successful")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(default=None, description="Request identifier for tracing")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: Dict[str, Any] = Field(description="Error details")
    success: bool = Field(default=False, description="Always false for errors")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


# Chat and message models
class Message(BaseModel):
    """Individual message in a conversation."""
    
    sender: str = Field(description="Message sender identifier")
    content: str = Field(description="Message content")
    timestamp: Optional[str] = Field(default=None, description="Message timestamp")


class ClientInfo(BaseModel):
    """Client information and profile."""
    
    client_id: str = Field(description="Unique client identifier")
    name: Optional[str] = Field(default=None, description="Client name")
    profile: Dict[str, Any] = Field(default_factory=dict, description="Additional client profile data")


class ConversationHistory(BaseModel):
    """Conversation history with messages."""
    
    messages: List[Message] = Field(description="List of messages in chronological order")
    client_info: ClientInfo = Field(description="Client information")


# Analysis models
class TriageDecision(BaseModel):
    """Triage decision for message processing."""
    
    primary_action: Literal["FLAG", "IGNORE", "RESPOND"] = Field(
        description="The primary action to take based on message content"
    )


class RiskAssessment(BaseModel):
    """Risk assessment for client retention."""
    
    risk_update: Literal["Low", "Medium", "High"] = Field(
        description="The final risk assessment level"
    )
    risk_score: int = Field(
        ge=0, le=100,
        description="Risk score from 0-100, higher means higher risk"
    )


class SentimentAnalysis(BaseModel):
    """Sentiment analysis results."""
    
    sentiment: Literal["Positive", "Neutral", "Negative"] = Field(
        description="The message's sentiment classification"
    )
    sentiment_score: int = Field(
        ge=0, le=100,
        description="Sentiment score from 0-100, higher means more negative"
    )


class EventDetails(BaseModel):
    """Details about an event or appointment."""
    
    date: Optional[str] = Field(default=None, description="The date of the event")
    time: Optional[str] = Field(default=None, description="The time of the event")
    location: Optional[str] = Field(default=None, description="The location of the event")
    event_type: Optional[str] = Field(default=None, description="The type of event")
    additional_info: Optional[str] = Field(default=None, description="Additional event information")


class EventDetection(BaseModel):
    """Event detection results."""
    
    has_event: bool = Field(description="True if the message contains mention of a future event")
    event_details: Optional[EventDetails] = Field(default=None, description="Event details if detected")
    suggested_reminder: Optional[str] = Field(default=None, description="Suggested reminder message")
    internal_note: Optional[str] = Field(default=None, description="Internal note about the event")


# Analysis result models
class MessageAnalysisResult(BaseModel):
    """Complete message analysis result."""
    
    action: str = Field(description="Primary action determined by triage")
    risk_update: str = Field(description="Risk level assessment")
    risk_score: int = Field(description="Numerical risk score")
    sentiment: str = Field(description="Sentiment classification")
    sentiment_score: int = Field(description="Numerical sentiment score")
    response_to_send: Optional[str] = Field(default=None, description="Generated response message")
    event_detection: EventDetection = Field(description="Event detection results")
    full_analysis: Dict[str, Any] = Field(description="Complete analysis breakdown")


# Insight models
class InsightRequest(BaseModel):
    """Request for generating insights."""
    
    client_id: str = Field(description="Client identifier")
    client_profile: Dict[str, Any] = Field(default_factory=dict, description="Client profile data")
    messages: List[Dict[str, Any]] = Field(description="Message history")
    previous_insight: Optional[str] = Field(default=None, description="Previous insight")
    previous_sentiment: Optional[str] = Field(default=None, description="Previous sentiment")


class MicroInsightResult(BaseModel):
    """Micro insight generation result."""
    
    insight: str = Field(description="Generated micro insight")
    sentiment: str = Field(description="Updated sentiment")


class HighLevelInsightRequest(BaseModel):
    """Request for high-level insights generation."""
    
    firm_name: str = Field(description="Law firm name")
    admin_names: List[str] = Field(description="Administrator names")
    report_period: str = Field(description="Reporting period")
    analysis_date: str = Field(description="Analysis date")
    firm_wide_data: Dict[str, Any] = Field(description="Firm-wide analytics data")
    user_performance_data: List[Dict[str, Any]] = Field(description="User performance data")


# Outbound message models
class OutboundMessageRequest(BaseModel):
    """Request for generating outbound messages."""
    
    information: str = Field(description="Context and objectives for the outbound message")
    messages: List[Dict[str, Any]] = Field(description="Conversation history")


class OutboundMessageResult(BaseModel):
    """Outbound message generation result."""
    
    message: str = Field(description="Generated outbound message")


# Chat summarization models
class ChatSummarizationRequest(BaseModel):
    """Request for chat summarization."""
    
    messages: List[Message] = Field(description="Messages to summarize")


class ChatSummarizationResult(BaseModel):
    """Chat summarization result."""
    
    summary: str = Field(description="Generated chat summary")


# Text processing models
class ConciseRequest(BaseModel):
    """Request for making text concise."""
    
    text: str = Field(description="Text to make concise", min_length=1)


class ConciseResult(BaseModel):
    """Result of text concisification."""
    
    concise_text: str = Field(description="Concise version of the input text")
