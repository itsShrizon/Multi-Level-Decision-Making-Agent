from langchain_core.tools import tool
from typing import Dict, List, Any, Optional
from app.features.chat.services import ChatOrchestrator
from app.features.insights.services import MicroInsightEngine
from app.features.outbound.services import OutboundMessageGenerator


chat_orchestrator = ChatOrchestrator()
micro_insight_engine = MicroInsightEngine()
outbound_generator = OutboundMessageGenerator()

@tool
async def analyze_chat_message(
    client_info: Dict[str, Any],
    conversation_history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Analyze a client's message to determine triage actions, risk, sentiment, and event detection.
    
    Args:
        client_info: Dictionary containing client profile information (must include 'client_id').
        conversation_history: List of message dictionaries. Each message must have 'content' and 'role' or 'sender'.
    """
    return await chat_orchestrator.analyze_message(client_info, conversation_history)

@tool
async def generate_micro_insight(
    client_id: str,
    client_profile: Dict[str, Any],
    messages: List[Dict[str, Any]],
    previous_insight: Optional[str] = None,
    previous_sentiment: Optional[str] = None
) -> str:
    """
    Generate a concise, single-sentence insight about the client's current state based on recent messages.
    
    Args:
        client_id: The client's unique identifier.
        client_profile: Dictionary of client details.
        messages: List of recent conversation messages.
        previous_insight: (Optional) The last generated insight for this client.
        previous_sentiment: (Optional) The last recorded sentiment (Positive, Negative, Neutral).
    """
    return await micro_insight_engine.run_micro_insight_engine(
        client_id=client_id,
        client_profile=client_profile,
        messages=messages,
        previous_insight=previous_insight,
        previous_sentiment=previous_sentiment
    )

@tool
async def generate_outbound_message(
    information: str,
    messages: List[Dict[str, Any]]
) -> str:
    """
    Generate a professional outbound message (e.g., weekly check-in) based on context.
    
    Args:
        information: The context or objective for the message (e.g., "Weekly check-in about physical therapy").
        messages: The conversation history to inform the tone and content.
    """
    return await outbound_generator.generate_outbound_message(
        information=information,
        messages=messages
    )

# List of tools to be used by the agent
agent_tools = [analyze_chat_message, generate_micro_insight, generate_outbound_message]
