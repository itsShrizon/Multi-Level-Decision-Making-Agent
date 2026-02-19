from langchain_core.tools import tool
from typing import Dict, List, Any, Optional
from app.features.chat.services import ChatOrchestrator
from app.features.insights.services import MicroInsightEngine
from app.features.outbound.services import OutboundMessageGenerator

# getting the gang together
chat_orchestrator = ChatOrchestrator()
micro_insight_engine = MicroInsightEngine()
outbound_generator = OutboundMessageGenerator()

@tool
async def analyze_chat_message(
    client_info: Dict[str, Any],
    conversation_history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    figures out what the client is on about. triage, risk, sentiment, the whole shebang.
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
    spits out a one-liner insight. keeps it brief.
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
    writes a fancy message to send out. professional stuff.
    """
    return await outbound_generator.generate_outbound_message(
        information=information,
        messages=messages
    )

# the toolbox
agent_tools = [analyze_chat_message, generate_micro_insight, generate_outbound_message]
