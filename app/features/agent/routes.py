from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from app.features.agent.graph import app as agent_app
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

class AgentInvokeRequest(BaseModel):
    messages: List[Dict[str, Any]] = Field(..., description="List of messages. Each message must have 'role' ('user' or 'assistant') and 'content'.")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context to provide to the agent.")

class AgentInvokeResponse(BaseModel):
    response: str
    tool_calls: List[Dict[str, Any]] = []

@router.post("/invoke", response_model=AgentInvokeResponse)
async def invoke_agent(request: AgentInvokeRequest):
    """
    Invoke the LangGraph agent with a list of messages.
    """
    try:
        lc_messages: List[BaseMessage] = []
        for msg in request.messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            elif role == "system":
                 lc_messages.append(SystemMessage(content=content))
            else:
                 continue
        
        # shove context in if we got it
        if request.context:
             context_str = f"\n\n[Context: {request.context}]"
             if lc_messages and isinstance(lc_messages[-1], HumanMessage):
                 last_msg = lc_messages.pop()
                 # slap context on the end of the last user msg
                 new_content = str(last_msg.content) + context_str
                 lc_messages.append(HumanMessage(content=new_content))
             else:
                 # fallback for weird emptiness
                 lc_messages.append(SystemMessage(content=f"Context: {request.context}"))

        initial_state = {"messages": lc_messages}
        
        # let it rip
        result = await agent_app.ainvoke(initial_state)
        
        last_message = result["messages"][-1]
        
        return AgentInvokeResponse(
            response=str(last_message.content),
            tool_calls=[tc for tc in getattr(last_message, "tool_calls", [])]
        )

    except Exception as e:
        logger.error(f"agent invocation blown up: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during agent execution")
