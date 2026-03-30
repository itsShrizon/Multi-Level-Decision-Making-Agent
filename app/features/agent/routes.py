"""Agent API. /invoke runs the tool-calling agent; /stream streams the
chat-analysis LangGraph as each node completes."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.features.agent.chat_graph import chat_graph
from app.features.agent.graph import app as agent_app

router = APIRouter()
logger = get_logger(__name__)


class AgentInvokeRequest(BaseModel):
    messages: list[dict[str, Any]] = Field(
        ..., description="Each message has 'role' (user|assistant|system) and 'content'."
    )
    context: dict[str, Any] | None = None


class AgentInvokeResponse(BaseModel):
    response: str
    tool_calls: list[dict[str, Any]] = []


class StreamRequest(BaseModel):
    message: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    client_info: dict[str, Any] = Field(default_factory=dict)


def _to_lc_messages(raw: list[dict[str, Any]], context: dict[str, Any] | None) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for msg in raw:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
    if context:
        addendum = f"\n\n[Context: {context}]"
        if out and isinstance(out[-1], HumanMessage):
            last = out.pop()
            out.append(HumanMessage(content=str(last.content) + addendum))
        else:
            out.append(SystemMessage(content=f"Context: {context}"))
    return out


@router.post("/invoke", response_model=AgentInvokeResponse)
async def invoke_agent(request: AgentInvokeRequest):
    try:
        messages = _to_lc_messages(request.messages, request.context)
        result = await agent_app.ainvoke({"messages": messages})
        last = result["messages"][-1]
        return AgentInvokeResponse(
            response=str(last.content),
            tool_calls=list(getattr(last, "tool_calls", []) or []),
        )
    except Exception as e:  # noqa: BLE001
        logger.error("agent_invoke_failed", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="agent execution failed") from e


@router.post("/stream")
async def stream_chat_analysis(request: StreamRequest):
    """Stream chat_graph node updates as Server-Sent Events.

    Each event is a `data:`-prefixed JSON line: {"node": "...", "update": {...}}.
    A final {"event": "done"} marks the end of the stream.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    payload = {
        "message": request.message,
        "history": request.history,
        "client_info": request.client_info,
    }

    async def gen() -> AsyncIterator[bytes]:
        try:
            async for chunk in chat_graph.astream(payload, stream_mode="updates"):
                # chunk: { "<node_name>": <state_diff_dict> }
                yield _sse({"updates": chunk})
            yield _sse({"event": "done"})
        except Exception as e:  # noqa: BLE001
            logger.error("agent_stream_failed", error=str(e), exc_info=True)
            yield _sse({"event": "error", "message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")
