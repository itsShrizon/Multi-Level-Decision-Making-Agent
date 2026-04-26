"""Agent API. /stream streams the chat-analysis LangGraph as each node
completes."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.features.agent.chat_graph import chat_graph

router = APIRouter()
logger = get_logger(__name__)


class StreamRequest(BaseModel):
    message: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    client_info: dict[str, Any] = Field(default_factory=dict)


@router.post("/stream")
async def stream_chat_analysis(request: StreamRequest):
    """Stream chat_graph node updates as Server-Sent Events.

    Each event is a `data:`-prefixed JSON line: {"updates": {<node>: <state_diff>}}.
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
                yield _sse({"updates": chunk})
            yield _sse({"event": "done"})
        except Exception as e:  # noqa: BLE001
            logger.error("agent_stream_failed", error=str(e), exc_info=True)
            yield _sse({"event": "error", "message": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()
