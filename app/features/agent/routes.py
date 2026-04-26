"""Agent API.

  POST /stream  — start (or resume) a chat-analysis run; SSE stream of node updates.
  POST /resume  — same as /stream but for a known thread_id paused at await_human.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.features.agent.chat_graph import chat_graph

router = APIRouter()
logger = get_logger(__name__)


class StreamRequest(BaseModel):
    message: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    client_info: dict[str, Any] = Field(default_factory=dict)
    # Optional. If absent, derived from client_info.client_id, else a UUID.
    thread_id: str | None = None


class ResumeRequest(BaseModel):
    thread_id: str
    # Reviewer's decision payload — see await_human_node docstring for shape.
    decision: dict[str, Any]


def _thread_id_for(req: StreamRequest) -> str:
    if req.thread_id:
        return req.thread_id
    cid = (req.client_info or {}).get("client_id")
    return str(cid) if cid else f"anon-{uuid.uuid4()}"


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, default=str)}\n\n".encode()


async def _stream_graph(input_value: Any, thread_id: str) -> AsyncIterator[bytes]:
    config = {"configurable": {"thread_id": thread_id}}
    yield _sse({"event": "start", "thread_id": thread_id})
    try:
        async for chunk in chat_graph.astream(input_value, config=config, stream_mode="updates"):
            yield _sse({"updates": chunk, "thread_id": thread_id})
        # report whether the graph finished or paused at an interrupt
        snapshot = await chat_graph.aget_state(config)
        paused = bool(snapshot.next)
        yield _sse({"event": "paused" if paused else "done", "thread_id": thread_id, "next": list(snapshot.next)})
    except Exception as e:  # noqa: BLE001
        logger.error("agent_stream_failed", error=str(e), thread_id=thread_id, exc_info=True)
        yield _sse({"event": "error", "message": str(e), "thread_id": thread_id})


@router.post("/stream")
async def stream_chat_analysis(request: StreamRequest):
    """Start a chat-analysis run, stream node updates as SSE."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="message is required")

    thread_id = _thread_id_for(request)
    payload = {
        "message": request.message,
        "history": request.history,
        "client_info": request.client_info,
    }
    return StreamingResponse(_stream_graph(payload, thread_id), media_type="text/event-stream")


@router.post("/resume")
async def resume_chat_analysis(request: ResumeRequest):
    """Resume a thread paused at await_human with a reviewer's decision."""
    return StreamingResponse(
        _stream_graph(Command(resume=request.decision), request.thread_id),
        media_type="text/event-stream",
    )
