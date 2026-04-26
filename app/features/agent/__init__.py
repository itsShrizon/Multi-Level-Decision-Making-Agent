"""LangGraph-based agent module.

chat_graph   — chat-analysis fan-out/fan-in DAG (the only graph for now)
routes       — exposes /stream
"""

from app.features.agent.routes import router

__all__ = ["router"]
