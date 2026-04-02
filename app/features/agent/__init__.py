"""LangGraph-based agent module.

graph        — high-level tool-calling agent (langchain ChatOpenAI + tools)
chat_graph   — chat-analysis fan-out/fan-in DAG
"""

from app.features.agent.routes import router

__all__ = ["router"]
