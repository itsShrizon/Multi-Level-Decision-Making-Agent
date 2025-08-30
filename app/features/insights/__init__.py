"""
Insights feature module.
"""

from app.features.insights.routes import router
from app.features.insights.services import MicroInsightEngine, HighLevelInsightEngine

__all__ = [
    "router",
    "MicroInsightEngine",
    "HighLevelInsightEngine",
]
