"""Backward-compat shim. Real DI lives in app.core.llm and app.core.rate_limit."""

from __future__ import annotations

from app.core.rate_limit import limiter as _slowapi_limiter


def get_rate_limiter():
    """Returns the slowapi Limiter. Kept for any caller that still imports
    it; new code should use app.core.rate_limit.limiter directly."""
    return _slowapi_limiter
