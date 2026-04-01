"""slowapi limiter shared by main.py and any route that needs custom limits."""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import get_settings

_settings = get_settings()

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[_settings.RATE_LIMIT],
    storage_uri=_settings.REDIS_URL or "memory://",
)
