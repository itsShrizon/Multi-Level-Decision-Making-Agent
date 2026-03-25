"""Tiny in-memory rate limiter for the legacy middleware path.

Will go away in sprint 5 when slowapi takes over. Keeping it here lets
the existing rate_limiting_middleware keep working in the meantime.
"""

from __future__ import annotations

import time


class RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, list[float]] = {}

    def is_allowed(self, key: str, limit: int = 100, window: int = 60) -> bool:
        now = time.time()
        bucket = [t for t in self._hits.get(key, []) if now - t < window]
        if len(bucket) >= limit:
            self._hits[key] = bucket
            return False
        bucket.append(now)
        self._hits[key] = bucket
        return True


_rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    return _rate_limiter
