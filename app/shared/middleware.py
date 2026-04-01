"""Custom middleware: request logging + last-resort error catcher.

Rate limiting moved to slowapi (see app.core.rate_limit + main.py).
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)


async def request_logging_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    started = time.time()
    client_ip = request.client.host if request.client else "unknown"

    logger.info(
        "http_request_in",
        method=request.method,
        path=request.url.path,
        client_ip=client_ip,
    )

    response = await call_next(request)

    duration = time.time() - started
    logger.info(
        "http_request_out",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration * 1000, 2),
        client_ip=client_ip,
    )
    response.headers["X-Process-Time"] = f"{duration:.4f}"
    return response


async def error_handling_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    try:
        return await call_next(request)
    except Exception as exc:
        logger.error(
            "unhandled_middleware_exception",
            method=request.method,
            path=request.url.path,
            exc_type=type(exc).__name__,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_error",
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred",
                }
            },
        )
