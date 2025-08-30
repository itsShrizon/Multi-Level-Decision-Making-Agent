"""
Custom middleware for the application.
"""

import time
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.dependencies import get_rate_limiter

settings = get_settings()
logger = get_logger(__name__)
rate_limiter = get_rate_limiter()


async def request_logging_middleware(request: Request, call_next: Callable) -> Response:
    """
    Log incoming requests and responses.
    """
    start_time = time.time()
    
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Log incoming request
    logger.info(
        f"Incoming request: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client_ip": client_ip,
            "user_agent": request.headers.get("user-agent", "unknown"),
        }
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(
        f"Request completed: {request.method} {request.url.path} - {response.status_code}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time": round(process_time, 4),
            "client_ip": client_ip,
        }
    )
    
    # Add processing time to response headers
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


async def rate_limiting_middleware(request: Request, call_next: Callable) -> Response:
    """
    Apply rate limiting to requests.
    """
    if not settings.ENABLE_RATE_LIMITING:
        return await call_next(request)
    
    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/", "/api/docs", "/api/redoc", "/api/openapi.json"]:
        return await call_next(request)
    
    # Get client identifier
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    if not rate_limiter.is_allowed(
        key=f"ip:{client_ip}",
        limit=settings.RATE_LIMIT_REQUESTS,
        window=settings.RATE_LIMIT_WINDOW
    ):
        logger.warning(
            f"Rate limit exceeded for client: {client_ip}",
            extra={
                "client_ip": client_ip,
                "path": request.url.path,
                "method": request.method,
            }
        )
        
        return JSONResponse(
            status_code=429,
            content={
                "error": {
                    "type": "rate_limit_error",
                    "code": "RATE_LIMIT_EXCEEDED",
                    "message": "Rate limit exceeded. Please try again later.",
                    "retry_after": settings.RATE_LIMIT_WINDOW,
                }
            },
            headers={"Retry-After": str(settings.RATE_LIMIT_WINDOW)}
        )
    
    return await call_next(request)


async def error_handling_middleware(request: Request, call_next: Callable) -> Response:
    """
    Global error handling middleware.
    """
    try:
        return await call_next(request)
    except Exception as exc:
        logger.error(
            f"Unhandled exception in middleware: {str(exc)}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "exception_type": type(exc).__name__,
            },
            exc_info=True
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": "internal_error",
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred",
                }
            }
        )
