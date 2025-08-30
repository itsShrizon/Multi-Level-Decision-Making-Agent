"""
Custom exceptions and exception handlers for the application.
"""

from typing import Any, Dict, Optional
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

logger = get_logger(__name__)


class ApplicationError(Exception):
    """Base application exception."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(ApplicationError):
    """Raised when input validation fails."""
    pass


class ServiceError(ApplicationError):
    """Raised when external service calls fail."""
    pass


class OpenAIError(ServiceError):
    """Raised when OpenAI API calls fail."""
    pass


class RateLimitError(ApplicationError):
    """Raised when rate limit is exceeded."""
    pass


class ConfigurationError(ApplicationError):
    """Raised when configuration is invalid."""
    pass


async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Handle validation errors."""
    logger.warning(
        f"Validation error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "path": request.url.path,
            "method": request.method,
            "details": exc.details,
        }
    )
    
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "type": "validation_error",
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.details,
            }
        }
    )


async def service_exception_handler(request: Request, exc: ServiceError) -> JSONResponse:
    """Handle service errors."""
    logger.error(
        f"Service error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "path": request.url.path,
            "method": request.method,
            "details": exc.details,
        }
    )
    
    return JSONResponse(
        status_code=503,
        content={
            "error": {
                "type": "service_error",
                "code": exc.error_code,
                "message": "Service temporarily unavailable",
                "details": {} if isinstance(exc, OpenAIError) else exc.details,
            }
        }
    )


async def rate_limit_exception_handler(request: Request, exc: RateLimitError) -> JSONResponse:
    """Handle rate limit errors."""
    logger.warning(
        f"Rate limit exceeded: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "path": request.url.path,
            "method": request.method,
            "client_ip": request.client.host if request.client else "unknown",
        }
    )
    
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "type": "rate_limit_error",
                "code": exc.error_code,
                "message": "Rate limit exceeded. Please try again later.",
                "retry_after": exc.details.get("retry_after", 60),
            }
        }
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle HTTP exceptions."""
    logger.warning(
        f"HTTP error {exc.status_code}: {exc.detail}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status_code": exc.status_code,
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "code": f"HTTP_{exc.status_code}",
                "message": exc.detail,
            }
        }
    )


async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle FastAPI request validation errors."""
    logger.warning(
        f"Request validation error: {exc.errors()}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "errors": exc.errors(),
        }
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "validation_error",
                "code": "REQUEST_VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": exc.errors(),
            }
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected errors."""
    logger.error(
        f"Unexpected error: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "exception_type": type(exc).__name__,
        },
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
        }
    )
