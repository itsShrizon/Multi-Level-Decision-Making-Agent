"""Exceptions + FastAPI handlers."""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger

logger = get_logger(__name__)


class ApplicationError(Exception):
    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(ApplicationError):
    pass


class ServiceError(ApplicationError):
    pass


def _err_body(kind: str, exc: ApplicationError, **extra: Any) -> dict[str, Any]:
    return {
        "error": {
            "type": kind,
            "code": exc.error_code,
            "message": exc.message,
            **extra,
        }
    }


async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    logger.warning("validation_error", path=request.url.path, code=exc.error_code, message=exc.message)
    return JSONResponse(status_code=400, content=_err_body("validation_error", exc, details=exc.details))


async def service_exception_handler(request: Request, exc: ServiceError) -> JSONResponse:
    logger.error("service_error", path=request.url.path, code=exc.error_code, message=exc.message)
    body = _err_body("service_error", exc, details=exc.details)
    body["error"]["message"] = "Service temporarily unavailable"
    return JSONResponse(status_code=503, content=body)


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    logger.warning("http_error", path=request.url.path, status=exc.status_code, detail=str(exc.detail))
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "http_error",
                "code": f"HTTP_{exc.status_code}",
                "message": str(exc.detail),
            }
        },
    )


async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.warning("request_validation_error", path=request.url.path, errors=exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "type": "validation_error",
                "code": "REQUEST_VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": exc.errors(),
            }
        },
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unexpected_error", path=request.url.path, exc_type=type(exc).__name__, exc_info=True)
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
