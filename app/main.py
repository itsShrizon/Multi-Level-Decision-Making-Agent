"""FastAPI app factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.core.config import get_settings
from app.core.exceptions import (
    ServiceError,
    ValidationError,
    general_exception_handler,
    service_exception_handler,
    validation_exception_handler,
)
from app.core.llm import configure_default_lm
from app.core.logging import get_logger, setup_logging
from app.core.rate_limit import limiter
from app.features.agent import router as agent_router
from app.features.chat import router as chat_router
from app.features.insights import router as insights_router
from app.features.outbound import router as outbound_router
from app.shared.middleware import error_handling_middleware, request_logging_middleware

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    log = get_logger(__name__)
    configure_default_lm()
    log.info("startup", env=settings.ENVIRONMENT, version=settings.API_VERSION)
    yield
    log.info("shutdown")


def create_application() -> FastAPI:
    setup_logging()
    show_docs = not settings.is_prod

    app = FastAPI(
        title="Multi-Level Decision-Making Agent",
        description="DSPy + LangGraph orchestrator for law-firm client comms",
        version=settings.API_VERSION,
        docs_url="/api/docs" if show_docs else None,
        redoc_url="/api/redoc" if show_docs else None,
        openapi_url="/api/openapi.json" if show_docs else None,
        lifespan=lifespan,
    )

    # CORS + trusted host
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    if settings.is_prod:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)

    # rate limiting via slowapi
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # custom middleware
    app.middleware("http")(error_handling_middleware)
    app.middleware("http")(request_logging_middleware)

    # global exception handlers
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(ServiceError, service_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # routers
    app.include_router(chat_router, prefix="/api/v1/chat", tags=["Chat Analysis"])
    app.include_router(insights_router, prefix="/api/v1/insights", tags=["Insights"])
    app.include_router(outbound_router, prefix="/api/v1/outbound", tags=["Outbound Messages"])
    app.include_router(agent_router, prefix="/api/v1/agent", tags=["Agent"])

    @app.get("/", tags=["Health"])
    async def root():
        return {
            "service": "mldm-agent",
            "version": settings.API_VERSION,
            "environment": settings.ENVIRONMENT,
            "status": "healthy",
        }

    @app.get("/health", tags=["Health"])
    async def health_check():
        return {
            "status": "healthy",
            "timestamp": settings.utc_now_iso(),
            "version": settings.API_VERSION,
        }

    return app


app = create_application()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENVIRONMENT == "development",
        log_level="info",
    )
