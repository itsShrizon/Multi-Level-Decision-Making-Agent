"""
FastAPI application entry point for the Multi-Level Chatbot system.

This application provides AI-powered chatbot services for law firms including:
- Message analysis and triage
- Sentiment analysis
- Risk assessment
- Insight generation
- Outbound message generation
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.core.exceptions import (
    ValidationError,
    ServiceError,
    validation_exception_handler,
    service_exception_handler,
    general_exception_handler,
)
from app.features.chat import router as chat_router
from app.features.insights import router as insights_router
from app.features.outbound import router as outbound_router
from app.features.agent import router as agent_router
from app.shared.middleware import (
    request_logging_middleware,
    rate_limiting_middleware,
    error_handling_middleware,
)

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup
    logger.info("Starting Multi-Level Chatbot API")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"API Version: {settings.API_VERSION}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Multi-Level Chatbot API")


def create_application() -> FastAPI:
    """
    Create and configure the FastAPI application.
    """
    # Setup logging first
    setup_logging()
    
    app = FastAPI(
        title="Multi-Level Chatbot API",
        description="AI-powered chatbot system for law firms with advanced message analysis and insights",
        version=settings.API_VERSION,
        docs_url="/api/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/api/redoc" if settings.ENVIRONMENT != "production" else None,
        openapi_url="/api/openapi.json" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Add trusted host middleware for security
    if settings.ENVIRONMENT == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS,
        )
    
    # Add custom middleware
    app.middleware("http")(error_handling_middleware)
    app.middleware("http")(request_logging_middleware)
    app.middleware("http")(rate_limiting_middleware)
    
    # Register exception handlers
    app.add_exception_handler(ValidationError, validation_exception_handler)
    app.add_exception_handler(ServiceError, service_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    
    # Include routers
    app.include_router(
        chat_router,
        prefix="/api/v1/chat",
        tags=["Chat Analysis"]
    )
    app.include_router(
        insights_router,
        prefix="/api/v1/insights",
        tags=["Insights"]
    )
    app.include_router(
        outbound_router,
        prefix="/api/v1/outbound",
        tags=["Outbound Messages"]
    )
    app.include_router(
        agent_router,
        prefix="/api/v1/agent",
        tags=["Agent"]
    )
    
    @app.get("/", tags=["Health"])
    async def root():  # type: ignore[misc]
        """Root endpoint providing API information."""
        return {
            "message": "Multi-Level Chatbot API",
            "version": settings.API_VERSION,
            "environment": settings.ENVIRONMENT,
            "status": "healthy"
        }
    
    @app.get("/health", tags=["Health"])
    async def health_check():  # type: ignore[misc]
        """Health check endpoint for monitoring."""
        return {
            "status": "healthy",
            "timestamp": settings.get_current_timestamp(),
            "version": settings.API_VERSION
        }
    
    return app


# Create the application instance
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
