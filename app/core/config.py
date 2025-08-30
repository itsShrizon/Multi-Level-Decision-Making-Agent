"""
Application configuration management using Pydantic settings.
"""

from datetime import datetime
from typing import List, Optional
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    """
    
    # Basic app configuration
    APP_NAME: str = Field(default="Multi-Level Chatbot API", description="Application name")
    API_VERSION: str = Field(default="1.0.0", description="API version")
    ENVIRONMENT: str = Field(default="development", description="Environment (development, staging, production)")
    DEBUG: bool = Field(default=True, description="Debug mode")
    
    # Server configuration
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    
    # Security
    SECRET_KEY: str = Field(default="your-secret-key-change-in-production", description="Secret key for security")
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080", "http://127.0.0.1:3000"],
        description="Allowed CORS origins"
    )
    ALLOWED_HOSTS: List[str] = Field(
        default=["localhost", "127.0.0.1", "0.0.0.0"],
        description="Allowed hosts for TrustedHostMiddleware"
    )
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = Field(description="OpenAI API key for AI services")
    OPENAI_MODEL_GPT4: str = Field(default="gpt-4o", description="GPT-4 model version")
    OPENAI_MODEL_GPT35: str = Field(default="gpt-3.5-turbo", description="GPT-3.5 model version")
    OPENAI_MODEL_MINI: str = Field(default="gpt-4o-mini", description="Mini model version")
    OPENAI_MAX_TOKENS: int = Field(default=1024, description="Maximum tokens for OpenAI requests")
    OPENAI_TEMPERATURE: float = Field(default=0.0, description="Temperature for OpenAI requests")
    
    # Google Gemini Configuration
    GEMINI_API_KEY: str = Field(description="Google Gemini API key for high-level insights")
    GEMINI_MODEL: str = Field(default="gemini-2.5-pro", description="Gemini model version")
    GEMINI_TEMPERATURE: float = Field(default=0.3, description="Temperature for Gemini requests")
    GEMINI_MAX_TOKENS: int = Field(default=4000, description="Maximum tokens for Gemini requests")
    
    # Database Configuration (for future use)
    DATABASE_URL: Optional[str] = Field(default=None, description="Database connection URL")
    DATABASE_ECHO: bool = Field(default=False, description="Echo SQL queries")
    
    # Redis Configuration (for caching and rate limiting)
    REDIS_URL: Optional[str] = Field(default="redis://localhost:6379", description="Redis connection URL")
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Logging format"
    )
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="Rate limit requests per minute")
    RATE_LIMIT_WINDOW: int = Field(default=60, description="Rate limit window in seconds")
    
    # Feature Flags
    ENABLE_ANALYTICS: bool = Field(default=True, description="Enable analytics features")
    ENABLE_CACHING: bool = Field(default=True, description="Enable response caching")
    ENABLE_RATE_LIMITING: bool = Field(default=True, description="Enable rate limiting")
    
    # Business Logic Configuration
    MAX_CONVERSATION_HISTORY: int = Field(default=500, description="Maximum conversation history length")
    INSIGHT_CACHE_TTL: int = Field(default=3600, description="Insight cache TTL in seconds")
    
    # Email Configuration (for high-level insights)
    SMTP_HOST: Optional[str] = Field(default=None, description="SMTP host for email sending")
    SMTP_PORT: int = Field(default=587, description="SMTP port")
    SMTP_USER: Optional[str] = Field(default=None, description="SMTP username")
    SMTP_PASSWORD: Optional[str] = Field(default=None, description="SMTP password")
    FROM_EMAIL: Optional[str] = Field(default=None, description="From email address")
    
    # Monitoring
    SENTRY_DSN: Optional[str] = Field(default=None, description="Sentry DSN for error monitoring")
    
    # Celery Configuration (for background tasks)
    CELERY_BROKER_URL: Optional[str] = Field(default=None, description="Celery broker URL")
    CELERY_RESULT_BACKEND: Optional[str] = Field(default=None, description="Celery result backend URL")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    def get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.utcnow().isoformat()
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT.lower() == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.ENVIRONMENT.lower() == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    """
    return Settings()


# Export commonly used settings
settings = get_settings()
