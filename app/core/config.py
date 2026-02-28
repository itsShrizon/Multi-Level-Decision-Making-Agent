"""App settings — env-driven, cached, no surprises."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "staging", "production"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # app
    APP_NAME: str = "mldm-agent"
    API_VERSION: str = "0.2.0"
    ENVIRONMENT: Environment = "development"

    # server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # CORS / hosts
    ALLOWED_ORIGINS: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    ALLOWED_HOSTS: list[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"])

    # LM providers
    OPENAI_API_KEY: str
    GEMINI_API_KEY: str | None = None

    # default models per task tier — keep small set, override per signature if needed
    LM_MAIN: str = "openai/gpt-4o"
    LM_FAST: str = "openai/gpt-4o-mini"
    LM_SUMMARY: str = "openai/gpt-3.5-turbo"
    LM_REPORT: str = "gemini/gemini-1.5-pro"

    LM_MAX_TOKENS: int = 1024
    LM_TEMPERATURE: float = 0.0
    LM_TIMEOUT_S: float = 30.0

    # storage
    DATABASE_URL: str | None = None
    REDIS_URL: str | None = "redis://localhost:6379/0"

    # rate limiting (slowapi reads these)
    RATE_LIMIT: str = "100/minute"

    # logs
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False  # flip on in prod

    # business knobs
    MAX_CONVERSATION_HISTORY: int = 500

    # observability — optional
    SENTRY_DSN: str | None = None

    @property
    def is_prod(self) -> bool:
        return self.ENVIRONMENT == "production"

    @staticmethod
    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
