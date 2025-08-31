"""
Core dependencies for dependency injection.
"""

from functools import lru_cache
import time
from openai import AsyncOpenAI
import google.generativeai as genai
from app.core.config import get_settings
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)


@lru_cache()
def get_openai_client() -> AsyncOpenAI:
    """
    Get cached OpenAI client instance.
    
    Returns:
        Configured AsyncOpenAI client
        
    Raises:
        ConfigurationError: If OpenAI API key is not configured
    """
    if not settings.OPENAI_API_KEY:
        raise ConfigurationError(
            "OpenAI API key not configured",
            error_code="OPENAI_API_KEY_MISSING"
        )
    
    return AsyncOpenAI(
        api_key=settings.OPENAI_API_KEY,
        timeout=30.0,
        max_retries=3,
    )


@lru_cache()
def get_gemini_client():
    """
    Get cached Gemini client instance.
    
    Returns:
        Configured Gemini GenerativeModel
        
    Raises:
        ConfigurationError: If Gemini API key is not configured
    """
    if not settings.GEMINI_API_KEY:
        raise ConfigurationError(
            "Gemini API key not configured",
            error_code="GEMINI_API_KEY_MISSING"
        )
    
    # Configure the Gemini client
    genai.configure(api_key=settings.GEMINI_API_KEY)
    
    # Create and return the model
    model = genai.GenerativeModel(
        model_name=settings.GEMINI_MODEL,
        generation_config={
            "temperature": settings.GEMINI_TEMPERATURE,
            "max_output_tokens": settings.GEMINI_MAX_TOKENS,
        }
    )
    
    return model


class RateLimiter:
    """
    Simple in-memory rate limiter.
    In production, this should use Redis for distributed rate limiting.
    """
    
    def __init__(self):
        self._requests = {}
    
    def is_allowed(self, key: str, limit: int = 100, window: int = 60) -> bool:
        """
        Check if request is allowed based on rate limiting.
        
        Args:
            key: Unique identifier (e.g., IP address, user ID)
            limit: Maximum requests allowed
            window: Time window in seconds
            
        Returns:
            True if request is allowed, False otherwise
        """
        
        current_time = time.time()
        
        if key not in self._requests:
            self._requests[key] = []
        
        # Remove old requests outside the window
        self._requests[key] = [
            req_time for req_time in self._requests[key]
            if current_time - req_time < window
        ]
        
        # Check if limit is exceeded
        if len(self._requests[key]) >= limit:
            return False
        
        # Add current request
        self._requests[key].append(current_time)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Get rate limiter instance."""
    return rate_limiter


def get_gemini_model():
    """Get Gemini model instance."""
    return get_gemini_client()
