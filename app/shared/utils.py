"""
Shared utilities for the application.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, TypeVar
from functools import wraps

from app.core.logging import get_logger
from app.core.exceptions import ServiceError

logger = get_logger(__name__)

T = TypeVar('T')


def retry_with_backoff(
    max_retries: int = 3,
    backoff_factor: float = 1.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        backoff_factor: Multiplier for delay between retries
        exceptions: Tuple of exceptions to catch and retry on
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    delay = backoff_factor * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}",
                        extra={"function": func.__name__, "attempt": attempt + 1}
                    )
                    await asyncio.sleep(delay)
            
            logger.error(
                f"All {max_retries + 1} attempts failed for {func.__name__}",
                extra={"function": func.__name__, "final_exception": str(last_exception)}
            )
            raise ServiceError(
                f"Function {func.__name__} failed after {max_retries + 1} attempts",
                error_code="MAX_RETRIES_EXCEEDED",
                details={"last_exception": str(last_exception)}
            )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    delay = backoff_factor * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed, retrying in {delay}s: {str(e)}",
                        extra={"function": func.__name__, "attempt": attempt + 1}
                    )
                    time.sleep(delay)
            
            logger.error(
                f"All {max_retries + 1} attempts failed for {func.__name__}",
                extra={"function": func.__name__, "final_exception": str(last_exception)}
            )
            raise ServiceError(
                f"Function {func.__name__} failed after {max_retries + 1} attempts",
                error_code="MAX_RETRIES_EXCEEDED",
                details={"last_exception": str(last_exception)}
            )
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def format_messages_for_openai(messages: List[Dict[str, Any]]) -> str:
    """
    Format conversation messages for OpenAI processing.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Formatted message string
    """
    if not messages:
        return ""
    
    formatted_messages = []
    for msg in messages:
        sender = msg.get("sender", "unknown")
        content = msg.get("content") or msg.get("body") or msg.get("text", "")
        timestamp = msg.get("timestamp", "")
        
        if timestamp:
            formatted_messages.append(f"[{timestamp}] {sender}: {content}")
        else:
            formatted_messages.append(f"{sender}: {content}")
    
    return "\n".join(formatted_messages)


def truncate_conversation_history(
    messages: List[Dict[str, Any]], 
    max_length: int = 500
) -> List[Dict[str, Any]]:
    """
    Truncate conversation history to maintain performance.
    
    Args:
        messages: List of message dictionaries
        max_length: Maximum number of messages to keep
        
    Returns:
        Truncated message list
    """
    if len(messages) <= max_length:
        return messages
    
    # Keep the most recent messages
    return messages[-max_length:]


def sanitize_text(text: str, max_length: int = 10000) -> str:
    """
    Sanitize and validate text input.
    
    Args:
        text: Input text to sanitize
        max_length: Maximum allowed text length
        
    Returns:
        Sanitized text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove potential harmful characters and normalize whitespace
    sanitized = text.strip()
    
    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."
        logger.warning(f"Text truncated to {max_length} characters")
    
    return sanitized


def extract_client_context(
    client_info: Dict[str, Any], 
    messages: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Extract relevant context from client information and messages.
    
    Args:
        client_info: Client profile information
        messages: Conversation messages
        
    Returns:
        Extracted context dictionary
    """
    context = {
        "client_id": client_info.get("client_id", "unknown"),
        "client_name": client_info.get("name", ""),
        "message_count": len(messages),
        "last_message": messages[-1] if messages else None,
        "conversation_span": None,
    }
    
    # Calculate conversation span if timestamps are available
    if messages and len(messages) > 1:
        try:
            first_msg = messages[0]
            last_msg = messages[-1]
            
            first_timestamp = first_msg.get("timestamp")
            last_timestamp = last_msg.get("timestamp")
            
            if first_timestamp and last_timestamp:
                context["conversation_span"] = {
                    "start": first_timestamp,
                    "end": last_timestamp
                }
        except Exception as e:
            logger.warning(f"Error calculating conversation span: {e}")
    
    return context


def validate_openai_response(response: Any) -> bool:
    """
    Validate OpenAI API response structure.
    
    Args:
        response: OpenAI API response
        
    Returns:
        True if response is valid, False otherwise
    """
    try:
        if not response:
            return False
        
        if hasattr(response, 'choices') and response.choices:
            if hasattr(response.choices[0], 'message'):
                return bool(response.choices[0].message.content)
            elif hasattr(response.choices[0], 'text'):
                return bool(response.choices[0].text)
        
        return False
    except Exception as e:
        logger.warning(f"Error validating OpenAI response: {e}")
        return False


class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        logger.info(f"{self.name} completed in {duration:.4f} seconds")
    
    @property
    def duration(self) -> Optional[float]:
        """Get operation duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
