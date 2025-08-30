"""
Chat summarization service using OpenAI.
"""

from typing import Dict, Any
from app.core.config import get_settings
from app.core.dependencies import get_openai_client
from app.core.exceptions import OpenAIError, ValidationError
from app.core.logging import get_logger
from app.shared.utils import retry_with_backoff, Timer

settings = get_settings()
logger = get_logger(__name__)


class ChatSummarizer:
    """Service for summarizing chat conversations."""
    
    def __init__(self):
        self.client = get_openai_client()
        self.model = settings.OPENAI_MODEL_GPT35
        self.temperature = 0
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def summarize_chat(self, chat_conversation: Dict[str, Any]) -> str:
        """
        Summarizes a chat conversation using OpenAI.

        Args:
            chat_conversation: A dictionary containing a list of messages.
                              Expected format: {"messages": [{"sender": "Name", "text": "..."}]}

        Returns:
            A formatted summary of the chat conversation or an error message.
        """
        try:
            # Validate input
            messages = chat_conversation.get("messages", [])
            if not messages:
                raise ValidationError("Chat conversation is empty or has an invalid format")

            # Format the chat log
            formatted_chat_log = "\n".join(
                [f"{msg['sender']}: {msg['text']}" for msg in messages]
            )

            with Timer("chat_summarization"):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert in conversation analysis. Your task is to read a chat conversation and generate a concise, structured summary based on the provided format."
                        },
                        {
                            "role": "user",
                            "content": f"""
            Please summarize the following chat conversation:

            --- CHAT LOG ---
            {formatted_chat_log}
            --- END CHAT LOG ---

            Generate the summary in the following strict format. Do not add any extra text or explanations.

            Chat Summary:
            Summary: The conversation started around "[first 5-7 words of the very first message]..." and the latest point discussed was "[a brief summary of the last user message]...". The client seems generally [positive, neutral, or concerned - choose one] based on the interaction. Overall, [a number representing the count of distinct topics] key topics were covered.
            """
                        }
                    ]
                )
                
                summary = response.choices[0].message.content.strip()
                
                logger.info(
                    "Chat summarization completed",
                    extra={
                        "message_count": len(messages),
                        "summary_length": len(summary)
                    }
                )
                
                return summary
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Chat summarization failed: {e}")
            raise OpenAIError(f"Chat summarization failed: {str(e)}")
