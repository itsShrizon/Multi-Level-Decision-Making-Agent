"""
Text processing services for making content concise.
"""

from app.core.config import get_settings
from app.core.dependencies import get_openai_client
from app.core.exceptions import OpenAIError, ValidationError
from app.core.logging import get_logger
from app.shared.utils import retry_with_backoff, Timer

settings = get_settings()
logger = get_logger(__name__)


class TextProcessor:
    """Service for processing and optimizing text content."""
    
    def __init__(self):
        self.client = get_openai_client()
        self.model = settings.OPENAI_MODEL_GPT4
        self.temperature = 0.0
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def make_concise(self, text_to_summarize: str) -> str:
        """
        Uses OpenAI to make a given text more concise.

        Args:
            text_to_summarize: The string of text you want to shorten.

        Returns:
            A string containing the concise version of the text.
        """
        try:
            if not text_to_summarize or not text_to_summarize.strip():
                raise ValidationError("Please provide some text to make concise")

            with Timer("text_concisification"):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant skilled at making text more concise and clear while retaining the original meaning within 3-4 words not more than 4 words."
                        },
                        {
                            "role": "user",
                            "content": f"Please make the following text more concise:\n\n---\n\n{text_to_summarize}"
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=settings.OPENAI_MAX_TOKENS,
                    top_p=0.9,
                    frequency_penalty=1.3,
                    presence_penalty=0.0
                )

                concise_text = response.choices[0].message.content.strip()
                
                logger.info(
                    "Text concisification completed",
                    extra={
                        "original_length": len(text_to_summarize),
                        "concise_length": len(concise_text),
                        "reduction_ratio": round(
                            (len(text_to_summarize) - len(concise_text)) / len(text_to_summarize) * 100, 2
                        )
                    }
                )
                
                return concise_text
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Text concisification failed: {e}")
            raise OpenAIError(f"Text concisification failed: {str(e)}")
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def extract_keywords(self, text: str, max_keywords: int = 10) -> list[str]:
        """
        Extract key terms and phrases from text.
        
        Args:
            text: Input text to analyze
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        try:
            if not text or not text.strip():
                raise ValidationError("Please provide text for keyword extraction")

            with Timer("keyword_extraction"):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"Extract up to {max_keywords} key terms and phrases from the given text. Return only the keywords separated by commas, no other text."
                        },
                        {
                            "role": "user",
                            "content": f"Extract keywords from this text:\n\n{text}"
                        }
                    ],
                    temperature=0.0,
                    max_tokens=200
                )

                keywords_text = response.choices[0].message.content.strip()
                keywords = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
                
                logger.info(
                    "Keywords extracted",
                    extra={
                        "text_length": len(text),
                        "keywords_count": len(keywords)
                    }
                )
                
                return keywords[:max_keywords]
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Keyword extraction failed: {e}")
            raise OpenAIError(f"Keyword extraction failed: {str(e)}")
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def classify_urgency(self, text: str) -> str:
        """
        Classify the urgency level of a text message.
        
        Args:
            text: Text to classify
            
        Returns:
            Urgency level: "Low", "Medium", or "High"
        """
        try:
            if not text or not text.strip():
                raise ValidationError("Please provide text for urgency classification")

            with Timer("urgency_classification"):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Classify the urgency level of the given text as Low, Medium, or High. Consider language intensity, time sensitivity, and emotional state. Return only one word: Low, Medium, or High."
                        },
                        {
                            "role": "user",
                            "content": f"Classify the urgency of this message:\n\n{text}"
                        }
                    ],
                    temperature=0.0,
                    max_tokens=10
                )

                urgency = response.choices[0].message.content.strip()
                
                if urgency not in ["Low", "Medium", "High"]:
                    urgency = "Medium"  # Default fallback
                
                logger.info(
                    "Urgency classification completed",
                    extra={
                        "text_length": len(text),
                        "urgency_level": urgency
                    }
                )
                
                return urgency
                
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Urgency classification failed: {e}")
            raise OpenAIError(f"Urgency classification failed: {str(e)}")
