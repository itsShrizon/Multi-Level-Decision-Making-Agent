"""
Chat analysis services with AI agent orchestration.
"""

import json
from typing import Dict, List, Optional, Any, Literal
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from app.core.config import get_settings
from app.core.dependencies import get_openai_client
from app.core.exceptions import OpenAIError, ValidationError
from app.core.logging import get_logger
from app.shared.utils import retry_with_backoff, Timer
from app.shared.schemas import (
    EventDetails
)

settings = get_settings()
logger = get_logger(__name__)


class EventDetectionResult(BaseModel):
    """Extended event detection result for internal processing."""
    
    has_event: bool = Field(description="True if the message contains mention of a future event")
    event_details: Optional[EventDetails] = Field(default=None, description="Event details if detected")
    suggested_reminder: Optional[str] = Field(default=None, description="Suggested reminder message")
    internal_note: Optional[str] = Field(default=None, description="Internal note about the event")


class TriageAgent:
    """Agent responsible for message triage decisions."""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.model = settings.OPENAI_MODEL_GPT4
        self.temperature = 0
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def analyze(self, message: str) -> str:
        """
        Analyze message and determine primary action.
        
        Args:
            message: The message content to analyze
            
        Returns:
            Primary action: FLAG, IGNORE, or RESPOND
        """
        try:
            with Timer("triage_analysis"):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert at triaging client messages for a law firm. Your only job is to decide the primary action based on a strict priority: `FLAG` > `IGNORE` > `RESPOND`. Your output MUST be a JSON object conforming to the `TriageDecision` schema.
             - `FLAG`: For URGENT issues: legal/medical advice questions, extreme emotional distress, new injuries, threats to leave, or requests to speak to a person.
             - `IGNORE`: ONLY for simple conversation enders with NO new information (e.g., "ok", "thanks") where no input is needed.
             - `RESPOND`: For any other message needing a reply, including mild frustration or status updates.
             
             Return only a JSON object with the structure: {"primary_action": "FLAG|IGNORE|RESPOND"}"""
                        },
                        {
                            "role": "user",
                            "content": f"Analyze the following message and determine the primary action: '{message}'"
                        }
                    ]
                )
                
                result = json.loads(response.choices[0].message.content)
                action = result.get("primary_action")
                
                if action not in ["FLAG", "IGNORE", "RESPOND"]:
                    raise ValidationError(f"Invalid triage action: {action}")
                
                return action
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse triage response: {e}")
            raise OpenAIError("Invalid response format from triage analysis")
        except Exception as e:
            logger.error(f"Triage analysis failed: {e}")
            raise OpenAIError(f"Triage analysis failed: {str(e)}")


class RiskAgent:
    """Agent responsible for risk assessment."""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.model = settings.OPENAI_MODEL_GPT4
        self.temperature = 0
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def analyze(self, message: str, primary_action: str) -> Dict[str, Any]:
        """
        Analyze message for client retention risk.
        
        Args:
            message: The message content to analyze
            primary_action: The primary action from triage
            
        Returns:
            Dictionary with risk_update and risk_score
        """
        try:
            with Timer("risk_analysis"):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a risk assessment expert for a law firm. Your job is to determine the client's long-term risk of leaving. Your output MUST be a JSON object conforming to the `RiskAssessment` schema.
             - `High`: For direct threats to leave, accusations of malpractice, frantic urgency, requests for financial aid, questions about case value, or suicidal ideation.
             - `Medium`: For messages expressing frustration, negative sentiment, vague dissatisfaction, or any message that was flagged for a non-High risk reason.
             - `Low`: For all other positive or neutral messages.
             
             Additionally, provide a risk_score from 0-100:
             - For 'High' risk: score between 70-100
             - For 'Medium' risk: score between 40-69
             - For 'Low' risk: score between 0-39
             
             Return only a JSON object with the structure: {"risk_update": "High|Medium|Low", "risk_score": number}"""
                        },
                        {
                            "role": "user",
                            "content": f"Given the message '{message}' and that the triage action was '{primary_action}', what is the risk level and score?"
                        }
                    ]
                )
                
                result = json.loads(response.choices[0].message.content)
                risk_update = result.get("risk_update")
                risk_score = result.get("risk_score")
                
                if risk_update not in ["High", "Medium", "Low"]:
                    raise ValidationError(f"Invalid risk level: {risk_update}")
                    
                if not isinstance(risk_score, int) or not (0 <= risk_score <= 100):
                    raise ValidationError(f"Invalid risk score: {risk_score}")
                
                return {"risk_update": risk_update, "risk_score": risk_score}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse risk response: {e}")
            raise OpenAIError("Invalid response format from risk analysis")
        except Exception as e:
            logger.error(f"Risk analysis failed: {e}")
            raise OpenAIError(f"Risk analysis failed: {str(e)}")


class SentimentAgent:
    """Agent responsible for sentiment analysis."""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.model = settings.OPENAI_MODEL_GPT4
        self.temperature = 0
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def analyze(self, message: str) -> Dict[str, Any]:
        """
        Analyze message sentiment.
        
        Args:
            message: The message content to analyze
            
        Returns:
            Dictionary with sentiment and sentiment_score
        """
        try:
            with Timer("sentiment_analysis"):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert in sentiment analysis. Classify the sentiment of the message as Positive, Neutral, or Negative. 
            
            Additionally, provide a sentiment_score from 0-100:
            - For 'Positive' sentiment: score between 0-30
            - For 'Neutral' sentiment: score between 31-60
            - For 'Negative' sentiment: score between 61-100
            
            The exact score should reflect the intensity of the sentiment within each category.
            
            Return only a JSON object with the structure: {"sentiment": "Positive|Neutral|Negative", "sentiment_score": number}"""
                        },
                        {
                            "role": "user",
                            "content": f"Message: '{message}'"
                        }
                    ]
                )
                
                result = json.loads(response.choices[0].message.content)
                sentiment = result.get("sentiment")
                sentiment_score = result.get("sentiment_score")
                
                if sentiment not in ["Positive", "Neutral", "Negative"]:
                    raise ValidationError(f"Invalid sentiment: {sentiment}")
                    
                if not isinstance(sentiment_score, int) or not (0 <= sentiment_score <= 100):
                    raise ValidationError(f"Invalid sentiment score: {sentiment_score}")
                
                return {"sentiment": sentiment, "sentiment_score": sentiment_score}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse sentiment response: {e}")
            raise OpenAIError("Invalid response format from sentiment analysis")
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            raise OpenAIError(f"Sentiment analysis failed: {str(e)}")


class EventDetectionAgent:
    """Agent responsible for detecting events and appointments."""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.model = settings.OPENAI_MODEL_GPT4
        self.temperature = 0
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def analyze(self, message: str) -> Dict[str, Any]:
        """
        Analyze message for events and appointments.
        
        Args:
            message: The message content to analyze
            
        Returns:
            Dictionary with event detection results
        """
        try:
            with Timer("event_detection"):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an expert at identifying mentions of future events or appointments in messages. 
             Your job is to detect if a message contains a future event, extract its details, and suggest a reminder. 
             
             When an event is detected:
             1. Extract details like date, time, event type, location
             2. Craft a helpful reminder message to send to the client before the event
             3. Create an internal note with context about the event
             
             Return a JSON object with this structure:
             {
               "has_event": boolean,
               "event_details": {
                 "date": "string or null",
                 "time": "string or null", 
                 "location": "string or null",
                 "event_type": "string or null",
                 "additional_info": "string or null"
               } or null,
               "suggested_reminder": "string or null",
               "internal_note": "string or null"
             }
             
             If no event is mentioned, return has_event as false and other fields as null."""
                        },
                        {
                            "role": "user",
                            "content": f"Analyze the following message for any mentions of future events or appointments: '{message}'"
                        }
                    ]
                )
                
                result = json.loads(response.choices[0].message.content)
                
                # Validate the structure
                has_event = result.get("has_event", False)
                if not isinstance(has_event, bool):
                    raise ValidationError("has_event must be a boolean")
                
                return result
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse event detection response: {e}")
            raise OpenAIError("Invalid response format from event detection")
        except Exception as e:
            logger.error(f"Event detection failed: {e}")
            raise OpenAIError(f"Event detection failed: {str(e)}")


class ResponseGenerator:
    """Agent responsible for generating contextual responses."""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        self.model = settings.OPENAI_MODEL_GPT4
        self.temperature = 0.7
    
    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def generate(
        self, 
        conversation_history: List[Dict[str, Any]], 
        primary_action: str, 
        risk_update: str
    ) -> Optional[str]:
        """
        Generate contextual response based on analysis.
        
        Args:
            conversation_history: List of conversation messages
            primary_action: The primary action from triage
            risk_update: The risk level assessment
            
        Returns:
            Generated response or None if no response needed
        """
        # Skip response for high-risk flagged messages or ignore actions
        if (primary_action == "FLAG" and risk_update == "High") or primary_action == "IGNORE":
            return None
        
        if primary_action not in ["RESPOND", "FLAG"]:
            return None
        
        try:
            with Timer("response_generation"):
                last_message = conversation_history[-1] if conversation_history else {}
                last_message_content = last_message.get("content", "")
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[
                        {
                            "role": "system",
                            "content": """You are a friendly, empathetic AI assistant for a law firm. Your only job is to write a short, human-sounding text message based on an action and the client's last message. 
        
        **Your top priority is to match the client's tone and sentiment to sound as human as possible.** Analyze their message and mirror their style, whether it's formal, casual, frustrated, or happy. Your response MUST be appropriate for the urgency and sentiment of the client's message.

        - If the action is "RESPOND", write a direct, empathetic response to their message that matches their tone.
        - If the action is "FLAG", you must escalate to a human. 
          1. First, **acknowledge the client's situation with empathy that matches the seriousness of their message.**
          2. Then, explain that you are getting a team member to help immediately.
          3. **Crucially, if the message is urgent or serious (e.g., involves medical distress), do NOT use casual phrases like "That's a great question."** Your tone must match the seriousness of the situation.

        For example, for an urgent message, a good response might be: "I understand this is an urgent matter, and I want to get you the right help immediately. I'm flagging this for a team member to review right away."

        Generate ONLY the response text. Do not include quotes or additional formatting."""
                        },
                        {
                            "role": "user",
                            "content": f"""Action: **{primary_action}**
Client's last message: "{last_message_content}"

Generate a response that matches the client's tone and the action required."""
                        }
                    ]
                )
                
                generated_response = response.choices[0].message.content.strip()
                
                # Clean up response
                generated_response = generated_response.strip('"').strip("'")
                
                return generated_response
                
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise OpenAIError(f"Response generation failed: {str(e)}")


class ChatOrchestrator:
    """Orchestrates the various AI agents for comprehensive chat analysis."""
    
    def __init__(self):
        self.client = get_openai_client()
        self.triage_agent = TriageAgent(self.client)
        self.risk_agent = RiskAgent(self.client)
        self.sentiment_agent = SentimentAgent(self.client)
        self.event_detection_agent = EventDetectionAgent(self.client)
        self.response_generator = ResponseGenerator(self.client)
    
    async def analyze_message(
        self, 
        client_info: Dict[str, Any], 
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a client message.
        
        Args:
            client_info: Client information and profile
            conversation_history: List of conversation messages
            
        Returns:
            Complete analysis results
        """
        if not conversation_history:
            raise ValidationError("Conversation history cannot be empty")
        
        current_message = conversation_history[-1].get("content", "")
        if not current_message:
            raise ValidationError("Current message content is empty")
        
        try:
            with Timer("complete_message_analysis"):
                # Run specialist agents in parallel for better performance
                import asyncio
                
                triage_task = self.triage_agent.analyze(current_message)
                sentiment_task = self.sentiment_agent.analyze(current_message)
                event_task = self.event_detection_agent.analyze(current_message)
                
                # Wait for parallel tasks
                primary_action, sentiment_result, event_detection = await asyncio.gather(
                    triage_task, sentiment_task, event_task
                )
                
                # Risk analysis depends on triage result
                risk_result = await self.risk_agent.analyze(current_message, primary_action)
                
                # Generate response based on analysis
                response_to_send = await self.response_generator.generate(
                    conversation_history, primary_action, risk_result["risk_update"]
                )
                
                # Assemble final result
                result = {
                    "action": primary_action,
                    "risk_update": risk_result["risk_update"],
                    "risk_score": risk_result["risk_score"],
                    "sentiment": sentiment_result["sentiment"],
                    "sentiment_score": sentiment_result["sentiment_score"],
                    "response_to_send": response_to_send,
                    "event_detection": event_detection,
                    "full_analysis": {
                        "primary_action": primary_action,
                        "risk_update": risk_result["risk_update"],
                        "sentiment": sentiment_result["sentiment"],
                        "event_detection": event_detection
                    }
                }
                
                logger.info(
                    "Message analysis completed",
                    extra={
                        "client_id": client_info.get("client_id", "unknown"),
                        "action": primary_action,
                        "risk": risk_result["risk_update"],
                        "sentiment": sentiment_result["sentiment"],
                        "has_event": event_detection.get("has_event", False)
                    }
                )
                
                return result
                
        except Exception as e:
            logger.error(f"Message analysis failed: {e}")
            # Return safe default on error
            return {
                "action": "FLAG",
                "risk_update": "High",
                "risk_score": 100,
                "sentiment": "Neutral",
                "sentiment_score": 50,
                "response_to_send": None,
                "event_detection": {"has_event": False},
                "full_analysis": {
                    "error": str(e)
                }
            }
