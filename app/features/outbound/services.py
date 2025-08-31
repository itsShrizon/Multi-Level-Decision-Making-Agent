"""
Outbound message generation services.
Refactored for clarity, maintainability, and robustness.
"""

import json
from typing import Dict, List, Any, Optional

from app.core.config import get_settings
from app.core.dependencies import get_openai_client
from app.core.exceptions import OpenAIError, ValidationError
from app.core.logging import get_logger
from app.shared.utils import retry_with_backoff, Timer

settings = get_settings()
logger = get_logger(__name__)


class OutboundMessageGenerator:
    """Service for generating proactive outbound messages to clients."""

    # REFACTOR: Prompts are now class constants for easier management and reuse.
    PROMPT_SYSTEM_OUTBOUND = (
        "You are a professional outbound message drafting assistant for a law firm. "
        "First, silently assess the client's overall mood, tone, and seriousness from the entire history. "
        "Then craft one empathetic, concise, professional weekly check-in message (not a reply) that acknowledges context and any stated preferences. "
        "Incorporate the provided scheduling/timing subtly (e.g., weekly cadence or specific day/time) without sounding robotic. "
        "Do not include your analysis or labels—output only the final message text. "
        "Keep it natural, helpful, and human; avoid overpromising or legal advice; be clear and respectful."
    )
    PROMPT_SYSTEM_FOLLOW_UP = (
        "You are generating a follow-up message for a law firm client. "
        "Based on the original message and any client response, create an appropriate follow-up. "
        "Keep it professional, brief, and contextually relevant. "
        "Do not be overly persistent or pushy."
    )
    PROMPT_SYSTEM_APPOINTMENT = (
        "You are generating an appointment reminder for a law firm client. "
        "{timing_context} Create a professional, helpful reminder that includes relevant details. "
        "Be clear about what the client needs to do or bring. "
        "Include contact information for questions or changes."
    )
    PROMPT_SYSTEM_CASE_UPDATE = (
        "You are generating a case update message for a law firm client. "
        "{update_guidance} Be clear, professional, and reassuring. Avoid legal jargon. "
        "If action is required from the client, make it very clear what they need to do and by when."
    )

    def __init__(self):
        self.client = get_openai_client()
        self.model = settings.OPENAI_MODEL_GPT4

    # REFACTOR: Centralized private method for all OpenAI calls to avoid repetition.
    async def _generate_completion(
        self,
        system_prompt: str,
        user_content: str,
        temperature: float
    ) -> str:
        """
        Private helper to make the chat completion request and handle the response.
        """
        try:
            with Timer(f"openai_completion_{system_prompt[:20]}"):
                response = await self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                )
            
            message = response.choices[0].message.content.strip()
            # Clean up potential quotes returned by the model
            return message.strip('"').strip("'")

        except Exception as e:
            # The calling function will catch this and log the specific context.
            raise e

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def generate_outbound_message(
        self,
        information: str,
        messages: List[Dict[str, Any]]
    ) -> str:
        """
        Analyze conversation history and generate a contextual outbound message.
        """
        if not information or not information.strip():
            raise ValidationError("Information/context for outbound message is required")
        if not messages:
            raise ValidationError("Message history is required for context")

        try:
            # REFACTOR: Cleaner message normalization using a list comprehension.
            history = [
                {
                    "timestamp": m.get("timestamp"),
                    "sender": m.get("sender", "unknown"),
                    "content": m.get("content") or m.get("body") or ""
                }
                for m in messages
            ]
            
            # REFACTOR: Payload creation is more explicit.
            user_payload = {
                "objective_and_timing": information,
                "full_message_history": history
            }
            
            user_content = (
                "Based on this JSON, produce exactly one outbound weekly check-in message (text only, not a reply):\n"
                + json.dumps(user_payload, ensure_ascii=False)
            )

            message = await self._generate_completion(
                system_prompt=self.PROMPT_SYSTEM_OUTBOUND,
                user_content=user_content,
                temperature=0.5
            )
            
            logger.info(
                "Outbound message generated",
                extra={
                    "message_length": len(message),
                    "context_messages": len(messages),
                    "information_length": len(information)
                }
            )
            return message
            
        except Exception as e:
            logger.error(f"Outbound message generation failed: {e}", exc_info=True)
            raise OpenAIError(f"Outbound message generation failed: {str(e)}")

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def generate_follow_up_message(
        self,
        original_message: str,
        client_response: Optional[str] = None,
        follow_up_type: str = "standard"
    ) -> str:
        """
        Generate a follow-up message based on client response or lack thereof.
        """
        try:
            context = {
                "original_message": original_message,
                "client_response": client_response,
                "follow_up_type": follow_up_type
            }
            
            # REFACTOR: Using a dictionary for prompt snippets is cleaner than if/elif.
            prompt_additions = {
                "urgent": " This is an urgent follow-up, so convey appropriate urgency while remaining professional.",
                "reminder": " This is a gentle reminder follow-up."
            }
            system_prompt = self.PROMPT_SYSTEM_FOLLOW_UP + prompt_additions.get(follow_up_type, "")

            user_content = f"Generate a follow-up message based on this context:\n{json.dumps(context)}"
            
            follow_up = await self._generate_completion(
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=0.5
            )
            
            logger.info(
                "Follow-up message generated",
                extra={
                    "follow_up_type": follow_up_type,
                    "has_client_response": bool(client_response),
                    "message_length": len(follow_up)
                }
            )
            return follow_up
            
        except Exception as e:
            logger.error(f"Follow-up message generation failed: {e}", exc_info=True)
            raise OpenAIError(f"Follow-up message generation failed: {str(e)}")

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def generate_appointment_reminder(
        self,
        appointment_details: Dict[str, Any],
        client_name: Optional[str] = None,
        reminder_type: str = "standard"
    ) -> str:
        """
        Generate appointment reminder messages.
        """
        try:
            context = {
                "appointment_details": appointment_details,
                "client_name": client_name,
                "reminder_type": reminder_type
            }
            
            timing_context = {
                "advance": "This is an advance reminder sent several days before the appointment.",
                "day_before": "This is a day-before reminder.",
                "same_day": "This is a same-day reminder sent on the morning of the appointment."
            }.get(reminder_type, "This is a standard appointment reminder.")
            
            system_prompt = self.PROMPT_SYSTEM_APPOINTMENT.format(timing_context=timing_context)
            user_content = f"Generate an appointment reminder based on this context:\n{json.dumps(context)}"

            reminder = await self._generate_completion(
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=0.3
            )
            
            logger.info(
                "Appointment reminder generated",
                extra={
                    "reminder_type": reminder_type,
                    "has_client_name": bool(client_name),
                    "appointment_type": appointment_details.get("type", "unknown")
                }
            )
            return reminder
            
        except Exception as e:
            logger.error(f"Appointment reminder generation failed: {e}", exc_info=True)
            raise OpenAIError(f"Appointment reminder generation failed: {str(e)}")

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    async def generate_case_update_message(
        self,
        case_info: Dict[str, Any],
        update_type: str,
        client_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate case update messages for clients.
        """
        try:
            context = {
                "case_info": case_info,
                "update_type": update_type,
                "client_context": client_context or {}
            }
            
            update_guidance = {
                "progress": "This is a general progress update. Focus on what has been accomplished and next steps.",
                "milestone": "This is a milestone update about a significant development in the case.",
                "requirement": "This is about a requirement or action needed from the client."
            }.get(update_type, "This is a general case update.")

            system_prompt = self.PROMPT_SYSTEM_CASE_UPDATE.format(update_guidance=update_guidance)
            user_content = f"Generate a case update message based on this context:\n{json.dumps(context)}"

            update_message = await self._generate_completion(
                system_prompt=system_prompt,
                user_content=user_content,
                temperature=0.4
            )
            
            logger.info(
                "Case update message generated",
                extra={
                    "update_type": update_type,
                    "case_id": case_info.get("case_id", "unknown"),
                    "message_length": len(update_message)
                }
            )
            return update_message
            
        except Exception as e:
            logger.error(f"Case update message generation failed: {e}", exc_info=True)
            raise OpenAIError(f"Case update message generation failed: {str(e)}")


class MessageScheduler:
    """Service for managing outbound message scheduling and delivery."""
    
    # REFACTOR: Use dependency injection for better decoupling and testing.
    def __init__(self, generator: OutboundMessageGenerator):
        self.generator = generator
    
    async def schedule_weekly_checkin(
        self,
        client_id: str,
        message_history: List[Dict[str, Any]],
        preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Schedule a weekly check-in message for a client.
        """
        try:
            preferred_day = preferences.get("preferred_day", "Monday")
            preferred_time = preferences.get("preferred_time", "10:00 AM")
            context_info = f"Weekly check-in scheduled for {preferred_day} at {preferred_time}"
            
            message = await self.generator.generate_outbound_message(
                information=context_info,
                messages=message_history
            )
            
            scheduling_info = {
                "client_id": client_id,
                "message_type": "weekly_checkin",
                "message_content": message,
                "scheduled_day": preferred_day,
                "scheduled_time": preferred_time,
                "status": "scheduled"
            }
            
            logger.info(
                "Weekly check-in scheduled",
                extra={
                    "client_id": client_id,
                    "scheduled_day": scheduling_info["scheduled_day"],
                    "scheduled_time": scheduling_info["scheduled_time"]
                }
            )
            return scheduling_info
            
        except Exception as e:
            logger.error(f"Failed to schedule weekly check-in for client {client_id}: {e}", exc_info=True)
            raise OpenAIError(f"Failed to schedule weekly check-in: {str(e)}")
    
    async def schedule_appointment_reminders(
        self,
        client_id: str,
        appointment_details: Dict[str, Any],
        reminder_schedule: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Schedule multiple appointment reminders.
        """
        # REFACTOR: Define the default directly in the signature for clarity.
        if reminder_schedule is None:
            reminder_schedule = ["advance", "day_before", "same_day"]
        
        try:
            scheduled_reminders = []
            client_name = appointment_details.get("client_name")
            
            for reminder_type in reminder_schedule:
                reminder_message = await self.generator.generate_appointment_reminder(
                    appointment_details=appointment_details,
                    client_name=client_name,
                    reminder_type=reminder_type
                )
                
                reminder_info = {
                    "client_id": client_id,
                    "message_type": f"appointment_reminder_{reminder_type}",
                    "message_content": reminder_message,
                    "appointment_id": appointment_details.get("appointment_id"),
                    "reminder_type": reminder_type,
                    "status": "scheduled"
                }
                scheduled_reminders.append(reminder_info)
            
            logger.info(
                "Appointment reminders scheduled",
                extra={
                    "client_id": client_id,
                    "appointment_id": appointment_details.get("appointment_id"),
                    "reminder_count": len(scheduled_reminders)
                }
            )
            return scheduled_reminders
            
        except Exception as e:
            logger.error(f"Failed to schedule appointment reminders for client {client_id}: {e}", exc_info=True)
            raise OpenAIError(f"Failed to schedule appointment reminders: {str(e)}")