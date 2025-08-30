"""
Outbound message API routes.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.core.exceptions import ValidationError, ServiceError
from app.shared.schemas import (
    BaseResponse,
    OutboundMessageRequest,
    OutboundMessageResult,
)
from app.shared.utils import sanitize_text, format_messages_for_openai
from app.features.outbound.services import OutboundMessageGenerator, MessageScheduler

logger = get_logger(__name__)
router = APIRouter()


# Additional request models specific to outbound messaging
class FollowUpRequest(BaseModel):
    """Request for generating follow-up messages."""
    
    original_message: str = Field(description="The original outbound message sent")
    client_response: Optional[str] = Field(default=None, description="Client's response (if any)")
    follow_up_type: str = Field(default="standard", description="Type of follow-up")


class AppointmentReminderRequest(BaseModel):
    """Request for generating appointment reminders."""
    
    appointment_details: Dict[str, Any] = Field(description="Appointment information")
    client_name: Optional[str] = Field(default=None, description="Client name for personalization")
    reminder_type: str = Field(default="standard", description="Type of reminder")


class CaseUpdateRequest(BaseModel):
    """Request for generating case update messages."""
    
    case_info: Dict[str, Any] = Field(description="Case information and update details")
    update_type: str = Field(description="Type of update")
    client_context: Optional[Dict[str, Any]] = Field(default=None, description="Client context")


class WeeklyCheckinRequest(BaseModel):
    """Request for scheduling weekly check-ins."""
    
    client_id: str = Field(description="Client identifier")
    message_history: List[Dict[str, Any]] = Field(description="Recent message history")
    preferences: Dict[str, Any] = Field(description="Client preferences for scheduling")


class AppointmentReminderScheduleRequest(BaseModel):
    """Request for scheduling appointment reminders."""
    
    client_id: str = Field(description="Client identifier")
    appointment_details: Dict[str, Any] = Field(description="Appointment information")
    reminder_schedule: Optional[List[str]] = Field(default=None, description="Reminder types to schedule")


# Dependencies
def get_outbound_generator() -> OutboundMessageGenerator:
    """Get outbound message generator instance."""
    return OutboundMessageGenerator()


def get_message_scheduler() -> MessageScheduler:
    """Get message scheduler instance."""
    return MessageScheduler()


@router.post("/generate", response_model=OutboundMessageResult)
async def generate_outbound_message(
    request: OutboundMessageRequest,
    background_tasks: BackgroundTasks,
    generator: OutboundMessageGenerator = Depends(get_outbound_generator)
) -> JSONResponse:
    """
    Generate a proactive outbound message based on conversation history and context.
    
    This endpoint analyzes the full conversation history to understand client mood,
    tone, and preferences, then generates an appropriate weekly check-in or 
    proactive communication message.
    """
    try:
        # Validate input
        if not request.information.strip():
            raise ValidationError("Information/context for outbound message is required")
        
        if not request.messages:
            raise ValidationError("Message history is required for context")
        
        # Sanitize information
        sanitized_info = sanitize_text(request.information)
        
        # Sanitize and prepare messages
        sanitized_messages = []
        for msg in request.messages:
            content = msg.get("content") or msg.get("body", "")
            sanitized_content = sanitize_text(content)
            if sanitized_content:
                sanitized_messages.append({
                    "timestamp": msg.get("timestamp"),
                    "sender": msg.get("sender", "unknown"),
                    "content": sanitized_content
                })
        
        if not sanitized_messages:
            raise ValidationError("No valid messages found after sanitization")
        
        logger.info(
            "Starting outbound message generation",
            extra={
                "message_count": len(sanitized_messages),
                "info_length": len(sanitized_info)
            }
        )
        
        # Generate outbound message
        message = await generator.generate_outbound_message(
            information=sanitized_info,
            messages=sanitized_messages
        )
        
        # Log for analytics
        background_tasks.add_task(
            log_outbound_message,
            message_type="proactive",
            message_length=len(message),
            context_messages=len(sanitized_messages)
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {"message": message},
                "metadata": {
                    "processed_messages": len(sanitized_messages),
                    "message_length": len(message),
                    "context_provided": bool(sanitized_info)
                }
            }
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error in outbound message generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        logger.error(f"Service error in outbound message generation: {e}")
        raise HTTPException(status_code=503, detail="Outbound message service temporarily unavailable")
    except Exception as e:
        logger.error(f"Unexpected error in outbound message generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/follow-up")
async def generate_follow_up(
    request: FollowUpRequest,
    background_tasks: BackgroundTasks,
    generator: OutboundMessageGenerator = Depends(get_outbound_generator)
) -> JSONResponse:
    """
    Generate a follow-up message based on original message and client response.
    
    This endpoint creates appropriate follow-up communications that consider
    the client's response (or lack thereof) to the original message.
    """
    try:
        # Validate input
        if not request.original_message.strip():
            raise ValidationError("Original message is required")
        
        logger.info(
            "Starting follow-up message generation",
            extra={
                "follow_up_type": request.follow_up_type,
                "has_client_response": bool(request.client_response)
            }
        )
        
        # Generate follow-up message
        follow_up = await generator.generate_follow_up_message(
            original_message=sanitize_text(request.original_message),
            client_response=sanitize_text(request.client_response) if request.client_response else None,
            follow_up_type=request.follow_up_type
        )
        
        # Log for analytics
        background_tasks.add_task(
            log_outbound_message,
            message_type="follow_up",
            message_length=len(follow_up),
            follow_up_type=request.follow_up_type
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {"message": follow_up},
                "metadata": {
                    "follow_up_type": request.follow_up_type,
                    "original_message_length": len(request.original_message),
                    "has_client_response": bool(request.client_response)
                }
            }
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error in follow-up generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        logger.error(f"Service error in follow-up generation: {e}")
        raise HTTPException(status_code=503, detail="Follow-up service temporarily unavailable")
    except Exception as e:
        logger.error(f"Unexpected error in follow-up generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/appointment-reminder")
async def generate_appointment_reminder(
    request: AppointmentReminderRequest,
    background_tasks: BackgroundTasks,
    generator: OutboundMessageGenerator = Depends(get_outbound_generator)
) -> JSONResponse:
    """
    Generate appointment reminder messages.
    
    This endpoint creates contextual appointment reminders that can be sent
    at different intervals (advance, day-before, same-day) with appropriate
    timing and urgency.
    """
    try:
        # Validate input
        if not request.appointment_details:
            raise ValidationError("Appointment details are required")
        
        logger.info(
            "Starting appointment reminder generation",
            extra={
                "reminder_type": request.reminder_type,
                "has_client_name": bool(request.client_name),
                "appointment_type": request.appointment_details.get("type", "unknown")
            }
        )
        
        # Generate appointment reminder
        reminder = await generator.generate_appointment_reminder(
            appointment_details=request.appointment_details,
            client_name=request.client_name,
            reminder_type=request.reminder_type
        )
        
        # Log for analytics
        background_tasks.add_task(
            log_outbound_message,
            message_type="appointment_reminder",
            message_length=len(reminder),
            reminder_type=request.reminder_type
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {"message": reminder},
                "metadata": {
                    "reminder_type": request.reminder_type,
                    "appointment_type": request.appointment_details.get("type"),
                    "client_personalized": bool(request.client_name)
                }
            }
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error in appointment reminder generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        logger.error(f"Service error in appointment reminder generation: {e}")
        raise HTTPException(status_code=503, detail="Appointment reminder service temporarily unavailable")
    except Exception as e:
        logger.error(f"Unexpected error in appointment reminder generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/case-update")
async def generate_case_update(
    request: CaseUpdateRequest,
    background_tasks: BackgroundTasks,
    generator: OutboundMessageGenerator = Depends(get_outbound_generator)
) -> JSONResponse:
    """
    Generate case update messages for clients.
    
    This endpoint creates professional case update communications that inform
    clients about progress, milestones, or required actions in their cases.
    """
    try:
        # Validate input
        if not request.case_info:
            raise ValidationError("Case information is required")
        
        if not request.update_type:
            raise ValidationError("Update type is required")
        
        logger.info(
            "Starting case update generation",
            extra={
                "update_type": request.update_type,
                "case_id": request.case_info.get("case_id", "unknown"),
                "has_client_context": bool(request.client_context)
            }
        )
        
        # Generate case update
        update_message = await generator.generate_case_update_message(
            case_info=request.case_info,
            update_type=request.update_type,
            client_context=request.client_context
        )
        
        # Log for analytics
        background_tasks.add_task(
            log_outbound_message,
            message_type="case_update",
            message_length=len(update_message),
            update_type=request.update_type
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {"message": update_message},
                "metadata": {
                    "update_type": request.update_type,
                    "case_id": request.case_info.get("case_id"),
                    "has_client_context": bool(request.client_context)
                }
            }
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error in case update generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        logger.error(f"Service error in case update generation: {e}")
        raise HTTPException(status_code=503, detail="Case update service temporarily unavailable")
    except Exception as e:
        logger.error(f"Unexpected error in case update generation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/schedule/weekly-checkin")
async def schedule_weekly_checkin(
    request: WeeklyCheckinRequest,
    background_tasks: BackgroundTasks,
    scheduler: MessageScheduler = Depends(get_message_scheduler)
) -> JSONResponse:
    """
    Schedule a weekly check-in message for a client.
    
    This endpoint sets up automated weekly communications based on client
    preferences and conversation history.
    """
    try:
        # Validate input
        if not request.client_id:
            raise ValidationError("Client ID is required")
        
        if not request.message_history:
            raise ValidationError("Message history is required")
        
        logger.info(
            "Starting weekly check-in scheduling",
            extra={
                "client_id": request.client_id,
                "preferred_day": request.preferences.get("preferred_day", "Monday"),
                "preferred_time": request.preferences.get("preferred_time", "10:00 AM")
            }
        )
        
        # Schedule weekly check-in
        scheduling_info = await scheduler.schedule_weekly_checkin(
            client_id=request.client_id,
            message_history=request.message_history,
            preferences=request.preferences
        )
        
        # Log for analytics
        background_tasks.add_task(
            log_message_scheduling,
            client_id=request.client_id,
            message_type="weekly_checkin",
            scheduling_info=scheduling_info
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": scheduling_info,
                "metadata": {
                    "client_id": request.client_id,
                    "schedule_created": True
                }
            }
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error in weekly check-in scheduling: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        logger.error(f"Service error in weekly check-in scheduling: {e}")
        raise HTTPException(status_code=503, detail="Scheduling service temporarily unavailable")
    except Exception as e:
        logger.error(f"Unexpected error in weekly check-in scheduling: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/schedule/appointment-reminders")
async def schedule_appointment_reminders(
    request: AppointmentReminderScheduleRequest,
    background_tasks: BackgroundTasks,
    scheduler: MessageScheduler = Depends(get_message_scheduler)
) -> JSONResponse:
    """
    Schedule multiple appointment reminder messages.
    
    This endpoint sets up a series of appointment reminders at different
    intervals leading up to the appointment.
    """
    try:
        # Validate input
        if not request.client_id:
            raise ValidationError("Client ID is required")
        
        if not request.appointment_details:
            raise ValidationError("Appointment details are required")
        
        logger.info(
            "Starting appointment reminder scheduling",
            extra={
                "client_id": request.client_id,
                "appointment_id": request.appointment_details.get("appointment_id"),
                "reminder_types": request.reminder_schedule or ["advance", "day_before", "same_day"]
            }
        )
        
        # Schedule appointment reminders
        scheduled_reminders = await scheduler.schedule_appointment_reminders(
            client_id=request.client_id,
            appointment_details=request.appointment_details,
            reminder_schedule=request.reminder_schedule
        )
        
        # Log for analytics
        background_tasks.add_task(
            log_message_scheduling,
            client_id=request.client_id,
            message_type="appointment_reminders",
            scheduling_info={"reminders": scheduled_reminders}
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": {"scheduled_reminders": scheduled_reminders},
                "metadata": {
                    "client_id": request.client_id,
                    "reminders_scheduled": len(scheduled_reminders),
                    "appointment_id": request.appointment_details.get("appointment_id")
                }
            }
        )
        
    except ValidationError as e:
        logger.warning(f"Validation error in appointment reminder scheduling: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except ServiceError as e:
        logger.error(f"Service error in appointment reminder scheduling: {e}")
        raise HTTPException(status_code=503, detail="Scheduling service temporarily unavailable")
    except Exception as e:
        logger.error(f"Unexpected error in appointment reminder scheduling: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def health_check():
    """Health check endpoint for the outbound messaging service."""
    return {
        "status": "healthy",
        "service": "outbound_messaging",
        "features": [
            "message_generation",
            "follow_up_messages",
            "appointment_reminders",
            "case_updates",
            "message_scheduling"
        ]
    }


# Background task functions
async def log_outbound_message(
    message_type: str,
    message_length: int,
    **kwargs
) -> None:
    """Log outbound message generation for analytics."""
    try:
        logger.info(
            "Outbound message generated and logged",
            extra={
                "message_type": message_type,
                "message_length": message_length,
                "timestamp": "utc_now",
                **kwargs
            }
        )
        
        # Here you could send data to analytics service, database, etc.
        # await analytics_service.log_outbound_message(message_type, message_length, kwargs)
        
    except Exception as e:
        logger.error(f"Failed to log outbound message: {e}")


async def log_message_scheduling(
    client_id: str,
    message_type: str,
    scheduling_info: Dict[str, Any]
) -> None:
    """Log message scheduling for tracking and analytics."""
    try:
        logger.info(
            "Message scheduling logged",
            extra={
                "client_id": client_id,
                "message_type": message_type,
                "scheduling_info": scheduling_info,
                "timestamp": "utc_now"
            }
        )
        
        # Here you could send data to scheduling service, database, etc.
        # await scheduling_service.log_scheduled_message(client_id, message_type, scheduling_info)
        
    except Exception as e:
        logger.error(f"Failed to log message scheduling: {e}")
