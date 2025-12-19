from fastapi import APIRouter, HTTPException
from typing import List
import uuid
from datetime import datetime

from models.database_models import ChatSessionCreate, ChatMessageRequest, ChatMessageResponse
from models.schemas import CitationSchema
from utils.logging import get_logger, log_api_call, log_error_event
from services.chat_service import ChatService

router = APIRouter()
logger = get_logger(__name__)

# Initialize chat service
chat_service = ChatService()

@router.post("/chat/new", response_model=dict)
async def create_chat_session(session_data: ChatSessionCreate = None):
    """Create a new chat session"""
    import time
    start_time = time.time()

    try:
        # Use the chat service to create the session in the database
        selected_text = session_data.selected_text if session_data else None
        session_id = chat_service.create_session(selected_text=selected_text)

        # Log successful API call with performance metrics
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        log_api_call(
            logger,
            "/chat/new",
            "POST",
            200,
            duration
        )

        logger.info(f"Created new chat session: {session_id}")
        return {"sessionId": session_id}
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Calculate duration for error case
        duration = (time.time() - start_time) * 1000

        logger.error(f"Error creating chat session: {str(e)}")

        # Log error event with context
        log_error_event(
            logger,
            "session_creation_error",
            str(e),
            {"response_time_ms": duration}
        )

        # Provide user-friendly error message
        raise HTTPException(
            status_code=500,
            detail="Sorry, I encountered an error creating a new chat session. Please try again."
        )

@router.post("/chat/{sessionId}/message", response_model=ChatMessageResponse)
async def send_message(sessionId: str, message_data: ChatMessageRequest):
    """Send a message to a chat session"""
    import time
    start_time = time.time()

    try:
        logger.info(f"Processing message for session: {sessionId}")

        # Validate input
        if not message_data.message or not message_data.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        # Use the chat service to process the message with RAG logic
        # Implement context window management (T046) with default window size of 5
        response_text, citations = chat_service.process_message(
            session_id=sessionId,
            message=message_data.message,
            selected_text=message_data.selected_text,
            context_window_size=5  # Default context window size
        )

        response = ChatMessageResponse(
            response=response_text,
            citations=citations,
            session_id=sessionId
        )

        # Log successful API call with performance metrics
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        log_api_call(
            logger,
            f"/chat/{sessionId}/message",
            "POST",
            200,
            duration,
            session_id=sessionId
        )

        logger.info(f"Response generated for session: {sessionId}")
        return response
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Calculate duration for error case
        duration = (time.time() - start_time) * 1000

        logger.error(f"Error processing message for session {sessionId}: {str(e)}")

        # Log error event with context
        log_error_event(
            logger,
            "message_processing_error",
            str(e),
            {"session_id": sessionId, "response_time_ms": duration}
        )

        # Provide user-friendly error message
        raise HTTPException(
            status_code=500,
            detail="Sorry, I encountered an error processing your request. Please try again."
        )

@router.get("/chat/{sessionId}/history", response_model=List[dict])
async def get_chat_history(sessionId: str):
    """Get chat history for a session"""
    import time
    start_time = time.time()

    try:
        logger.info(f"Retrieving chat history for session: {sessionId}")

        # Use the chat service to get history from database
        history = chat_service.get_session_history(sessionId)

        # Log successful API call with performance metrics
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        log_api_call(
            logger,
            f"/chat/{sessionId}/history",
            "GET",
            200,
            duration,
            session_id=sessionId
        )

        return history
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Calculate duration for error case
        duration = (time.time() - start_time) * 1000

        logger.error(f"Error retrieving chat history for session {sessionId}: {str(e)}")

        # Log error event with context
        log_error_event(
            logger,
            "history_retrieval_error",
            str(e),
            {"session_id": sessionId, "response_time_ms": duration}
        )

        # Provide user-friendly error message
        raise HTTPException(
            status_code=500,
            detail="Sorry, I encountered an error retrieving chat history. Please try again."
        )