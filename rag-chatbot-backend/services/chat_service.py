from typing import List, Dict, Optional
import uuid
from datetime import datetime
import json
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from models.database_models import CitationBase, ChatSessionDB
from utils.llm_client import llm_client
from utils.embeddings import embeddings_client
from utils.vector_db import vector_db
from utils.logging import get_logger
from services.database_service import DatabaseService
from config.database import SessionLocal

logger = get_logger(__name__)

class ChatService:
    def __init__(self):
        self.llm_client = llm_client
        self.embeddings_client = embeddings_client
        self.vector_db = vector_db

    def create_session(self, selected_text: Optional[str] = None) -> str:
        """Create a new chat session in the database"""
        # Create a database session
        db_session = SessionLocal()
        db_service = DatabaseService(db_session)

        try:
            # Create session in database
            db_session_obj = db_service.create_session(metadata={"selected_text_initial": selected_text} if selected_text else None)
            session_id = str(db_session_obj.id)

            logger.info(f"Created new chat session in database: {session_id}")
            return session_id
        finally:
            db_session.close()

    def process_message(
        self,
        session_id: str,
        message: str,
        selected_text: Optional[str] = None,
        context_window_size: int = 5
    ) -> tuple[str, List[CitationBase]]:
        """Process a message and return response with citations using RAG logic"""
        try:
            logger.info(f"Processing message for session {session_id}")

            # Create database session
            db_session = SessionLocal()
            db_service = DatabaseService(db_session)

            try:
                # Store the user message in the database (T030)
                user_citations_json = []  # User message has no citations
                db_service.create_message(
                    session_id=uuid.UUID(session_id),
                    role="user",
                    content=message,
                    citations=user_citations_json,
                    model_used=None
                )

                # If selected_text is provided, use it as context (selected-text-only mode)
                if selected_text and selected_text.strip():
                    # Use only the selected text for context (implementing T040: selected-text-only RAG logic)
                    context_chunks = [{
                        "content": selected_text,
                        "metadata": {
                            "chapter": "Selected Text",
                            "section": "User Selection",
                            "page": 0,
                            "source_file": "user_selection"
                        }
                    }]

                    # Ensure responses are constrained to selected text only (T041)
                    response = self._generate_response_with_selected_text_only(
                        query=message,
                        selected_text=selected_text
                    )

                    # Create citation for the selected text
                    citations = [CitationBase(
                        source=selected_text[:200] + "..." if len(selected_text) > 200 else selected_text,
                        chapter="Selected Text",
                        section="User Selection",
                        page=0,
                        source_file="user_selection",
                        confidence=1.0
                    )]
                else:
                    # Get recent conversation history for context (T045, T046)
                    recent_messages = db_service.get_recent_messages(
                        session_id=uuid.UUID(session_id),
                        count=context_window_size
                    )

                    # Build context from recent messages
                    context_messages = []
                    for msg in reversed(recent_messages):  # Reverse to get chronological order
                        if msg.role == "user":
                            context_messages.append(f"User: {msg.content}")
                        elif msg.role == "assistant":
                            context_messages.append(f"Assistant: {msg.content}")

                    context_str = "\n".join(context_messages)

                    # Use standard RAG to retrieve relevant content from book
                    response, citations = self._perform_standard_rag_retrieval_with_context(
                        query=message,
                        conversation_context=context_str
                    )

                # Convert citations to JSON-serializable format for database storage
                citations_json = []
                for citation in citations:
                    citations_json.append({
                        "source": citation.source,
                        "chapter": citation.chapter,
                        "section": citation.section,
                        "page": citation.page,
                        "source_file": citation.source_file,
                        "confidence": citation.confidence
                    })

                # Store the assistant response in the database (T030)
                db_service.create_message(
                    session_id=uuid.UUID(session_id),
                    role="assistant",
                    content=response,
                    citations=citations_json,
                    model_used="openchat/openchat-7b:free"  # Using default model
                )

                # Update session last active time
                db_service.update_session_last_active(uuid.UUID(session_id))

                logger.info(f"Response generated and stored for session {session_id}")
                return response, citations

            finally:
                db_session.close()

        except Exception as e:
            logger.error(f"Error processing message for session {session_id}: {str(e)}")
            raise e

    def _perform_standard_rag_retrieval(self, query: str) -> tuple[str, List[CitationBase]]:
        """Perform standard RAG retrieval and generation"""
        # Generate embedding for the query
        query_embedding = self.embeddings_client.generate_embedding(query)

        # Search for similar content in the vector database
        similar_chunks = self.vector_db.search_similar(query_embedding, limit=5)

        if similar_chunks:
            # Generate response using retrieved context
            response = self.llm_client.generate_with_context(
                query=query,
                context_chunks=similar_chunks
            )

            # Create citations from the retrieved chunks
            citations = []
            for chunk in similar_chunks:
                citations.append(CitationBase(
                    source=chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                    chapter=chunk["metadata"].get("chapter", "Unknown"),
                    section=chunk["metadata"].get("section", "Unknown"),
                    page=chunk["metadata"].get("page", 0),
                    source_file=chunk["metadata"].get("source_file", "unknown"),
                    confidence=chunk["score"]
                ))
        else:
            # No relevant content found - implement fallback response (T053)
            response = "I couldn't find relevant information in the book to answer this question."
            citations = []

        return response, citations

    def _perform_standard_rag_retrieval_with_context(self, query: str, conversation_context: str = "") -> tuple[str, List[CitationBase]]:
        """Perform standard RAG retrieval and generation with conversation context"""
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings_client.generate_embedding(query)

            # Search for similar content in the vector database
            similar_chunks = self.vector_db.search_similar(query_embedding, limit=5)

            if similar_chunks:
                # Generate response using retrieved context and conversation history
                response = self.llm_client.generate_with_context_and_history(
                    query=query,
                    context_chunks=similar_chunks,
                    conversation_history=conversation_context
                )

                # Create citations from the retrieved chunks
                citations = []
                for chunk in similar_chunks:
                    citations.append(CitationBase(
                        source=chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                        chapter=chunk["metadata"].get("chapter", "Unknown"),
                        section=chunk["metadata"].get("section", "Unknown"),
                        page=chunk["metadata"].get("page", 0),
                        source_file=chunk["metadata"].get("source_file", "unknown"),
                        confidence=chunk["score"]
                    ))
            else:
                # No relevant content found - implement fallback response (T053)
                response = self._handle_no_content_found(query)
                citations = []

            return response, citations
        except Exception as e:
            logger.error(f"Error in RAG retrieval with context: {str(e)}")
            # Handle vector database unavailable gracefully (T053c)
            return self._handle_database_error(), []

    def _handle_no_content_found(self, query: str) -> str:
        """Handle cases where no relevant content is found in the book"""
        # Check for ambiguous questions that could refer to multiple book sections (T053a)
        if self._is_ambiguous_question(query):
            return ("The question you asked is ambiguous and could refer to multiple sections in the book. "
                    "Please provide more specific details about what you're looking for.")

        # Check for very long or complex questions (T053b)
        if len(query) > 500:  # arbitrary threshold for long questions
            return ("Your question is quite long. Please try to break it down into smaller, more specific questions "
                    "for better results.")

        # Standard fallback response
        return "I couldn't find relevant information in the book to answer this question."

    def _is_ambiguous_question(self, query: str) -> bool:
        """Check if a question is potentially ambiguous"""
        # Look for common ambiguous terms or phrases
        ambiguous_indicators = [
            "it", "this", "that", "they", "these", "those",  # pronouns without clear reference
            "the section", "the chapter", "the text", "the book",  # vague references
        ]

        query_lower = query.lower()
        return any(indicator in query_lower for indicator in ambiguous_indicators)

    def _handle_database_error(self) -> str:
        """Handle cases where vector database is unavailable (T053c)"""
        return ("I'm currently experiencing technical difficulties and cannot access the book content. "
                "Please try again later or contact support if the issue persists.")

    def _generate_response_with_selected_text_only(self, query: str, selected_text: str) -> str:
        """Generate response constrained to only the selected text"""
        # Create a prompt that explicitly tells the LLM to only use the provided text
        prompt = f"""
        You are an assistant for the Physical AI & Humanoid Robotics book.
        Answer the user's question based ONLY on the provided selected text.
        Do not use any other knowledge or information beyond what is in the selected text.
        If the selected text doesn't contain enough information to answer the question,
        say "The selected text doesn't contain enough information to answer this question."

        Selected text:
        {selected_text}

        User question: {query}

        Answer (based ONLY on the selected text):
        """

        return self.llm_client.generate_response(prompt)

    def get_session_history(self, session_id: str) -> List[Dict]:
        """Get chat history for a session from the database"""
        logger.info(f"Retrieving chat history for session: {session_id}")

        # Create database session
        db_session = SessionLocal()
        db_service = DatabaseService(db_session)

        try:
            # Get messages from database
            messages_db = db_service.get_session_messages(uuid.UUID(session_id))

            # Convert to the expected format
            messages = []
            for msg in messages_db:
                message_dict = {
                    "id": str(msg.id),
                    "session_id": str(msg.session_id),
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
                    "model_used": msg.model_used
                }

                # Parse citations from JSON if they exist
                if msg.citations_json:
                    try:
                        citations_data = json.loads(msg.citations_json)
                        message_dict["citations"] = citations_data
                    except json.JSONDecodeError:
                        message_dict["citations"] = []
                else:
                    message_dict["citations"] = []

                messages.append(message_dict)

            logger.info(f"Retrieved {len(messages)} messages for session {session_id}")
            return messages

        finally:
            db_session.close()