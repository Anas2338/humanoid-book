from sqlalchemy.orm import Session
from models.database_models import ChatSessionDB, ChatMessageDB, UserDB
from typing import List, Optional
import json
from datetime import datetime
from uuid import UUID
import uuid
import time
from sqlalchemy.exc import DisconnectionError, OperationalError

class DatabaseService:
    def __init__(self, db_session: Session):
        self.db = db_session

    def _execute_with_retry(self, operation, *args, max_retries=3, delay=0.5):
        """Execute a database operation with retry logic for connection issues"""
        for attempt in range(max_retries):
            try:
                return operation(*args)
            except (DisconnectionError, OperationalError) as e:
                if "SSL connection has been closed unexpectedly" in str(e) or "connection already closed" in str(e).lower():
                    if attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                        # Refresh the session
                        self.db.rollback()
                        continue
                    else:
                        raise e
                else:
                    raise e
            except Exception as e:
                raise e

    def create_session(self, user_id: Optional[UUID] = None, metadata: Optional[dict] = None) -> ChatSessionDB:
        """Create a new chat session in the database"""
        def _create_session_op():
            session = ChatSessionDB(
                user_id=user_id,
                metadata_json=json.dumps(metadata) if metadata else None
            )
            self.db.add(session)
            self.db.commit()
            self.db.refresh(session)
            return session

        return self._execute_with_retry(_create_session_op)

    def get_session(self, session_id: UUID) -> Optional[ChatSessionDB]:
        """Get a chat session by ID"""
        def _get_session_op():
            return self.db.query(ChatSessionDB).filter(ChatSessionDB.id == session_id).first()

        return self._execute_with_retry(_get_session_op)

    def update_session_last_active(self, session_id: UUID):
        """Update the last active timestamp for a session"""
        def _update_session_op():
            session = self.get_session(session_id)
            if session:
                session.last_active = datetime.utcnow()
                self.db.commit()

        self._execute_with_retry(_update_session_op)

    def create_message(self, session_id: UUID, role: str, content: str,
                      citations: List[dict] = None, model_used: str = None) -> ChatMessageDB:
        """Create a new chat message in the database"""
        def _create_message_op():
            message = ChatMessageDB(
                session_id=session_id,
                role=role,
                content=content,
                citations_json=json.dumps(citations) if citations else None,
                model_used=model_used
            )
            self.db.add(message)
            self.db.commit()
            self.db.refresh(message)
            return message

        return self._execute_with_retry(_create_message_op)

    def get_session_messages(self, session_id: UUID, limit: int = 50) -> List[ChatMessageDB]:
        """Get messages for a session, ordered by timestamp"""
        def _get_session_messages_op():
            return self.db.query(ChatMessageDB)\
                         .filter(ChatMessageDB.session_id == session_id)\
                         .order_by(ChatMessageDB.timestamp.asc())\
                         .limit(limit).all()

        return self._execute_with_retry(_get_session_messages_op)

    def get_recent_messages(self, session_id: UUID, count: int = 5) -> List[ChatMessageDB]:
        """Get the most recent messages for a session (for context window)"""
        def _get_recent_messages_op():
            return self.db.query(ChatMessageDB)\
                         .filter(ChatMessageDB.session_id == session_id)\
                         .order_by(ChatMessageDB.timestamp.desc())\
                         .limit(count).all()

        return self._execute_with_retry(_get_recent_messages_op)

    def create_user(self, user_id: Optional[UUID] = None, preferences: Optional[dict] = None,
                    anonymous: bool = True, email: Optional[str] = None) -> Optional[UserDB]:
        """Create a new user in the database"""
        def _create_user_op():
            if anonymous and email:
                # If email is provided, user is not anonymous
                anonymous = False
            elif not email:
                # If no email, generate a user_id if not provided
                if not user_id:
                    user_id = uuid.uuid4()

            user = UserDB(
                id=user_id or uuid.uuid4(),
                preferences_json=json.dumps(preferences) if preferences else None,
                anonymous=anonymous,
                email=email
            )
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            return user

        return self._execute_with_retry(_create_user_op)