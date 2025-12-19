from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, Float, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from config.database import Base
import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any

# SQLAlchemy Models (for database)
class ChatSessionDB(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=True)  # Optional user reference
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(Text)  # JSON string for additional metadata

    # Relationship to messages
    messages = relationship("ChatMessageDB", back_populates="session", order_by="ChatMessageDB.timestamp")

class ChatMessageDB(Base):
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id"))
    role = Column(String(20))  # "user" or "assistant"
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_used = Column(String(100))  # Which LLM model was used
    citations_json = Column(Text)  # JSON string for citations

    # Relationship to session
    session = relationship("ChatSessionDB", back_populates="messages")

class UserDB(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow)
    preferences_json = Column(Text)  # JSON string for user preferences
    anonymous = Column(Boolean, default=True)
    email = Column(String(255), nullable=True, unique=True)  # Only for registered users

# Pydantic Models (for API)
class CitationBase(BaseModel):
    source: str
    chapter: str
    section: str
    page: int
    source_file: str
    confidence: float

class ChatSessionBase(BaseModel):
    id: str
    user_id: Optional[str] = None
    created_at: datetime
    last_active: datetime
    metadata: Optional[Dict[str, Any]] = {}

    model_config = {'from_attributes': True}

class ChatMessageBase(BaseModel):
    id: str
    session_id: str
    role: str
    content: str
    timestamp: datetime
    citations: List[CitationBase] = []
    model_used: Optional[str] = None

    model_config = {'from_attributes': True}

class UserBase(BaseModel):
    id: str
    created_at: datetime
    preferences: Optional[Dict[str, Any]] = {}
    anonymous: bool
    email: Optional[str] = None

    model_config = {'from_attributes': True}

# Request/Response Models
class ChatSessionCreate(BaseModel):
    selected_text: Optional[str] = None

class ChatMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    selected_text: Optional[str] = Field(None, max_length=10000)

    @field_validator('message')
    @classmethod
    def validate_message_content(cls, v):
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        if len(v) > 5000:
            raise ValueError('Message too long, must be less than 5000 characters')
        return v.strip()

class ChatMessageResponse(BaseModel):
    response: str
    citations: List[CitationBase]
    session_id: str

class QueryBase(BaseModel):
    id: str
    session_id: str
    input: str
    processed_input: Optional[str] = None
    timestamp: datetime
    selected_text: Optional[str] = None
    context_window_size: Optional[int] = 5

class ResponseBase(BaseModel):
    id: str
    query_id: str
    content: str
    timestamp: datetime
    model_used: str
    citations: List[CitationBase] = []
    response_time_ms: Optional[int] = None
    confidence: Optional[float] = None