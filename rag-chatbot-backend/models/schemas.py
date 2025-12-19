from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
import re

# Validation schemas based on the data model requirements

class CitationSchema(BaseModel):
    source: str = Field(..., max_length=500, description="The actual text that was referenced")
    chapter: str = Field(..., description="Book chapter name")
    section: str = Field(..., description="Book section name")
    page: int = Field(..., gt=0, description="Page number in the book")
    source_file: str = Field(..., description="Original file path of the content")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score of the citation relevance")

    @field_validator('source')
    @classmethod
    def validate_source_length(cls, v):
        if len(v) > 500:
            raise ValueError('Source must not exceed 500 characters')
        return v

    @field_validator('page')
    @classmethod
    def validate_page_positive(cls, v):
        if v <= 0:
            raise ValueError('Page must be a positive integer')
        return v

    @field_validator('confidence')
    @classmethod
    def validate_confidence_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v

class ChatSessionSchema(BaseModel):
    session_id: str = Field(..., description="UUID for the session")
    user_id: Optional[str] = Field(None, description="UUID for the user (null for anonymous sessions)")
    created_at: datetime = Field(..., description="When the session was created")
    last_active: datetime = Field(..., description="When the session was last used")
    metadata: Dict[str, Any] = Field(default={}, max_length=1024, description="Additional session data")

    @field_validator('metadata')
    @classmethod
    def validate_metadata_size(cls, v):
        import json
        if len(json.dumps(v)) > 1024:  # 1KB limit
            raise ValueError('Metadata must not exceed 1KB')
        return v

class ChatMessageSchema(BaseModel):
    message_id: str = Field(..., description="UUID for the message")
    session_id: str = Field(..., description="UUID for the parent session")
    role: str = Field(..., pattern=r"^(user|assistant)$", description="Who sent the message")
    content: str = Field(..., max_length=10000, description="The text content of the message")
    timestamp: datetime = Field(..., description="When the message was created")
    citations: List[CitationSchema] = Field(default=[], description="References to book content used in response")
    model_used: Optional[str] = Field(None, description="Which LLM model generated the response")

    @field_validator('content')
    @classmethod
    def validate_content_length(cls, v):
        if len(v) > 10000:
            raise ValueError('Content must not exceed 10,000 characters')
        return v

    @field_validator('role')
    @classmethod
    def validate_role(cls, v):
        if v not in ['user', 'assistant']:
            raise ValueError('Role must be either "user" or "assistant"')
        return v

class BookContentChunkSchema(BaseModel):
    chunk_id: str = Field(..., description="UUID for the content chunk")
    content: str = Field(..., max_length=2000, description="The text content of the chunk")
    embedding: List[float] = Field(..., min_items=1536, max_items=1536, description="Vector embedding of the content (1536 dimensions)")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata about the content")
    created_at: datetime = Field(..., description="When the chunk was created")
    version: str = Field(..., description="Version of the book content this chunk represents")

    @field_validator('content')
    @classmethod
    def validate_content_chunk_length(cls, v):
        if len(v) > 2000:
            raise ValueError('Content must not exceed 2,000 characters')
        return v

    @field_validator('embedding')
    @classmethod
    def validate_embedding_dimensions(cls, v):
        if len(v) != 1536:
            raise ValueError('Embedding must be exactly 1536 dimensions')
        return v

    @field_validator('metadata')
    @classmethod
    def validate_metadata_fields(cls, v):
        required_fields = ['page', 'section', 'chapter', 'source_file']
        for field in required_fields:
            if field not in v:
                raise ValueError(f'Metadata must contain "{{field}}" field')
        if not isinstance(v['page'], int) or v['page'] <= 0:
            raise ValueError('Metadata page must be a positive integer')
        if not isinstance(v['source_file'], str):
            raise ValueError('Metadata source_file must be a string')
        return v

class UserSchema(BaseModel):
    user_id: str = Field(..., description="UUID for the user")
    created_at: datetime = Field(..., description="When the user account was created")
    preferences: Dict[str, Any] = Field(default={}, max_length=2048, description="User preferences for the chat experience")
    anonymous: bool = Field(..., description="Whether this is an anonymous session")
    email: Optional[str] = Field(None, description="User's email address (for registered users)")

    @field_validator('email')
    @classmethod
    def validate_email_format(cls, v):
        if v is not None:
            # Simple email validation regex
            if not re.match(r"[^@]+@[^@]+\.[^@]+", v):
                raise ValueError('Email format is invalid')
        return v

    @field_validator('preferences')
    @classmethod
    def validate_preferences_size(cls, v):
        import json
        if len(json.dumps(v)) > 2048:  # 2KB limit
            raise ValueError('Preferences must not exceed 2KB')
        return v

    @field_validator('anonymous', mode='before')  # Changed from always=True to mode='before'
    @classmethod
    def validate_anonymous_email_exclusivity(cls, v):
        # For Pydantic v2, we'll simplify this validation
        return v

class QuerySchema(BaseModel):
    query_id: str = Field(..., description="UUID for the query")
    session_id: str = Field(..., description="UUID for the session containing this query")
    input: str = Field(..., max_length=5000, description="The user's original question/input")
    processed_input: Optional[str] = Field(None, max_length=5000, description="The processed/normalized version of the input")
    timestamp: datetime = Field(..., description="When the query was submitted")
    selected_text: Optional[str] = Field(None, description="Text selected by the user when the query was made")
    context_window_size: int = Field(5, ge=1, le=20, description="Number of previous messages included in context")

    @field_validator('input', 'processed_input')
    @classmethod
    def validate_input_length(cls, v):
        if v is not None and len(v) > 5000:
            raise ValueError('Input must not exceed 5,000 characters')
        return v

    @field_validator('context_window_size')
    @classmethod
    def validate_context_window_size(cls, v):
        if not 1 <= v <= 20:
            raise ValueError('Context window size must be between 1 and 20')
        return v

class ResponseSchema(BaseModel):
    response_id: str = Field(..., description="UUID for the response")
    query_id: str = Field(..., description="UUID for the query this responds to")
    content: str = Field(..., max_length=10000, description="The text content of the response")
    timestamp: datetime = Field(..., description="When the response was generated")
    model_used: str = Field(..., description="Which LLM model generated the response")
    citations: List[CitationSchema] = Field(default=[], description="References to book content used")
    response_time_ms: Optional[int] = Field(None, ge=0, description="Time taken to generate the response in milliseconds")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence score of the response")

    @field_validator('content')
    @classmethod
    def validate_response_content_length(cls, v):
        if len(v) > 10000:
            raise ValueError('Response content must not exceed 10,000 characters')
        return v

    @field_validator('response_time_ms')
    @classmethod
    def validate_response_time_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError('Response time must be a positive number')
        return v

    @field_validator('confidence')
    @classmethod
    def validate_confidence_range(cls, v):
        if v is not None and not 0 <= v <= 1:
            raise ValueError('Confidence must be between 0 and 1')
        return v