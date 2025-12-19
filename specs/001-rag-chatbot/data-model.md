# Data Model: RAG Chatbot for Physical AI & Humanoid Robotics Book

## Entity: ChatSession

**Description**: Represents a user's conversation with the chatbot, including conversation history and context

**Fields**:
- `sessionId`: string (UUID) - Unique identifier for the session
- `userId`: string (optional, UUID) - Reference to user (null for anonymous sessions)
- `createdAt`: timestamp - When the session was created
- `lastActive`: timestamp - When the session was last used
- `metadata`: JSON object - Additional session data (selected text, conversation context, etc.)

**Validation Rules**:
- `sessionId` must be a valid UUID
- `createdAt` must be before or equal to `lastActive`
- `metadata` must be a valid JSON object with maximum size of 1KB

**State Transitions**:
- Active: Session has ongoing conversation
- Inactive: Session has timed out (24 hours since lastActive)
- Archived: Session has been moved to long-term storage

## Entity: ChatMessage

**Description**: An individual message in a conversation, containing user input and system response

**Fields**:
- `messageId`: string (UUID) - Unique identifier for the message
- `sessionId`: string (UUID) - Reference to the parent session
- `role`: string enum ("user" | "assistant") - Who sent the message
- `content`: string - The text content of the message
- `timestamp`: timestamp - When the message was created
- `citations`: array of Citation objects - References to book content used in response
- `modelUsed`: string - Which LLM model generated the response

**Validation Rules**:
- `messageId` must be a valid UUID
- `sessionId` must reference an existing ChatSession
- `role` must be either "user" or "assistant"
- `content` must not exceed 10,000 characters
- `citations` must be valid Citation objects if present

**Relationships**:
- Belongs to one ChatSession
- Multiple ChatMessages per ChatSession (ordered by timestamp)

## Entity: Citation

**Description**: Reference to specific book content used in a response

**Fields**:
- `source`: string - The actual text that was referenced
- `chapter`: string - Book chapter name
- `section`: string - Book section name
- `page`: number - Page number in the book
- `sourceFile`: string - Original file path of the content
- `confidence`: number (0-1) - Confidence score of the citation relevance

**Validation Rules**:
- `page` must be a positive integer
- `confidence` must be between 0 and 1
- `source` must not exceed 500 characters

## Entity: BookContentChunk

**Description**: A processed segment of book content stored in the vector database with metadata

**Fields**:
- `chunkId`: string (UUID) - Unique identifier for the content chunk
- `content`: string - The text content of the chunk
- `embedding`: array of numbers - Vector embedding of the content (1536 dimensions)
- `metadata`: object - Additional metadata about the content
  - `page`: number - Page number in the book
  - `section`: string - Section name
  - `chapter`: string - Chapter name
  - `sourceFile`: string - Original file path
  - `heading`: string - The heading that precedes this content
- `createdAt`: timestamp - When the chunk was created
- `version`: string - Version of the book content this chunk represents

**Validation Rules**:
- `chunkId` must be a valid UUID
- `embedding` must be an array of exactly 1536 numbers
- `content` must not exceed 2000 characters
- `metadata.page` must be a positive integer
- `metadata.sourceFile` must be a valid file path

**State Transitions**:
- Active: Current version of book content
- Outdated: No longer matches current book content (when book is updated)
- Archived: Removed from active search index but kept for historical purposes

## Entity: User

**Description**: An individual interacting with the chatbot, with optional account for conversation history persistence

**Fields**:
- `userId`: string (UUID) - Unique identifier for the user
- `createdAt`: timestamp - When the user account was created
- `preferences`: JSON object - User preferences for the chat experience
- `anonymous`: boolean - Whether this is an anonymous session
- `email`: string (optional) - User's email address (for registered users)

**Validation Rules**:
- `userId` must be a valid UUID
- `anonymous` and `email` are mutually exclusive (if email exists, anonymous must be false)
- `preferences` must be a valid JSON object with maximum size of 2KB

**Relationships**:
- One User to many ChatSessions
- One User to many ChatMessages through ChatSessions

## Entity: Query

**Description**: A user's question or input to the RAG system

**Fields**:
- `queryId`: string (UUID) - Unique identifier for the query
- `sessionId`: string (UUID) - Reference to the session containing this query
- `input`: string - The user's original question/input
- `processedInput`: string - The processed/normalized version of the input
- `timestamp`: timestamp - When the query was submitted
- `selectedText`: string (optional) - Text selected by the user when the query was made
- `contextWindowSize`: number - Number of previous messages included in context

**Validation Rules**:
- `queryId` must be a valid UUID
- `sessionId` must reference an existing ChatSession
- `input` must not exceed 5000 characters
- `processedInput` must not exceed 5000 characters

## Entity: Response

**Description**: The system's answer to a user's query with citations to source content

**Fields**:
- `responseId`: string (UUID) - Unique identifier for the response
- `queryId`: string (UUID) - Reference to the query this responds to
- `content`: string - The text content of the response
- `timestamp`: timestamp - When the response was generated
- `modelUsed`: string - Which LLM model generated the response
- `citations`: array of Citation objects - References to book content used
- `responseTimeMs`: number - Time taken to generate the response in milliseconds
- `confidence`: number (0-1) - Confidence score of the response

**Validation Rules**:
- `responseId` must be a valid UUID
- `queryId` must reference an existing Query
- `content` must not exceed 10,000 characters
- `citations` must be valid Citation objects if present
- `responseTimeMs` must be a positive number
- `confidence` must be between 0 and 1

**Relationships**:
- One Query to one Response (one-to-one relationship)
- Multiple Citations per Response

## Relationships Overview

```
User (1) -- (0..*) ChatSession (1) -- (0..*) ChatMessage
User (1) -- (0..*) ChatSession (1) -- (0..*) Query -- (1) Response -- (0..*) Citation
BookContentChunk (1..*) -- (0..*) Citation (1..*)
```

## Indexes

**ChatSession**:
- sessionId (primary key, unique)
- userId (index for user session lookup)
- lastActive (index for session cleanup)

**ChatMessage**:
- messageId (primary key, unique)
- sessionId + timestamp (composite index for chronological retrieval)

**BookContentChunk**:
- chunkId (primary key, unique)
- embedding (vector index for similarity search)
- metadata.chapter (index for chapter-based queries)

**Query/Response**:
- queryId/responseId (primary keys, unique)
- sessionId (index for session-based queries)