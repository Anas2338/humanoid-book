# RAG Chatbot Backend

This is the backend service for the Retrieval-Augmented Generation (RAG) chatbot for the Physical AI & Humanoid Robotics book.

## Overview

The backend provides:
- FastAPI endpoints for chat functionality
- Integration with Qdrant vector database for book content retrieval
- Connection to OpenRouter for LLM responses
- Session management for conversations
- Citation formatting for book references

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   - Copy `.env.example` to `.env` and fill in your API keys

3. Run the application:
   ```bash
   uvicorn main:app --reload
   ```

## API Endpoints

- `GET /` - Health check
- `POST /api/chat/new` - Create new chat session
- `POST /api/chat/{sessionId}/message` - Send message to chat session
- `GET /api/chat/{sessionId}/history` - Get chat history

## Architecture

The backend follows a multi-tier architecture with:
- FastAPI for web framework
- Qdrant Cloud for vector storage
- Neon Postgres for metadata
- OpenRouter for LLM services