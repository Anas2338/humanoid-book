from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import chat  # Import the chat router
from utils.logging import get_logger

app = FastAPI(
    title="RAG Chatbot API",
    description="""
    # RAG Chatbot API for Physical AI & Humanoid Robotics Book

    This API provides a Retrieval-Augmented Generation (RAG) chatbot that can answer questions about the Physical AI & Humanoid Robotics book content with proper citations.

    ## Features
    - Ask questions about book content
    - Get responses with citations to specific book sections
    - Support for user-selected text queries
    - Multi-turn conversation context

    ## Endpoints
    - `POST /api/chat/new` - Create a new chat session
    - `POST /api/chat/{sessionId}/message` - Send a message to a session
    - `GET /api/chat/{sessionId}/history` - Get chat history

    For more information about the RAG system architecture, check our [documentation](https://example.com/docs).
    """,
    version="1.0.0",
    contact={
        "name": "RAG Chatbot Team",
        "url": "https://example.com/contact",
        "email": "support@example.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(chat.router, prefix="/api", tags=["chat"])

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": __import__('datetime').datetime.utcnow()}

# Initialize logger
logger = get_logger(__name__)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)