# Quickstart Guide: RAG Chatbot for Physical AI & Humanoid Robotics Book

## Overview
This guide provides the essential steps to set up and run the RAG chatbot for the Physical AI & Humanoid Robotics book. The system uses Retrieval-Augmented Generation to provide accurate answers based on book content.

## Prerequisites
- Node.js (v18 or higher)
- Python (v3.9 or higher)
- Access to OpenRouter API
- Qdrant Cloud account (or local instance for development)
- Neon Postgres account (or local PostgreSQL for development)

## Environment Setup

### 1. Clone and Navigate to Project
```bash
git clone <repository-url>
cd humanoid-book
```

### 2. Install Frontend Dependencies
```bash
cd website  # or wherever your Docusaurus project is located
npm install
npm install @openai/chatkit-client  # Add OpenAI ChatKit
```

### 3. Install Backend Dependencies
```bash
pip install fastapi uvicorn python-dotenv qdrant-client openai psycopg2-binary
```

### 4. Set Up Environment Variables
Create a `.env` file in your backend directory with the following:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
NEON_DATABASE_URL=your_neon_postgres_connection_string
QWEN_EMBEDDING_MODEL=qwen/qwen2-7b-instruct  # or appropriate model
```

## Backend Setup

### 1. Initialize Vector Database
```bash
# Process book content and create embeddings
python scripts/process_book_content.py
```

### 2. Start Backend Service
```bash
uvicorn main:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`

## Frontend Integration

### 1. Add Chat Component to Docusaurus
The chat component uses OpenAI ChatKit and is integrated as a React component that can be added to your Docusaurus pages. The component is designed to work with the existing theme.

### 2. Configure OpenAI ChatKit
Create a ChatKit configuration file to handle the integration:
- Set up the ChatKit client with your backend API endpoint
- Configure message formatting to display citations properly
- Customize the UI to match your Docusaurus theme

### 3. Build and Serve
```bash
npm run build
npm run serve
```

## API Endpoints

### Chat Endpoints
- `POST /api/chat/new` - Create a new chat session
- `POST /api/chat/{sessionId}/message` - Send a message to a session
- `GET /api/chat/{sessionId}/history` - Get chat history

### Example API Usage
```javascript
// Create new session
const response = await fetch('/api/chat/new', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ selectedText: 'optional selected text' })
});

const { sessionId } = await response.json();

// Send message
const messageResponse = await fetch(`/api/chat/${sessionId}/message`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'Your question about the book',
    selectedText: 'optional text selected by user'
  })
});
```

## Development Workflow

### 1. Local Development
- Run backend: `uvicorn main:app --reload --port 8000`
- Run frontend: `npm run start` in the Docusaurus directory

### 2. Testing
Run the test suite to ensure all components work correctly:
```bash
# Backend tests
python -m pytest tests/

# Frontend tests
npm run test
```

## Troubleshooting

### Common Issues
1. **API Rate Limits**: Ensure your OpenRouter/Qwen API keys have sufficient quota
2. **Vector Search Performance**: If responses are slow, check your Qdrant indexing
3. **Docusaurus Integration**: Verify that the chat component CSS doesn't conflict with existing styles

### Environment Variables
Ensure all required environment variables are set before starting services.

## Next Steps
1. Customize the chat interface to match your book's design
2. Add analytics to track user interactions
3. Implement additional features like conversation export
4. Set up monitoring and logging for production deployment