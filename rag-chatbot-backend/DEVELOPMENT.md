# Development Setup Instructions

## Prerequisites

- Python 3.9 or higher
- Node.js 18 or higher
- Access to OpenRouter API
- Qdrant Cloud account
- Neon Postgres account

## Environment Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your API keys and connection strings

## Running the Application

1. Start the backend:
   ```bash
   cd rag-chatbot-backend
   uvicorn main:app --reload --port 8000
   ```

2. The backend will be available at `http://localhost:8000`

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Frontend Integration

For Docusaurus integration, see the rag-chatbot-frontend directory.