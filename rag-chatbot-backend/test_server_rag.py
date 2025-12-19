import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the exact same modules as used in the server
from utils.embeddings import embeddings_client
from utils.vector_db import vector_db
from services.chat_service import ChatService

def test_server_rag():
    """Test RAG using the same instance as the server"""
    print("Testing RAG with server's embeddings client instance...")

    print(f"Server embeddings client model: {embeddings_client.model}")

    # Create a chat service instance (like in the API)
    chat_service = ChatService()

    # Test the same query as in the failing case
    query = "What is Physical AI according to the book?"

    try:
        # Test embedding generation (like in chat_service)
        query_embedding = chat_service.embeddings_client.generate_embedding(query)
        print(f"SUCCESS: Query embedding generated successfully! Length: {len(query_embedding)}")

        # Test vector search (like in chat_service)
        similar_chunks = chat_service.vector_db.search_similar(query_embedding, limit=5)
        print(f"SUCCESS: Found {len(similar_chunks)} similar chunks in vector database")

        if similar_chunks:
            print(f"  First chunk sample: {similar_chunks[0]['content'][:100]}...")
            print(f"  Score: {similar_chunks[0]['score']}")
        else:
            print("  No similar chunks found")

        # Test end-to-end RAG retrieval
        response, citations = chat_service._perform_standard_rag_retrieval_with_context(query, "")
        print(f"SUCCESS: RAG retrieval successful!")
        print(f"  Response: {response[:200]}...")
        print(f"  Citations: {len(citations)}")

    except Exception as e:
        print(f"ERROR: Error in server RAG process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_server_rag()