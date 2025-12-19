import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.embeddings import QwenEmbeddingsClient
from utils.vector_db import vector_db

def test_direct_rag():
    """Test RAG directly with a new embeddings client instance"""
    print("Testing RAG with direct embeddings client...")

    # Create a new embeddings client instance (not the global one)
    embeddings_client = QwenEmbeddingsClient()

    print(f"Embeddings client model: {embeddings_client.model}")

    # Test embedding generation
    query = "What is Physical AI?"
    try:
        embedding = embeddings_client.generate_embedding(query)
        print(f"SUCCESS: Embedding generated successfully! Length: {len(embedding)}")

        # Test vector search
        similar_chunks = vector_db.search_similar(embedding, limit=3)
        print(f"SUCCESS: Found {len(similar_chunks)} similar chunks in vector database")

        if similar_chunks:
            print(f"  First chunk sample: {similar_chunks[0]['content'][:100]}...")
            print(f"  Score: {similar_chunks[0]['score']}")
        else:
            print("  No similar chunks found - this might be expected if content wasn't properly indexed")

    except Exception as e:
        print(f"ERROR: Error in RAG process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direct_rag()