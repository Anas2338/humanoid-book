import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.embeddings import embeddings_client
from utils.vector_db import vector_db
from utils.llm_client import llm_client

def test_rag_flow():
    """Test the complete RAG flow that happens in the chat service"""
    print("Testing complete RAG flow...")

    # 1. Test embedding generation (like in chat service)
    query = "What is Physical AI according to the book?"
    print(f"1. Generating embedding for query: {query}")

    try:
        query_embedding = embeddings_client.generate_embedding(query)
        print(f"   Embedding generated successfully! Length: {len(query_embedding)}")
    except Exception as e:
        print(f"   ERROR generating embedding: {str(e)}")
        return

    # 2. Test vector search (like in chat service)
    print("2. Searching for similar content in vector database...")
    try:
        similar_chunks = vector_db.search_similar(query_embedding, limit=5)
        print(f"   Found {len(similar_chunks)} similar chunks")
        if similar_chunks:
            print(f"   First chunk sample: {similar_chunks[0]['content'][:100]}...")
    except Exception as e:
        print(f"   ERROR searching vector database: {str(e)}")
        return

    # 3. Test LLM response generation (like in chat service)
    if similar_chunks:
        print("3. Generating LLM response with context...")
        try:
            response = llm_client.generate_with_context(
                query=query,
                context_chunks=similar_chunks
            )
            print(f"   LLM response generated successfully!")
            print(f"   Response sample: {response[:200]}...")
        except Exception as e:
            print(f"   ERROR generating LLM response: {str(e)}")
            return
    else:
        print("3. No similar chunks found, skipping LLM generation")

    print("\nRAG flow test completed successfully!")

if __name__ == "__main__":
    test_rag_flow()