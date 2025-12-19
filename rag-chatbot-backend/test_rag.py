#!/usr/bin/env python3
"""
Test script to verify RAG functionality directly
"""

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

def test_rag_functionality():
    """Test RAG functionality step by step"""
    print("Testing RAG functionality...")

    # Create a chat service instance (like in the API)
    chat_service = ChatService()

    # Test the same query as in the failing case
    query = "What is Physical AI according to the book?"

    try:
        print(f"1. Testing embedding generation for query: '{query}'")
        # Test embedding generation (like in chat_service)
        query_embedding = chat_service.embeddings_client.generate_embedding(query)
        print(f"   SUCCESS: Query embedding generated successfully! Length: {len(query_embedding)}")

        print("2. Testing vector search in database...")
        # Test vector search (like in chat_service)
        similar_chunks = chat_service.vector_db.search_similar(query_embedding, limit=5)
        print(f"   SUCCESS: Found {len(similar_chunks)} similar chunks in vector database")

        if similar_chunks:
            print("   First few chunks:")
            for i, chunk in enumerate(similar_chunks[:3]):  # Show first 3 chunks
                print(f"     Chunk {i+1}: {chunk['content'][:100]}...")
                print(f"     Score: {chunk['score']}")
                print(f"     Metadata: {chunk['metadata']}")
        else:
            print("   No similar chunks found - this might be the issue!")

        print("3. Testing end-to-end RAG retrieval...")
        # Test end-to-end RAG retrieval
        response, citations = chat_service._perform_standard_rag_retrieval_with_context(query, "")
        print(f"   SUCCESS: RAG retrieval completed!")
        print(f"   Response: {response[:200]}...")
        print(f"   Citations: {len(citations)}")

        if citations:
            print("   First citation:")
            print(f"     Source: {citations[0].source[:100]}...")
            print(f"     Chapter: {citations[0].chapter}")
            print(f"     Section: {citations[0].section}")
            print(f"     Confidence: {citations[0].confidence}")

    except Exception as e:
        print(f"ERROR: Error in RAG process: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rag_functionality()