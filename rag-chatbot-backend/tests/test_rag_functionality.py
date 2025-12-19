"""
Tests for RAG (Retrieval-Augmented Generation) functionality.
"""

import pytest
from unittest.mock import Mock, patch
from services.chat_service import ChatService
from utils.embeddings import embeddings_client
from utils.vector_db import vector_db

def test_embedding_generation():
    """Test that embeddings can be generated for text"""
    test_text = "This is a test sentence for embedding."

    # Test that embedding generation works
    embedding = embeddings_client.generate_embedding(test_text)

    # Verify embedding is a list of numbers with correct dimensions
    assert isinstance(embedding, list)
    assert len(embedding) == 1536  # Qwen embeddings should be 1536-dimensional
    assert all(isinstance(val, (int, float)) for val in embedding)

    print(f"Generated embedding of length {len(embedding)} for test text")


def test_vector_storage_and_retrieval():
    """Test storing and retrieving vectors from the database"""
    # This test would require actual Qdrant setup, so we'll mock it
    chat_service = ChatService()

    # Mock data for testing
    test_chunk_id = "test-chunk-123"
    test_content = "Humanoid robots use advanced control systems for coordination."
    test_embedding = [0.1] * 1536  # Mock embedding
    test_metadata = {
        "chapter": "Control Systems",
        "section": "Advanced Control",
        "page": 15,
        "source_file": "control_systems.md"
    }

    # Test storing the embedding
    with patch.object(vector_db.client, 'upsert') as mock_upsert:
        vector_db.store_embedding(
            chunk_id=test_chunk_id,
            content=test_content,
            embedding=test_embedding,
            metadata=test_metadata
        )

        # Verify upsert was called
        assert mock_upsert.called
        print("Vector storage test passed")


def test_rag_retrieval():
    """Test the RAG retrieval process"""
    chat_service = ChatService()

    # Mock search results
    mock_search_results = [{
        "content": "Humanoid robots use advanced control systems for coordination and movement.",
        "metadata": {
            "chapter": "Control Systems",
            "section": "Advanced Control",
            "page": 15,
            "source_file": "control_systems.md"
        },
        "score": 0.85
    }, {
        "content": "Sensors provide perception capabilities for humanoid robots.",
        "metadata": {
            "chapter": "Sensors",
            "section": "Perception",
            "page": 10,
            "source_file": "sensors.md"
        },
        "score": 0.78
    }]

    # Test the RAG retrieval function
    with patch.object(chat_service.vector_db, 'search_similar') as mock_search:
        mock_search.return_value = mock_search_results

        response, citations = chat_service._perform_standard_rag_retrieval("How do humanoid robots coordinate movement?")

        # Verify that results were returned
        assert len(citations) == 2
        assert any("control" in cit.chapter.lower() for cit in citations)
        assert any("sensors" in cit.chapter.lower() for cit in citations)

        print(f"RAG retrieval response: {response[:100]}...")
        print(f"Number of citations: {len(citations)}")


def test_selected_text_rag_logic():
    """Test the selected-text-only RAG logic"""
    chat_service = ChatService()

    # Test with selected text
    query = "What do actuators do?"
    selected_text = "Actuators in humanoid robots provide the mechanical power for movement and action."

    # Mock the LLM response
    with patch.object(chat_service.llm_client, 'generate_response') as mock_llm:
        mock_llm.return_value = "Actuators provide the mechanical power for movement and action in humanoid robots."

        response = chat_service._generate_response_with_selected_text_only(
            query=query,
            selected_text=selected_text
        )

        # Verify that the LLM was called with the correct prompt
        assert mock_llm.called
        args, kwargs = mock_llm.call_args
        prompt = args[0]

        # Verify the prompt contains both the selected text and the query
        assert selected_text in prompt
        assert query in prompt
        assert "ONLY" in prompt  # Check that the constraint is in the prompt

        print(f"Selected text response: {response}")


if __name__ == "__main__":
    print("Running RAG functionality tests...")

    test_embedding_generation()
    print("✓ Embedding generation test passed")

    test_vector_storage_and_retrieval()
    print("✓ Vector storage test passed")

    test_rag_retrieval()
    print("✓ RAG retrieval test passed")

    test_selected_text_rag_logic()
    print("✓ Selected text RAG logic test passed")

    print("All RAG functionality tests passed!")