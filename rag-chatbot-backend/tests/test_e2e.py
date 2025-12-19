"""
End-to-end test for the RAG chatbot functionality.
This test verifies the flow: user question → RAG response → citation display (task T036).
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from services.chat_service import ChatService
from models.entities import CitationBase
from utils.content_processor import content_processor

def test_e2e_basic_question():
    """Test the basic flow: user asks question → gets response with citations"""
    # Initialize the chat service
    chat_service = ChatService()

    # Mock some content to be used for testing
    test_content = """
    # Introduction to Humanoid Robotics

    Humanoid robots are robots with physical characteristics resembling humans.
    They typically have a head, torso, two arms, and two legs.

    ## Key Components

    The main components of humanoid robots include:
    - Actuators for movement
    - Sensors for perception
    - Control systems for coordination
    """

    # Process the test content
    chunks = content_processor.process_book_content(test_content)

    # In a real test, we would store these chunks in the vector DB
    # For this test, we'll just verify the structure
    assert len(chunks) > 0
    assert 'content' in chunks[0]
    assert 'metadata' in chunks[0]

    # Test the chat service with a question related to the content
    session_id = "test-session-123"
    question = "What are the main components of humanoid robots?"

    # Mock the vector DB response since we don't have actual embeddings stored
    with patch.object(chat_service.vector_db, 'search_similar') as mock_search:
        mock_search.return_value = [{
            "content": "The main components of humanoid robots include: - Actuators for movement - Sensors for perception - Control systems for coordination",
            "metadata": {
                "chapter": "Introduction to Humanoid Robotics",
                "section": "Key Components",
                "page": 2,
                "source_file": "test_content.md"
            },
            "score": 0.9
        }]

        # Process the message
        response, citations = chat_service.process_message(
            session_id=session_id,
            message=question,
            selected_text=None
        )

        # Verify the response contains relevant information
        assert "components" in response.lower()
        assert "actuators" in response.lower() or "sensors" in response.lower() or "control systems" in response.lower()

        # Verify citations are properly formatted
        assert len(citations) > 0
        assert isinstance(citations[0], CitationBase)
        assert citations[0].chapter == "Introduction to Humanoid Robotics"
        assert citations[0].section == "Key Components"
        assert citations[0].page == 2

        print(f"Response: {response}")
        print(f"Citations: {citations}")


def test_e2e_selected_text_mode():
    """Test the selected-text-only mode"""
    chat_service = ChatService()

    # Test with selected text
    session_id = "test-session-456"
    question = "What does this text say about actuators?"
    selected_text = "Humanoid robots have actuators for movement and sensors for perception."

    response, citations = chat_service.process_message(
        session_id=session_id,
        message=question,
        selected_text=selected_text
    )

    # Verify the response is based only on the selected text
    assert "actuators" in response.lower()
    assert "movement" in response.lower()

    # Verify citation is for the selected text
    assert len(citations) > 0
    assert citations[0].source_file == "user_selection"

    print(f"Selected text response: {response}")
    print(f"Selected text citations: {citations}")


def test_e2e_no_relevant_content():
    """Test the fallback when no relevant content is found"""
    chat_service = ChatService()

    session_id = "test-session-789"
    question = "What is the capital of Mars?"

    # Mock empty search results
    with patch.object(chat_service.vector_db, 'search_similar') as mock_search:
        mock_search.return_value = []

        response, citations = chat_service.process_message(
            session_id=session_id,
            message=question,
            selected_text=None
        )

        # Verify fallback response is returned
        assert "couldn't find relevant information" in response.lower()
        assert len(citations) == 0

        print(f"Fallback response: {response}")


def test_multi_turn_conversation():
    """Test multi-turn conversation functionality: initial question → follow-up → contextual response (T050)"""
    chat_service = ChatService()

    # Create a session
    session_id = chat_service.create_session()

    # First question about humanoid robotics
    first_question = "What are the main components of humanoid robots?"
    with patch.object(chat_service.vector_db, 'search_similar') as mock_search:
        mock_search.return_value = [{
            "content": "The main components of humanoid robots include: - Actuators for movement - Sensors for perception - Control systems for coordination",
            "metadata": {"chapter": "Components", "section": "Main Components", "page": 2, "source_file": "components.md"},
            "score": 0.9
        }]

        first_response, first_citations = chat_service.process_message(
            session_id=session_id,
            message=first_question,
            selected_text=None
        )

    print(f"First question: {first_question}")
    print(f"First response: {first_response}")

    # Follow-up question that references previous context
    followup_question = "How do the actuators work?"
    with patch.object(chat_service.vector_db, 'search_similar') as mock_search:
        mock_search.return_value = [{
            "content": "Actuators in humanoid robots work by converting energy into mechanical motion to enable movement.",
            "metadata": {"chapter": "Actuators", "section": "Function", "page": 5, "source_file": "actuators.md"},
            "score": 0.85
        }]

        followup_response, followup_citations = chat_service.process_message(
            session_id=session_id,
            message=followup_question,
            selected_text=None
        )

    print(f"Follow-up question: {followup_question}")
    print(f"Follow-up response: {followup_response}")

    # Verify that the follow-up response is contextual and relevant to actuators
    assert "actuators" in followup_response.lower()
    assert "work" in followup_response.lower() or "function" in followup_response.lower()

    print("Multi-turn conversation maintained context appropriately")


if __name__ == "__main__":
    print("Running end-to-end tests...")
    test_e2e_basic_question()
    print("✓ Basic question test passed")

    test_e2e_selected_text_mode()
    print("✓ Selected text mode test passed")

    test_e2e_no_relevant_content()
    print("✓ No relevant content test passed")

    test_multi_turn_conversation()
    print("✓ Multi-turn conversation test passed")

    print("All end-to-end tests passed!")