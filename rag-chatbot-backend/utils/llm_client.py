import openai
import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

class OpenRouterClient:
    def __init__(self):
        # Set OpenRouter API key
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"

        # Set default model (can be overridden)
        self.default_model = "mistralai/mistral-7b-instruct:free"  # Use a free model available on OpenRouter

        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_response(self, prompt: str, model: str = None, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """Generate a response from the LLM based on the prompt"""
        if model is None:
            model = self.default_model

        data = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self.headers,
            json=data
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")

    def generate_with_context(self, query: str, context_chunks: List[Dict], model: str = None) -> str:
        """Generate a response using retrieved context chunks"""
        # Format the context
        context_text = "\n\n".join([chunk["content"] for chunk in context_chunks])

        # Create a prompt with context
        prompt = f"""
        You are an assistant for the Physical AI & Humanoid Robotics book.
        Answer the user's question based on the provided context from the book.
        If the context doesn't contain enough information to answer the question,
        say "I couldn't find relevant information in the book to answer this question."

        Context from the book:
        {context_text}

        User question: {query}

        Answer:
        """

        return self.generate_response(prompt, model)

    def generate_with_context_and_history(self, query: str, context_chunks: List[Dict], conversation_history: str = "", model: str = None) -> str:
        """Generate a response using retrieved context chunks and conversation history"""
        # Format the context
        context_text = "\n\n".join([chunk["content"] for chunk in context_chunks])

        # Create a prompt with context and conversation history
        if conversation_history.strip():
            prompt = f"""
            You are an assistant for the Physical AI & Humanoid Robotics book.
            Answer the user's question based on the provided context from the book.
            Use the conversation history to provide contextual and coherent responses.
            If the context doesn't contain enough information to answer the question,
            say "I couldn't find relevant information in the book to answer this question."

            Conversation history:
            {conversation_history}

            Context from the book:
            {context_text}

            User question: {query}

            Answer (provide contextual response based on conversation history and book context):
            """
        else:
            # Fall back to basic context if no conversation history
            prompt = f"""
            You are an assistant for the Physical AI & Humanoid Robotics book.
            Answer the user's question based on the provided context from the book.
            If the context doesn't contain enough information to answer the question,
            say "I couldn't find relevant information in the book to answer this question."

            Context from the book:
            {context_text}

            User question: {query}

            Answer:
            """

        return self.generate_response(prompt, model)

# Create a global instance
llm_client = OpenRouterClient()