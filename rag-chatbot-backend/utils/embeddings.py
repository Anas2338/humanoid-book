import openai
import os
from typing import List
from dotenv import load_dotenv
import requests
import numpy as np

# Load environment variables
load_dotenv()

class QwenEmbeddingsClient:
    def __init__(self):
        # Set OpenRouter API key (Qwen embeddings accessed through OpenRouter)
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.model = os.getenv("OPENROUTER_EMBEDDING_MODEL", "text-embedding-ada-002")

        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a single text"""
        data = {
            "model": self.model,
            "input": text
        }

        response = requests.post(
            f"{self.base_url}/embeddings",
            headers=self.headers,
            json=data
        )

        if response.status_code == 200:
            result = response.json()
            # Extract the embedding from the response
            embedding = result["data"][0]["embedding"]
            return embedding
        else:
            raise Exception(f"Embedding API request failed: {response.status_code} - {response.text}")

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        embeddings = []
        for text in texts:
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        # Convert to numpy arrays
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        return dot_product / (norm_v1 * norm_v2)

# Create a global instance
embeddings_client = QwenEmbeddingsClient()