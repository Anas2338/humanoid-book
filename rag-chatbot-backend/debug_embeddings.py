import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.embeddings import embeddings_client

def test_embeddings_client():
    """Test the embeddings client configuration"""
    print(f"Embeddings client model: {embeddings_client.model}")
    print(f"Embeddings client base URL: {embeddings_client.base_url}")

    # Try a simple embedding to see if it works
    try:
        test_text = "This is a test."
        embedding = embeddings_client.generate_embedding(test_text)
        print(f"Embedding generated successfully! Length: {len(embedding)}")
        print(f"First few values: {embedding[:5]}")
    except Exception as e:
        print(f"Embedding generation failed: {str(e)}")

if __name__ == "__main__":
    test_embeddings_client()