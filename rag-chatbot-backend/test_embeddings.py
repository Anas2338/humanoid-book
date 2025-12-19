import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")
base_url = "https://openrouter.ai/api/v1"

# Headers for API requests
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Test with a known working embedding model
model = "text-embedding-ada-002"  # This is the OpenAI model that should work

# Test text
test_text = "This is a test sentence for embedding."

data = {
    "model": model,
    "input": test_text
}

response = requests.post(
    f"{base_url}/embeddings",
    headers=headers,
    json=data
)

print(f"Response status: {response.status_code}")
print(f"Response text: {response.text}")

if response.status_code == 200:
    result = response.json()
    print(f"Success! Embedding dimension: {len(result['data'][0]['embedding'])}")
else:
    print("Failed to get embedding. Let's try to list available models...")

    # Try to list models
    models_response = requests.get(
        f"{base_url}/models",
        headers=headers
    )

    print(f"Models response status: {models_response.status_code}")
    if models_response.status_code == 200:
        models_data = models_response.json()
        print("Available embedding models:")
        for model_info in models_data.get('data', []):
            if 'embedding' in model_info.get('description', '').lower() or 'embed' in model_info.get('id', '').lower():
                print(f"  - {model_info['id']}")
    else:
        print(f"Models response text: {models_response.text}")