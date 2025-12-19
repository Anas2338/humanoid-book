import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Environment variables check:")
print(f"OPENROUTER_API_KEY exists: {'YES' if os.getenv('OPENROUTER_API_KEY') else 'NO'}")
print(f"OPENROUTER_EMBEDDING_MODEL: {os.getenv('OPENROUTER_EMBEDDING_MODEL', 'NOT SET')}")
print(f"QWEN_EMBEDDING_MODEL: {os.getenv('QWEN_EMBEDDING_MODEL', 'NOT SET')}")
print(f"NEON_DATABASE_URL exists: {'YES' if os.getenv('NEON_DATABASE_URL') else 'NO'}")
print(f"QDRANT_URL exists: {'YES' if os.getenv('QDRANT_URL') else 'NO'}")
print(f"QDRANT_API_KEY exists: {'YES' if os.getenv('QDRANT_API_KEY') else 'NO'}")