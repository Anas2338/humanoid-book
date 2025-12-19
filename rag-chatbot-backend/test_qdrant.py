import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.vector_db import vector_db

def test_qdrant():
    """Test the Qdrant client methods"""
    print("Testing Qdrant client...")

    # Check available methods
    print("Available methods on Qdrant client:")
    methods = [method for method in dir(vector_db.client) if not method.startswith('_')]
    for method in sorted(methods):
        print(f"  - {method}")

    # Check if search method exists
    if hasattr(vector_db.client, 'search'):
        print("\n'search' method exists!")
    else:
        print("\n'search' method does NOT exist!")

    # Check collection info
    try:
        info = vector_db.get_collection_info()
        print(f"\nCollection info: {info}")
    except Exception as e:
        print(f"\nError getting collection info: {str(e)}")

if __name__ == "__main__":
    test_qdrant()