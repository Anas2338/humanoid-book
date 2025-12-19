import requests
import json

# Test the chat functionality
base_url = "http://localhost:8000"

# Create a new chat session
print("Creating a new chat session...")
response = requests.post(f"{base_url}/api/chat/new")
session_data = response.json()
session_id = session_data.get("sessionId")  # Changed from session_id to sessionId
print(f"Session created with ID: {session_id}")
print(f"Response: {session_data}")

# Test a query about the book content
print("\nSending a query about Physical AI...")
query_data = {
    "message": "What is Physical AI according to the book?"
}

response = requests.post(f"{base_url}/api/chat/{session_id}/message", json=query_data)
chat_response = response.json()
print(f"Chat response: {json.dumps(chat_response, indent=2)}")

# Test another query about locomotion
print("\nSending a query about bipedal locomotion...")
query_data = {
    "message": "Explain bipedal locomotion in humanoid robots"
}

response = requests.post(f"{base_url}/api/chat/{session_id}/message", json=query_data)
chat_response = response.json()
print(f"Chat response: {json.dumps(chat_response, indent=2)}")

# Test a follow-up question to test multi-turn conversation
print("\nSending a follow-up question about ZMP...")
query_data = {
    "message": "What is Zero Moment Point (ZMP) control?"
}

response = requests.post(f"{base_url}/api/chat/{session_id}/message", json=query_data)
chat_response = response.json()
print(f"Chat response: {json.dumps(chat_response, indent=2)}")