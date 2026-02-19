import os
import json
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from fastapi.testclient import TestClient

# Set environment variables BEFORE importing app to bypass validation
os.environ["OPENAI_API_KEY"] = "mock_key"
os.environ["GEMINI_API_KEY"] = "mock_key"

# Now import app
from app.main import app

def test_agent_sync():
    # Patch the agent_app.ainvoke to return a mock response
    # We need to use AsyncMock if the function is async, but ainvoke is async.
    # However, since we are patching it, we can make it return an awaitable or just use a sync mock if the caller handles it?
    # In `app/features/agent/routes.py`, `await agent_app.ainvoke(initial_state)` is called.
    # So the return value of the mock must be awaitable.
    
    async def mock_ainvoke(*args, **kwargs):
        return {
            "messages": [AIMessage(content="I am a mocked agent response.")]
        }

    with patch("app.features.agent.routes.agent_app") as mock_app:
        mock_app.ainvoke.side_effect = mock_ainvoke
        client = TestClient(app)
        
        # Test 1: Simple chat
        print("Test 1: Simple chat")
        response = client.post("/api/v1/agent/invoke", json={
            "messages": [
                {"role": "user", "content": "Hello, how are you?"}
            ]
        })
        print(response.json())
        assert response.status_code == 200
        assert response.json()["response"] == "I am a mocked agent response."

        # Test 2: Tool call (Mocked)
        print("\nTest 2: Analyze Chat Tool (Mocked)")
        response = client.post("/api/v1/agent/invoke", json={
            "messages": [
                {"role": "user", "content": "Analyze this message..."}
            ],
            "context": {"client_id": "123"}
        })
        print(response.json())
        assert response.status_code == 200

if __name__ == "__main__":
    test_agent_sync()
