"""
Basic tests for the API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

from app.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns basic info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Multi-Level Chatbot API"
        assert "version" in data
        assert "status" in data
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data


class TestChatEndpoints:
    """Test chat analysis endpoints."""
    
    @patch('app.features.chat.services.ChatOrchestrator.analyze_message')
    def test_analyze_message_success(self, mock_analyze):
        """Test successful message analysis."""
        # Mock response
        mock_analyze.return_value = {
            "action": "RESPOND",
            "risk_update": "Low", 
            "risk_score": 20,
            "sentiment": "Positive",
            "sentiment_score": 15,
            "response_to_send": "Thank you for your message.",
            "event_detection": {"has_event": False},
            "full_analysis": {}
        }
        
        # Test data
        test_data = {
            "messages": [
                {
                    "sender": "client",
                    "content": "Hello, I have a question about my case.",
                    "timestamp": "2023-01-01T10:00:00Z"
                }
            ],
            "client_info": {
                "client_id": "test_client_123",
                "name": "Test Client"
            }
        }
        
        response = client.post("/api/v1/chat/analyze", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "data" in data
    
    def test_analyze_message_empty_messages(self):
        """Test analysis with empty messages."""
        test_data = {
            "messages": [],
            "client_info": {
                "client_id": "test_client_123",
                "name": "Test Client"
            }
        }
        
        response = client.post("/api/v1/chat/analyze", json=test_data)
        assert response.status_code == 400


class TestInsightsEndpoints:
    """Test insights generation endpoints."""
    
    @patch('app.features.insights.services.MicroInsightEngine.run_micro_insight_engine')
    def test_micro_insight_success(self, mock_insight):
        """Test successful micro insight generation."""
        mock_insight.return_value = "Sentiment: Positive — Client is satisfied with recent case progress."
        
        test_data = {
            "client_id": "test_client_123",
            "client_profile": {"name": "Test Client"},
            "messages": [
                {
                    "sender": "client",
                    "content": "Thank you for the update on my case.",
                    "timestamp": "2023-01-01T10:00:00Z"
                }
            ]
        }
        
        response = client.post("/api/v1/insights/micro", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "insight" in data["data"]


class TestOutboundEndpoints:
    """Test outbound messaging endpoints."""
    
    @patch('app.features.outbound.services.OutboundMessageGenerator.generate_outbound_message')
    def test_generate_outbound_success(self, mock_generate):
        """Test successful outbound message generation."""
        mock_generate.return_value = "Hi John, I wanted to check in on your case progress. How are you feeling about things?"
        
        test_data = {
            "information": "Weekly check-in scheduled for Monday at 10:00 AM",
            "messages": [
                {
                    "sender": "client",
                    "content": "I'm concerned about the timeline.",
                    "timestamp": "2023-01-01T10:00:00Z"
                }
            ]
        }
        
        response = client.post("/api/v1/outbound/generate", json=test_data)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "message" in data["data"]


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    with patch('app.core.dependencies.get_openai_client') as mock:
        mock_client = AsyncMock()
        mock.return_value = mock_client
        yield mock_client


class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_validation_error_response(self):
        """Test validation error response format."""
        response = client.post("/api/v1/chat/analyze", json={})
        assert response.status_code == 422  # FastAPI validation error
        
    def test_health_endpoints_always_available(self):
        """Test that health endpoints are always available."""
        # These should work even if other services are down
        response = client.get("/")
        assert response.status_code == 200
        
        response = client.get("/health")
        assert response.status_code == 200
        
        response = client.get("/api/v1/chat/health")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__])
