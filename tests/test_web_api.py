"""
Unit tests for the web API
"""

import os
import sys
import unittest
import asyncio
import tempfile
import json
from unittest.mock import MagicMock, patch

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from fastapi import FastAPI
import web_api

class TestWebAPI(unittest.TestCase):
    """Test cases for the Web API."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'ANTHROPIC_API_KEY': 'mock-api-key',
            'WEB_API_KEY': 'test-api-key',
        })
        self.env_patcher.start()
        
        # Mock agent
        self.mock_agent = MagicMock()
        self.mock_agent.process_message.return_value = "Mock response"
        self.mock_agent.process_message_with_image.return_value = "Mock image response"
        self.mock_agent.memory = MagicMock()
        self.mock_agent.tools = {}
        self.mock_agent.tool_categories = {}
        
        # Patch agent creation
        self.get_agent_patcher = patch('web_api.get_agent', return_value=self.mock_agent)
        self.get_agent_patcher.start()
        
        # Create test client
        self.client = TestClient(web_api.app)
        
        # Add test headers
        self.headers = {"X-API-Key": "test-api-key"}
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
        self.get_agent_patcher.stop()
    
    def test_root(self):
        """Test root endpoint."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["name"], "Anthropic-Powered Agent API")
    
    def test_health(self):
        """Test health endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "healthy")
    
    def test_auth_failure(self):
        """Test authentication failure."""
        response = self.client.get("/models", headers={"X-API-Key": "wrong-key"})
        self.assertEqual(response.status_code, 401)
    
    def test_send_message(self):
        """Test sending a message."""
        # Set up test data
        test_data = {
            "message": "Test message",
            "stream": False
        }
        
        # Send request
        response = self.client.post("/message", json=test_data, headers=self.headers)
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["response"], "Mock response")
        
        # Verify agent was called correctly
        self.mock_agent.process_message.assert_called_once_with("Test message")
    
    def test_conversations(self):
        """Test conversation endpoints."""
        # Create a test conversation
        web_api.conversations = {}
        test_data = {
            "message": "Test message",
            "stream": False
        }
        response = self.client.post("/message", json=test_data, headers=self.headers)
        conversation_id = response.json()["conversation_id"]
        
        # Test listing conversations
        list_response = self.client.get("/conversations", headers=self.headers)
        self.assertEqual(list_response.status_code, 200)
        self.assertEqual(len(list_response.json()["conversations"]), 1)
        
        # Test getting a conversation
        get_response = self.client.get(f"/conversations/{conversation_id}", headers=self.headers)
        self.assertEqual(get_response.status_code, 200)
        self.assertEqual(get_response.json()["id"], conversation_id)
        
        # Test deleting a conversation
        delete_response = self.client.delete(f"/conversations/{conversation_id}", headers=self.headers)
        self.assertEqual(delete_response.status_code, 200)
        
        # Verify conversation was deleted
        list_response = self.client.get("/conversations", headers=self.headers)
        self.assertEqual(len(list_response.json()["conversations"]), 0)
