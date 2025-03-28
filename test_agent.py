"""
Unit tests for the Anthropic-powered Agent
"""

import os
import sys
import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import json
import tempfile
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from anthropic_agent import Agent, Tool, ToolParameter
from anthropic_agent.memory import Memory

class TestAgent(unittest.TestCase):
    """Test cases for the Agent class."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'ANTHROPIC_API_KEY': 'mock-api-key',
        })
        self.env_patcher.start()
        
        # Create agent with mocked client
        self.agent = Agent(model="claude-3-haiku-20240307")
        self.agent.client = MagicMock()
        
        # Set up mock response
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "This is a test response."
        mock_content.type = "text"
        mock_response.content = [mock_content]
        self.agent.client.messages.create.return_value = mock_response
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
    
    def test_init(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.model, "claude-3-haiku-20240307")
        self.assertIsInstance(self.agent.memory, Memory)
        self.assertEqual(len(self.agent.tools), 0)
        self.assertEqual(len(self.agent.tool_categories), 0)
    
    def test_register_tools(self):
        """Test registering tools with the agent."""
        # Create mock tools
        test_tool1 = Tool(
            name="test_tool1",
            description="Test tool 1",
            parameters=[
                ToolParameter(name="param1", type="string", description="Parameter 1")
            ],
            function=lambda param1: {"result": param1},
            category="test"
        )
        
        test_tool2 = Tool(
            name="test_tool2",
            description="Test tool 2",
            parameters=[
                ToolParameter(name="param1", type="string", description="Parameter 1"),
                ToolParameter(name="param2", type="integer", description="Parameter 2", required=False)
            ],
            function=lambda param1, param2=None: {"result": f"{param1} {param2}"},
            category="test"
        )
        
        # Register tools
        self.agent.register_tools([test_tool1, test_tool2])
        
        # Check if tools were registered
        self.assertEqual(len(self.agent.tools), 2)
        self.assertIn("test_tool1", self.agent.tools)
        self.assertIn("test_tool2", self.agent.tools)
        
        # Check if categories were updated
        self.assertEqual(len(self.agent.tool_categories), 1)
        self.assertIn("test", self.agent.tool_categories)
        self.assertEqual(len(self.agent.tool_categories["test"]), 2)
        self.assertIn("test_tool1", self.agent.tool_categories["test"])
        self.assertIn("test_tool2", self.agent.tool_categories["test"])
    
    @patch('anthropic_agent.Agent._enrich_with_rag')
    async def test_process_message(self, mock_enrich_with_rag):
        """Test processing a message."""
        # Set up mock for RAG
        mock_enrich_with_rag.return_value = "RAG context"
        
        # Process a message
        response = await self.agent.process_message("Hello, world!")
        
        # Check that the client was called with correct arguments
        self.agent.client.messages.create.assert_called_once()
        call_kwargs = self.agent.client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "claude-3-haiku-20240307")
        self.assertEqual(len(call_kwargs["messages"]), 2)  # System message + user message
        
        # Check that the response was processed correctly
        self.assertEqual(response, "This is a test response.")
        
        # Check that the message was added to memory
        self.assertEqual(len(self.agent.memory.messages), 2)  # User message + assistant response
        self.assertEqual(self.agent.memory.messages[0]["role"], "user")
        self.assertEqual(self.agent.memory.messages[0]["content"], "Hello, world!")
        self.assertEqual(self.agent.memory.messages[1]["role"], "assistant")
        self.assertEqual(self.agent.memory.messages[1]["content"], "This is a test response.")
    
    @patch('anthropic_agent.Agent._get_tool_definitions')
    @patch('anthropic_agent.Agent._enrich_with_rag')
    async def test_process_message_with_tools(self, mock_enrich_with_rag, mock_get_tool_definitions):
        """Test processing a message with tools."""
        # Set up mocks
        mock_enrich_with_rag.return_value = "RAG context"
        mock_get_tool_definitions.return_value = [{"type": "function", "function": {"name": "test_tool"}}]
        
        # Add mock tools
        self.agent.tools = {"test_tool": MagicMock()}
        
        # Process a message
        response = await self.agent.process_message("Use the test tool")
        
        # Check that the client was called with correct arguments
        self.agent.client.messages.create.assert_called_once()
        call_kwargs = self.agent.client.messages.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "claude-3-haiku-20240307")
        self.assertIn("tools", call_kwargs)
        
        # Check that the tool definitions were included
        mock_get_tool_definitions.assert_called_once()
        self.assertEqual(call_kwargs["tools"], mock_get_tool_definitions.return_value)
    
    @patch('image_processing.ImageProcessor.encode_image_base64')
    async def test_process_message_with_image(self, mock_encode_image):
        """Test processing a message with an image."""
        # Set up mock for image encoding
        mock_image_content = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": "mock_image_data"
            }
        }
        mock_encode_image.return_value = mock_image_content
        
        # Process a message with an image
        with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_image:
            # Create a fake image file
            temp_image.write(b'fake image data')
            temp_image.flush()
            
            # Process the message with the image
            response = await self.agent.process_message_with_image("Describe this image", temp_image.name)
        
        # Check that the image processor was called
        mock_encode_image.assert_called_once()
        
        # Check that the client was called with correct arguments
        self.agent.client.messages.create.assert_called_once()
        call_kwargs = self.agent.client.messages.create.call_args[1]
        
        # Verify multimodal content was included
        self.assertIsInstance(call_kwargs["messages"][-1]["content"], list)
        self.assertEqual(len(call_kwargs["messages"][-1]["content"]), 2)
        self.assertEqual(call_kwargs["messages"][-1]["content"][0]["type"], "text")
        self.assertEqual(call_kwargs["messages"][-1]["content"][1], mock_image_content)
        
        # Check the response
        self.assertEqual(response, "This is a test response.")
    
    def test_memory_operations(self):
        """Test memory operations."""
        # Add messages to memory
        self.agent.memory.add("user", "Hello")
        self.agent.memory.add("assistant", "Hi there")
        
        # Check that messages were added
        self.assertEqual(len(self.agent.memory.messages), 2)
        self.assertEqual(self.agent.memory.messages[0]["role"], "user")
        self.assertEqual(self.agent.memory.messages[0]["content"], "Hello")
        self.assertEqual(self.agent.memory.messages[1]["role"], "assistant")
        self.assertEqual(self.agent.memory.messages[1]["content"], "Hi there")
        
        # Get history
        history = self.agent.memory.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "Hello")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], "Hi there")
        
        # Clear memory
        self.agent.memory.clear()
        self.assertEqual(len(self.agent.memory.messages), 0)
    
    @patch('anthropic_agent.Agent._execute_tool')
    async def test_tool_execution(self, mock_execute_tool):
        """Test executing a tool."""
        # Set up mock for tool execution
        mock_execute_tool.return_value = {"result": "Tool executed successfully"}
        
        # Set up tool response
        tool_response = MagicMock()
        tool_response.content = [
            MagicMock(type="tool_use", tool_use=MagicMock(name="test_tool", parameters={"param1": "value1"}))
        ]
        self.agent.client.messages.create.return_value = tool_response
        
        # Add mock tool
        self.agent.tools = {"test_tool": MagicMock()}
        
        # Process a message that should trigger tool execution
        response = await self.agent.process_message("Execute the test tool")
        
        # Check that the tool execution method was called
        mock_execute_tool.assert_called_once()
        
        # Verify arguments to the execution method
        call_args = mock_execute_tool.call_args[0]
        self.assertEqual(call_args[0], "test_tool")
        self.assertEqual(call_args[1], {"param1": "value1"})

class TestMemory(unittest.TestCase):
    """Test cases for the Memory class."""
    
    def setUp(self):
        """Set up test environment."""
        self.memory = Memory()
    
    def test_add_and_get(self):
        """Test adding messages and getting history."""
        # Add messages
        self.memory.add("user", "Hello")
        self.memory.add("assistant", "Hi there")
        self.memory.add("user", "How are you?")
        
        # Check that messages were added
        self.assertEqual(len(self.memory.messages), 3)
        
        # Get history
        history = self.memory.get_history()
        self.assertEqual(len(history), 3)
        
        # Check message content
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[0]["content"], "Hello")
        self.assertEqual(history[1]["role"], "assistant")
        self.assertEqual(history[1]["content"], "Hi there")
        self.assertEqual(history[2]["role"], "user")
        self.assertEqual(history[2]["content"], "How are you?")
    
    def test_clear(self):
        """Test clearing memory."""
        # Add messages
        self.memory.add("user", "Hello")
        self.memory.add("assistant", "Hi there")
        
        # Check that messages were added
        self.assertEqual(len(self.memory.messages), 2)
        
        # Clear memory
        self.memory.clear()
        
        # Check that memory is empty
        self.assertEqual(len(self.memory.messages), 0)
        
        # Get history after clearing
        history = self.memory.get_history()
        self.assertEqual(len(history), 0)
    
    def test_save_and_load(self):
        """Test saving and loading memory."""
        # Add messages
        self.memory.add("user", "Hello")
        self.memory.add("assistant", "Hi there")
        
        # Save memory to a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_path = temp_file.name
            self.memory.save(temp_path)
        
        # Create a new memory instance and load the saved memory
        new_memory = Memory()
        new_memory.load(temp_path)
        
        # Check that the loaded memory matches the original
        self.assertEqual(len(new_memory.messages), 2)
        self.assertEqual(new_memory.messages[0]["role"], "user")
        self.assertEqual(new_memory.messages[0]["content"], "Hello")
        self.assertEqual(new_memory.messages[1]["role"], "assistant")
        self.assertEqual(new_memory.messages[1]["content"], "Hi there")
        
        # Clean up
        os.unlink(temp_path)