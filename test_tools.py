"""
Unit tests for tool functionality
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

from anthropic_agent import Tool, ToolParameter
from github_tools import get_github_tools
from claude_tools import get_claude_tools
from system_tools import get_system_tools
from anthropic_cookbook import get_cookbook_tools

class TestTools(unittest.TestCase):
    """Test cases for the Tool class and tool definitions."""
    
    def test_tool_creation(self):
        """Test creating a tool."""
        # Create a simple tool
        def test_function(param1, param2=None):
            return {"param1": param1, "param2": param2}
        
        tool = Tool(
            name="test_tool",
            description="Test tool description",
            parameters=[
                ToolParameter(name="param1", type="string", description="First parameter"),
                ToolParameter(name="param2", type="string", description="Second parameter", required=False)
            ],
            function=test_function,
            category="test"
        )
        
        # Check tool properties
        self.assertEqual(tool.name, "test_tool")
        self.assertEqual(tool.description, "Test tool description")
        self.assertEqual(len(tool.parameters), 2)
        self.assertEqual(tool.parameters[0].name, "param1")
        self.assertEqual(tool.parameters[0].type, "string")
        self.assertEqual(tool.parameters[0].description, "First parameter")
        self.assertEqual(tool.parameters[0].required, True)
        self.assertEqual(tool.parameters[1].name, "param2")
        self.assertEqual(tool.parameters[1].type, "string")
        self.assertEqual(tool.parameters[1].description, "Second parameter")
        self.assertEqual(tool.parameters[1].required, False)
        self.assertEqual(tool.category, "test")
        
        # Test function execution
        result = tool.function("value1", "value2")
        self.assertEqual(result, {"param1": "value1", "param2": "value2"})
    
    def test_tool_parameter_validation(self):
        """Test parameter validation."""
        # Create parameters with different types
        string_param = ToolParameter(name="string_param", type="string", description="String parameter")
        int_param = ToolParameter(name="int_param", type="integer", description="Integer parameter")
        bool_param = ToolParameter(name="bool_param", type="boolean", description="Boolean parameter")
        float_param = ToolParameter(name="float_param", type="number", description="Number parameter")
        array_param = ToolParameter(name="array_param", type="array", description="Array parameter")
        object_param = ToolParameter(name="object_param", type="object", description="Object parameter")
        
        # Check parameter properties
        self.assertEqual(string_param.name, "string_param")
        self.assertEqual(string_param.type, "string")
        self.assertEqual(string_param.description, "String parameter")
        self.assertEqual(string_param.required, True)
        
        # Check default values
        default_param = ToolParameter(name="default_param", type="string", 
                                     description="Parameter with default", 
                                     required=False, default="default")
        self.assertEqual(default_param.default, "default")
    
    def test_github_tools(self):
        """Test GitHub tools."""
        # Get GitHub tools
        github_tools = get_github_tools()
        
        # Check that tools were returned
        self.assertGreater(len(github_tools), 0)
        
        # Check that each tool has required attributes
        for tool in github_tools:
            self.assertIsInstance(tool, Tool)
            self.assertIsNotNone(tool.name)
            self.assertIsNotNone(tool.description)
            self.assertIsNotNone(tool.parameters)
            self.assertIsNotNone(tool.function)
            self.assertEqual(tool.category, "github")
    
    def test_claude_tools(self):
        """Test Claude tools."""
        # Get Claude tools
        claude_tools = get_claude_tools()
        
        # Check that tools were returned
        self.assertGreater(len(claude_tools), 0)
        
        # Check that each tool has required attributes
        for tool in claude_tools:
            self.assertIsInstance(tool, Tool)
            self.assertIsNotNone(tool.name)
            self.assertIsNotNone(tool.description)
            self.assertIsNotNone(tool.parameters)
            self.assertIsNotNone(tool.function)
            self.assertEqual(tool.category, "claude")
    
    def test_system_tools(self):
        """Test system tools."""
        # Get system tools
        system_tools = get_system_tools()
        
        # Check that tools were returned
        self.assertGreater(len(system_tools), 0)
        
        # Check that each tool has required attributes
        for tool in system_tools:
            self.assertIsInstance(tool, Tool)
            self.assertIsNotNone(tool.name)
            self.assertIsNotNone(tool.description)
            self.assertIsNotNone(tool.parameters)
            self.assertIsNotNone(tool.function)
            self.assertEqual(tool.category, "system")
    
    def test_cookbook_tools(self):
        """Test cookbook tools."""
        # Get cookbook tools
        cookbook_tools = get_cookbook_tools()
        
        # Check that tools were returned
        self.assertGreater(len(cookbook_tools), 0)
        
        # Check that each tool has required attributes
        for tool in cookbook_tools:
            self.assertIsInstance(tool, Tool)
            self.assertIsNotNone(tool.name)
            self.assertIsNotNone(tool.description)
            self.assertIsNotNone(tool.parameters)
            self.assertIsNotNone(tool.function)
            self.assertEqual(tool.category, "cookbook")