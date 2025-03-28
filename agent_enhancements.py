"""
Enhancements for the Anthropic-powered Agent
Integrates security and performance features
"""

import logging
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from functools import wraps
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import security and performance modules
from security import InputValidator, validate_input, rate_limit, global_permission_system, permission_required
from performance import cached, global_response_cache, global_parallel_executor

# Create enhanced versions of agent methods with security and performance features
def enhance_agent(agent):
    """Add security and performance enhancements to an agent."""
    
    # Store original methods
    original_process_message = agent.process_message
    original_stream_response = agent.stream_response
    original_process_message_with_image = agent.process_message_with_image
    original_get_structured_response = agent.get_structured_response
    
    # Enhance process_message with caching, rate limiting, and input validation
    @cached(ttl=60)  # Cache responses for 60 seconds
    @rate_limit(tokens=1, limit_by="user")  # Apply rate limiting by user
    @validate_input(lambda user_message, user_id=None, **kwargs: (
        len(user_message) <= 10000,  # Validate input length
        "Input message too long. Maximum length is 10000 characters."
    ))
    async def enhanced_process_message(user_message: str, user_id: Optional[str] = None, **kwargs) -> str:
        """Enhanced version of process_message with security and caching."""
        # Sanitize input
        sanitized_message = InputValidator.sanitize_string(user_message)
        
        # Check permissions if user_id provided
        if user_id and not global_permission_system.check_permission(user_id, "agent.process_message", "r"):
            return "Permission denied: You don't have access to this functionality."
        
        # Call original method
        return await original_process_message(sanitized_message)
    
    # Enhance stream_response
    @rate_limit(tokens=1, limit_by="user")  # Apply rate limiting by user
    @validate_input(lambda user_message, user_id=None, **kwargs: (
        len(user_message) <= 10000,  # Validate input length
        "Input message too long. Maximum length is 10000 characters."
    ))
    async def enhanced_stream_response(user_message: str, user_id: Optional[str] = None, **kwargs):
        """Enhanced version of stream_response with security features."""
        # Sanitize input
        sanitized_message = InputValidator.sanitize_string(user_message)
        
        # Check permissions if user_id provided
        if user_id and not global_permission_system.check_permission(user_id, "agent.stream_response", "r"):
            yield "Permission denied: You don't have access to this functionality."
            return
        
        # Call original method
        async for chunk in original_stream_response(sanitized_message):
            yield chunk
    
    # Enhance process_message_with_image
    @rate_limit(tokens=2, limit_by="user")  # Higher token cost for image processing
    @validate_input(lambda user_message, image_path, user_id=None, **kwargs: (
        len(user_message) <= 10000 and InputValidator.validate_path(image_path),
        "Invalid input: Message too long or invalid image path."
    ))
    async def enhanced_process_message_with_image(user_message: str, image_path: str, user_id: Optional[str] = None, **kwargs) -> str:
        """Enhanced version of process_message_with_image with security features."""
        # Sanitize input
        sanitized_message = InputValidator.sanitize_string(user_message)
        
        # Check permissions if user_id provided
        if user_id and not global_permission_system.check_permission(user_id, "agent.process_message_with_image", "r"):
            return "Permission denied: You don't have access to this functionality."
        
        # Call original method
        return await original_process_message_with_image(sanitized_message, image_path)
    
    # Enhance get_structured_response
    @cached(ttl=300)  # Cache structured responses for 5 minutes
    @rate_limit(tokens=1, limit_by="user")
    @validate_input(lambda user_message, schema, user_id=None, **kwargs: (
        len(user_message) <= 10000,
        "Input message too long. Maximum length is 10000 characters."
    ))
    async def enhanced_get_structured_response(user_message: str, schema, user_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Enhanced version of get_structured_response with security and caching."""
        # Sanitize input
        sanitized_message = InputValidator.sanitize_string(user_message)
        
        # Check permissions if user_id provided
        if user_id and not global_permission_system.check_permission(user_id, "agent.get_structured_response", "r"):
            return {"error": "Permission denied: You don't have access to this functionality."}
        
        # Call original method
        return await original_get_structured_response(sanitized_message, schema)
    
    # Add a new method for parallel tool execution
    async def execute_tools_in_parallel(tools_to_execute: List[Dict[str, Any]], user_id: Optional[str] = None) -> List[Any]:
        """
        Execute multiple tools in parallel.
        
        Args:
            tools_to_execute: List of tool definitions
            user_id: Optional user identifier for permission checking
            
        Returns:
            List of results for each tool
        """
        # Check permissions for each tool
        if user_id:
            for tool in tools_to_execute:
                tool_name = tool.get("name", "unknown")
                if not global_permission_system.check_permission(user_id, f"tool.{tool_name}", "r"):
                    return [{"error": f"Permission denied for tool: {tool_name}"}]
        
        # Prepare functions for parallel execution
        functions = []
        
        for tool in tools_to_execute:
            tool_name = tool.get("name", "unknown")
            
            if tool_name in agent.tools:
                tool_obj = agent.tools[tool_name]
                args = tool.get("args", [])
                kwargs = tool.get("kwargs", {})
                
                # Sanitize inputs
                sanitized_args = []
                for arg in args:
                    if isinstance(arg, str):
                        sanitized_args.append(InputValidator.sanitize_string(arg))
                    else:
                        sanitized_args.append(arg)
                
                sanitized_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, str):
                        sanitized_kwargs[key] = InputValidator.sanitize_string(value)
                    else:
                        sanitized_kwargs[key] = value
                
                functions.append((tool_obj.function, sanitized_args, sanitized_kwargs))
            else:
                # Tool not found
                functions.append((lambda: {"error": f"Tool not found: {tool_name}"}), [], {})
        
        # Execute in parallel
        return await global_parallel_executor.execute_parallel(functions)
    
    # Replace original methods with enhanced versions
    agent.process_message = enhanced_process_message
    agent.stream_response = enhanced_stream_response
    agent.process_message_with_image = enhanced_process_message_with_image
    agent.get_structured_response = enhanced_get_structured_response
    
    # Add new methods
    agent.execute_tools_in_parallel = execute_tools_in_parallel
    
    # Add security and performance attributes
    agent.security = {
        "permission_system": global_permission_system,
        "input_validator": InputValidator,
    }
    
    agent.performance = {
        "response_cache": global_response_cache,
        "parallel_executor": global_parallel_executor,
    }
    
    # Add enhanced flag
    agent.is_enhanced = True
    
    return agent

# Function to enhance tools with security features
def enhance_tool(tool, resource_name: str = None):
    """
    Enhance a tool with security features.
    
    Args:
        tool: Tool to enhance
        resource_name: Optional resource name for permission checking
    
    Returns:
        Enhanced tool
    """
    original_function = tool.