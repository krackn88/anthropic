"""
Anthropic-powered Agent with tool execution capabilities
"""

import os
import json
import logging
import base64
import mimetypes
import asyncio
from typing import Dict, List, Optional, Any, Union, Type, Tuple
from datetime import datetime
from anthropic import Anthropic, APIConnectionError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ToolParameter:
    """A parameter for a tool."""
    
    def __init__(self, name: str, type: str, description: str, required: bool = True, default: Any = None):
        """Initialize a tool parameter."""
        self.name = name
        self.type = type
        self.description = description
        self.required = required
        self.default = default

class Tool:
    """A tool that can be executed by the agent."""
    
    def __init__(self, name: str, description: str, parameters: List[ToolParameter], function: Any, category: str = "general"):
        """Initialize a tool."""
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        self.category = category

class Memory:
    """A memory system that stores conversation history."""
    
    def __init__(self, max_messages: int = 100):
        """Initialize memory."""
        self.messages = []
        self.max_messages = max_messages
    
    def add(self, role: str, content: str):
        """Add a message to memory."""
        self.messages.append({"role": role, "content": content})
        
        # Trim if exceeds max_messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.messages
    
    def clear(self):
        """Clear memory."""
        self.messages = []

class Agent:
    """Anthropic-powered agent with tool execution capabilities."""
    
    def __init__(self, model: str = "claude-3-opus-20240229", use_rag: bool = False, max_tokens: int = 4096):
        """Initialize the agent."""
        self.model = model
        self.memory = Memory()
        self.tools = {}
        self.tool_categories = {}
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.use_rag = use_rag
        self.rag_system = None
        self.max_tokens = max_tokens
        self.usage_stats = {
            "api_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "last_call_timestamp": None
        }
        
        # Initialize RAG system if enabled
        if use_rag:
            from rag import RAGSystem
            self.rag_system = RAGSystem()
    
    def register_tools(self, tools: List[Tool]):
        """Register tools with the agent."""
        for tool in tools:
            self.tools[tool.name] = tool
            if tool.category not in self.tool_categories:
                self.tool_categories[tool.category] = []
            self.tool_categories[tool.category].append(tool.name)
    
    def _get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Get tool definitions for Claude API."""
        tool_defs = []
        for name, tool in self.tools.items():
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }
            
            for param in tool.parameters:
                param_type = param.type
                # Map Python types to JSON Schema types
                if param_type == "string":
                    param_schema = {"type": "string"}
                elif param_type == "integer":
                    param_schema = {"type": "integer"}
                elif param_type == "number":
                    param_schema = {"type": "number"}
                elif param_type == "boolean":
                    param_schema = {"type": "boolean"}
                elif param_type == "array":
                    param_schema = {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                elif param_type == "object":
                    param_schema = {"type": "object"}
                else:
                    param_schema = {"type": "string"}
                
                parameters["properties"][param.name] = param_schema
                if param.required:
                    parameters["required"].append(param.name)
            
            tool_def = {
                "name": name,
                "description": tool.description,
                "input_schema": parameters
            }
            tool_defs.append(tool_def)
        
        return tool_defs
    
    def _enrich_system_prompt(self) -> str:
        """Enrich the system prompt with context."""
        enriched = "You are an assistant powered by Anthropic's Claude model. You have access to various tools and capabilities."
        
        # If using RAG, mention it
        if self.use_rag:
            enriched += "\n\nYou have access to a knowledge base of documents. When appropriate, reference information from this knowledge base in your responses."
        
        # Add tool information if available
        if self.tools:
            enriched += f"\n\nYou have access to {len(self.tools)} tools across {len(self.tool_categories)} categories:"
            for category, tool_names in self.tool_categories.items():
                enriched += f"\n- {category.capitalize()}: {', '.join(tool_names)}"
        
        return enriched
    
    async def _enrich_with_rag(self, query: str) -> str:
        """Enrich a query with RAG context."""
        if not self.rag_system:
            return ""
        
        try:
            context = await self.rag_system.get_query_context(query)
            return context
        except Exception as e:
            logger.error(f"Error enriching with RAG: {str(e)}")
            return ""
    
    def _update_usage_stats(self, response):
        """Update usage statistics from a response."""
        self.usage_stats["api_calls"] += 1
        self.usage_stats["last_call_timestamp"] = datetime.now().isoformat()
        
        # Get token counts if available
        if hasattr(response, 'usage'):
            prompt_tokens = response.usage.input_tokens
            completion_tokens = response.usage.output_tokens
            
            self.usage_stats["total_prompt_tokens"] += prompt_tokens
            self.usage_stats["total_completion_tokens"] += completion_tokens
            self.usage_stats["total_tokens"] += prompt_tokens + completion_tokens
            
            # Update cost estimation (approximate rates)
            # These rates should be updated based on Anthropic's pricing
            model_cost_mapping = {
                "claude-3-opus-20240229": (15.0, 75.0),  # (input_per_million, output_per_million)
                "claude-3-sonnet-20240229": (3.0, 15.0),
                "claude-3-haiku-20240307": (0.25, 1.25),
                "claude-2.1": (8.0, 24.0),
                "claude-2.0": (8.0, 24.0),
                "claude-instant-1.2": (1.63, 5.51),
            }
            
            # Get cost rates for the model, default to opus if not found
            input_rate, output_rate = model_cost_mapping.get(
                self.model, model_cost_mapping["claude-3-opus-20240229"]
            )
            
            # Calculate cost
            input_cost = (prompt_tokens / 1_000_000) * input_rate
            output_cost = (completion_tokens / 1_000_000) * output_rate
            total_cost = input_cost + output_cost
            
            self.usage_stats["estimated_cost_usd"] += total_cost
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return self.usage_stats
    
    async def process_message(self, user_message: str) -> str:
        """Process a user message and generate a response."""
        # Add user message to memory
        self.memory.add("user", user_message)
        
        # Get RAG context if enabled
        rag_context = ""
        if self.use_rag:
            rag_context = await self._enrich_with_rag(user_message)
        
        # Prepare messages
        enriched_system_prompt = self._enrich_system_prompt()
        messages = [
            {"role": "system", "content": enriched_system_prompt}
        ] + self.memory.get_history()
        
        # If RAG context is available, add it before the user's message
        if rag_context:
            # Find the last user message
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    # Replace with the enriched version
                    original_message = messages[i]["content"]
                    messages[i]["content"] = f"{rag_context}\n\nUser query: {original_message}\n\nPlease use the above context to help answer the query."
                    break
        
        # Prepare API call
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "temperature": 0.7,
        }
        
        if self.tools:
            kwargs["tools"] = self._get_tool_definitions()
        
        try:
            # Call the model
            response = self.client.messages.create(**kwargs)
            
            # Update usage statistics
            self._update_usage_stats(response)
            
            # Process response
            if hasattr(response, 'content'):
                text_blocks = [block.text for block in response.content if block.type == 'text']
                assistant_response = ''.join(text_blocks)
                self.memory.add("assistant", assistant_response)
                return assistant_response
            else:
                return "I apologize, but I encountered an issue processing your request."
        
        except APIConnectionError as e:
            logger.error(f"API connection error: {str(e)}")
            return f"I'm having trouble connecting to the API: {str(e)}"
            
        except Exception as e:
            logger.error(f"API call error: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    async def stream_response(self, user_message: str):
        """Stream a response to a user message."""
        # Add user message to memory
        self.memory.add("user", user_message)
        
        # Get RAG context if enabled
        rag_context = ""
        if self.use_rag:
            rag_context = await self._enrich_with_rag(user_message)
        
        # Prepare messages
        enriched_system_prompt = self._enrich_system_prompt()
        messages = [
            {"role": "system", "content": enriched_system_prompt}
        ] + self.memory.get_history()
        
        # If RAG context is available, add it before the user's message
        if rag_context:
            # Find the last user message
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    # Replace with the enriched version
                    original_message = messages[i]["content"]
                    messages[i]["content"] = f"{rag_context}\n\nUser query: {original_message}\n\nPlease use the above context to help answer the query."
                    break
        
        # Prepare API call
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "temperature": 0.7,
            "stream": True
        }
        
        if self.tools:
            kwargs["tools"] = self._get_tool_definitions()
        
        try:
            # Stream response
            full_response = ""
            with self.client.messages.stream(**kwargs) as stream:
                for chunk in stream:
                    if chunk.type == "content_block_delta" and chunk.delta.type == "text":
                        yield chunk.delta.text
                        full_response += chunk.delta.text
            
            # Add full response to memory
            self.memory.add("assistant", full_response)
            
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"I encountered an error while processing your request: {str(e)}"
    
    async def process_message_with_image(self, user_message: str, image_path: str) -> str:
        """Process a user message with an image and generate a response."""
        # Add user message to memory
        self.memory.add("user", f"[Message with image] {user_message}")
        
        # Prepare messages
        enriched_system_prompt = self._enrich_system_prompt()
        messages = [
            {"role": "system", "content": enriched_system_prompt}
        ]
        
        # Add conversation history excluding the last message (which will be replaced)
        history = self.memory.get_history()[:-1]  # Exclude the last message we just added
        messages.extend(history)
        
        # Get RAG context if enabled
        rag_context = ""
        if self.use_rag:
            rag_context = await self._enrich_with_rag(user_message)
        
        # Prepare the multimodal message
        try:
            from image_processing import ImageProcessor
            # Encode image
            image_content = ImageProcessor.encode_image_base64(image_path)
            
            # Create multimodal message content
            multimodal_content = [
                {
                    "type": "text",
                    "text": f"{rag_context}\n\n{user_message}" if rag_context else user_message
                },
                image_content
            ]
            
            # Add to messages
            messages.append({
                "role": "user",
                "content": multimodal_content
            })
            
        except Exception as e:
            # Fall back to text-only if image processing fails
            logger.error(f"Error processing image: {str(e)}")
            messages.append({
                "role": "user",
                "content": f"{rag_context}\n\n{user_message} [Note: Failed to process image: {str(e)}]" if rag_context else f"{user_message} [Note: Failed to process image: {str(e)}]"
            })
        
        # Prepare API call
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
        }
        
        if self.tools:
            kwargs["tools"] = self._get_tool_definitions()
        
        try:
            # Call the model
            response = self.client.messages.create(**kwargs)
            
            # Update usage statistics
            self._update_usage_stats(response)
            
            # Process response
            if hasattr(response, 'content'):
                text_blocks = [block.text for block in response.content if block.type == 'text']
                assistant_response = ''.join(text_blocks)
                self.memory.add("assistant", assistant_response)
                return assistant_response
            else:
                return "I apologize, but I encountered an issue processing your request with the image."
        
        except Exception as e:
            logger.error(f"API call error: {str(e)}")
            return f"I encountered an error while processing your request with the image: {str(e)}"
    
    async def get_structured_response(self, user_message: str, schema: Type) -> Dict[str, Any]:
        """Get a structured response using JSON mode."""
        # Add user message to memory
        self.memory.add("user", user_message)
        
        # Get RAG context if enabled
        rag_context = ""
        if self.use_rag:
            rag_context = await self._enrich_with_rag(user_message)
        
        # Prepare messages
        enriched_system_prompt = self._enrich_system_prompt()
        # Add schema information to system prompt
        schema_prompt = f"\n\nOutput should be formatted according to the following JSON schema: {schema.schema_json()}"
        system_prompt = enriched_system_prompt + schema_prompt
        
        messages = [
            {"role": "system", "content": system_prompt}
        ] + self.memory.get_history()
        
        # If RAG context is available, add it before the user's message
        if rag_context:
            # Find the last user message
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    # Replace with the enriched version
                    original_message = messages[i]["content"]
                    messages[i]["content"] = f"{rag_context}\n\nUser query: {original_message}\n\nPlease use the above context to help answer the query."
                    break
        
        # Prepare API call
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "temperature": 0.7,
            "response_format": {"type": "json_object"}
        }
        
        try:
            # Call the model
            response = self.client.messages.create(**kwargs)
            
            # Update usage statistics
            self._update_usage_stats(response)
            
            # Process response
            if hasattr(response, 'content'):
                text_blocks = [block.text for block in response.content if block.type == 'text']
                json_response = ''.join(text_blocks)
                
                # Parse JSON response
                try:
                    parsed_json = json.loads(json_response)
                    # Validate against schema
                    validated_data = schema.parse_obj(parsed_json)
                    
                    # Add response to memory
                    self.memory.add("assistant", json_response)
                    
                    return validated_data.dict()
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {str(e)}")
                    return {"error": f"Failed to parse JSON response: {str(e)}", "raw_response": json_response}
                except Exception as e:
                    logger.error(f"Schema validation error: {str(e)}")
                    return {"error": f"Failed to validate response against schema: {str(e)}", "raw_response": json_response}
            else:
                return {"error": "No content in response"}
        
        except Exception as e:
            logger.error(f"API call error: {str(e)}")
            return {"error": f"API call error: {str(e)}"}