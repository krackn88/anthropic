"""
Security module for the Anthropic-powered Agent
Handles input validation, rate limiting, and permissions
"""

import time
import re
import hashlib
import logging
import json
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from pydantic import BaseModel, ValidationError, create_model, validator
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiting implementation using token bucket algorithm."""
    
    def __init__(self, rate: float, capacity: int):
        """
        Initialize rate limiter.
        
        Args:
            rate: Tokens per second to refill
            capacity: Maximum token bucket capacity
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_time = time.time()
        self.user_buckets = {}  # For user-specific rate limiting
        self.ip_buckets = {}    # For IP-based rate limiting
    
    def check_and_update(self, tokens: int = 1) -> bool:
        """
        Check if operation is allowed and update token count.
        
        Args:
            tokens: Number of tokens to consume (default: 1)
            
        Returns:
            True if operation is allowed, False otherwise
        """
        current_time = time.time()
        time_passed = current_time - self.last_time
        self.last_time = current_time
        
        # Refill tokens based on time passed
        self.tokens = min(self.capacity, self.tokens + time_passed * self.rate)
        
        # Check if we have enough tokens
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        else:
            return False
    
    def check_user(self, user_id: str, tokens: int = 1) -> bool:
        """
        Check rate limit for a specific user.
        
        Args:
            user_id: User identifier
            tokens: Number of tokens to consume
            
        Returns:
            True if operation is allowed, False otherwise
        """
        # Initialize user bucket if needed
        if user_id not in self.user_buckets:
            self.user_buckets[user_id] = {
                "tokens": self.capacity,
                "last_time": time.time()
            }
        
        # Get user bucket
        bucket = self.user_buckets[user_id]
        current_time = time.time()
        time_passed = current_time - bucket["last_time"]
        bucket["last_time"] = current_time
        
        # Refill tokens based on time passed
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + time_passed * self.rate)
        
        # Check if we have enough tokens
        if bucket["tokens"] >= tokens:
            bucket["tokens"] -= tokens
            return True
        else:
            return False
    
    def check_ip(self, ip_address: str, tokens: int = 1) -> bool:
        """
        Check rate limit for a specific IP address.
        
        Args:
            ip_address: IP address
            tokens: Number of tokens to consume
            
        Returns:
            True if operation is allowed, False otherwise
        """
        # Similar to check_user, but for IP addresses
        if ip_address not in self.ip_buckets:
            self.ip_buckets[ip_address] = {
                "tokens": self.capacity,
                "last_time": time.time()
            }
        
        bucket = self.ip_buckets[ip_address]
        current_time = time.time()
        time_passed = current_time - bucket["last_time"]
        bucket["last_time"] = current_time
        
        bucket["tokens"] = min(self.capacity, bucket["tokens"] + time_passed * self.rate)
        
        if bucket["tokens"] >= tokens:
            bucket["tokens"] -= tokens
            return True
        else:
            return False

class InputValidator:
    """Validate and sanitize inputs to prevent security issues."""
    
    @staticmethod
    def sanitize_string(input_string: str, max_length: int = 10000) -> str:
        """
        Sanitize a string input.
        
        Args:
            input_string: Input string to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
        """
        if not input_string:
            return ""
        
        # Truncate if too long
        if len(input_string) > max_length:
            input_string = input_string[:max_length]
        
        # Remove potentially malicious characters
        # This is a basic example - more advanced sanitization might be needed
        sanitized = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\[\]\{\}\"\'\`\/\+\=\*\&\%\$\#\@\~]', '', input_string)
        
        return sanitized
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate an email address.
        
        Args:
            email: Email address to validate
            
        Returns:
            True if valid, False otherwise
        """
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(email_pattern, email))
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """
        Validate a URL.
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid, False otherwise
        """
        url_pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
        return bool(re.match(url_pattern, url))
    
    @staticmethod
    def validate_path(path: str) -> bool:
        """
        Validate a file path to prevent path traversal attacks.
        
        Args:
            path: File path to validate
            
        Returns:
            True if valid, False otherwise
        """
        # Check for path traversal sequences
        if '..' in path or '~' in path:
            return False
        
        # Disallow absolute paths
        if path.startswith('/') or (len(path) > 1 and path[1] == ':'):
            return False
        
        return True
    
    @staticmethod
    def create_validator_model(schema: Dict[str, Any]) -> BaseModel:
        """
        Create a Pydantic model for validation based on a schema.
        
        Args:
            schema: Schema definition
            
        Returns:
            Pydantic model class
        """
        field_definitions = {}
        
        for field_name, field_def in schema.get("properties", {}).items():
            field_type = field_def.get("type", "string")
            
            # Map JSON Schema types to Python types
            type_mapping = {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool,
                "array": list,
                "object": dict
            }
            
            python_type = type_mapping.get(field_type, str)
            is_required = field_name in schema.get("required", [])
            
            if is_required:
                field_definitions[field_name] = (python_type, ...)
            else:
                field_definitions[field_name] = (Optional[python_type], None)
        
        # Create and return model
        return create_model('DynamicModel', **field_definitions)

class PermissionSystem:
    """Manage permissions for tools and resources."""
    
    def __init__(self):
        """Initialize the permission system."""
        self.roles = {
            "admin": {"description": "Full access to all tools and resources"},
            "user": {"description": "Standard user with limited access"},
            "guest": {"description": "Read-only access to basic resources"}
        }
        
        self.permissions = {
            "admin": {"*": "rw"},  # Full access to everything
            "user": {               # Standard user permissions
                "github": "rw",
                "claude_tools": "rw",
                "system_tools": "r",  # Read-only for system tools
                "rag": "rw",
                "image": "rw"
            },
            "guest": {              # Guest permissions
                "github": "r",       # Read-only for GitHub
                "claude_tools": "r",  # Read-only for Claude tools
                "rag": "r",           # Read-only for RAG
                "image": "r"          # Read-only for images
            }
        }
        
        self.user_roles = {}  # Map of user IDs to roles
    
    def assign_role(self, user_id: str, role: str) -> bool:
        """
        Assign a role to a user.
        
        Args:
            user_id: User identifier
            role: Role to assign
            
        Returns:
            True if successful, False otherwise
        """
        if role not in self.roles:
            return False
        
        self.user_roles[user_id] = role
        return True
    
    def get_user_role(self, user_id: str) -> str:
        """
        Get a user's role.
        
        Args:
            user_id: User identifier
            
        Returns:
            Role name or 'guest' if not assigned
        """
        return self.user_roles.get(user_id, "guest")
    
    def check_permission(self, user_id: str, resource: str, access_type: str) -> bool:
        """
        Check if a user has permission to access a resource.
        
        Args:
            user_id: User identifier
            resource: Resource to access (e.g., 'github', 'system_tools')
            access_type: Type of access ('r' for read, 'w' for write)
            
        Returns:
            True if allowed, False otherwise
        """
        role = self.get_user_role(user_id)
        role_permissions = self.permissions.get(role, {})
        
        # Check for wildcard permission
        if "*" in role_permissions and access_type in role_permissions["*"]:
            return True
        
        # Check for specific resource permission
        if resource in role_permissions and access_type in role_permissions[resource]:
            return True
        
        # Check for category permissions (if resource contains '.')
        if "." in resource:
            category = resource.split(".")[0]
            if category in role_permissions and access_type in role_permissions[category]:
                return True
        
        return False
    
    def permission_required(self, resource: str, access_type: str):
        """
        Decorator for functions that require permission.
        
        Args:
            resource: Resource to access
            access_type: Type of access ('r' for read, 'w' for write)
            
        Returns:
            Decorator function
        """
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Get user_id from kwargs
                user_id = kwargs.get("user_id")
                
                if not user_id:
                    # Try to extract from first argument if it's a dict-like object
                    if args and hasattr(args[0], "get"):
                        user_id = args[0].get("user_id")
                
                if not user_id:
                    return {"error": "User identification required"}
                
                # Check permission
                if not self.check_permission(user_id, resource, access_type):
                    return {"error": f"Permission denied: {resource}:{access_type}"}
                
                # Call original function
                return await func(*args, **kwargs)
            
            return wrapper
        
        return decorator

# Create singleton instances for global use
global_rate_limiter = RateLimiter(rate=10, capacity=20)  # 10 tokens per second, max 20
global_permission_system = PermissionSystem()

def validate_input(validator_func: Callable) -> Callable:
    """
    Decorator to validate inputs.
    
    Args:
        validator_func: Function that returns (is_valid, error_message)
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Validate inputs
            valid, error = validator_func(*args, **kwargs)
            if not valid:
                return {"error": error}
            
            # Call original function
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def rate_limit(tokens: int = 1, limit_by: str = "global") -> Callable:
    """
    Rate limiting decorator.
    
    Args:
        tokens: Number of tokens to consume
        limit_by: 'global', 'user', or 'ip' to determine rate limit scope
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Apply rate limiting
            if limit_by == "global":
                allowed = global_rate_limiter.check_and_update(tokens)
            elif limit_by == "user":
                user_id = kwargs.get("user_id")
                if not user_id:
                    return {"error": "User ID required for user-based rate limiting"}
                allowed = global_rate_limiter.check_user(user_id, tokens)
            elif limit_by == "ip":
                ip_address = kwargs.get("ip_address")
                if not ip_address:
                    return {"error": "IP address required for IP-based rate limiting"}
                allowed = global_rate_limiter.check_ip(ip_address, tokens)
            else:
                allowed = True  # Default to allowed if invalid limit_by
            
            if not allowed:
                return {"error": "Rate limit exceeded. Please try again later."}
            
            # Call original function
            return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator