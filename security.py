"""
Security module for the Anthropic-powered Agent
"""

import re
import time
import hashlib
import logging
import functools
import ipaddress
import urllib.parse
from typing import Dict, List, Any, Optional, Union, Callable, Set, Tuple
from datetime import datetime, timedelta
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InputValidator:
    """Validate and sanitize inputs to prevent security issues."""
    
    @staticmethod
    def validate_string(value: str, min_length: int = 0, max_length: int = 1000000, 
                        pattern: Optional[str] = None, strip: bool = True) -> Tuple[bool, str, Optional[str]]:
        """
        Validate a string input.
        
        Args:
            value: The string to validate
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            pattern: Regex pattern the string must match
            strip: Whether to strip whitespace
            
        Returns:
            (is_valid, sanitized_value, error_message)
        """
        # Check if value is a string
        if not isinstance(value, str):
            return False, "", f"Expected string, got {type(value).__name__}"
        
        # Strip whitespace if requested
        if strip:
            value = value.strip()
        
        # Check length constraints
        if len(value) < min_length:
            return False, value, f"Input is too short (minimum length: {min_length})"
        
        if len(value) > max_length:
            # Truncate to max_length for the returned value
            return False, value[:max_length], f"Input is too long (maximum length: {max_length})"
        
        # Check pattern
        if pattern and not re.match(pattern, value):
            return False, value, f"Input does not match required pattern"
        
        return True, value, None
    
    @staticmethod
    def validate_integer(value: Any, min_value: Optional[int] = None, 
                        max_value: Optional[int] = None) -> Tuple[bool, Optional[int], Optional[str]]:
        """
        Validate an integer input.
        
        Args:
            value: The value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            (is_valid, sanitized_value, error_message)
        """
        # Try to convert to int
        try:
            if isinstance(value, str):
                value = value.strip()
            int_value = int(value)
        except (ValueError, TypeError):
            return False, None, f"Expected integer, got {type(value).__name__}"
        
        # Check range constraints
        if min_value is not None and int_value < min_value:
            return False, int_value, f"Value is too small (minimum: {min_value})"
        
        if max_value is not None and int_value > max_value:
            return False, int_value, f"Value is too large (maximum: {max_value})"
        
        return True, int_value, None
    
    @staticmethod
    def validate_float(value: Any, min_value: Optional[float] = None, 
                      max_value: Optional[float] = None) -> Tuple[bool, Optional[float], Optional[str]]:
        """
        Validate a float input.
        
        Args:
            value: The value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            (is_valid, sanitized_value, error_message)
        """
        # Try to convert to float
        try:
            if isinstance(value, str):
                value = value.strip()
            float_value = float(value)
        except (ValueError, TypeError):
            return False, None, f"Expected float, got {type(value).__name__}"
        
        # Check range constraints
        if min_value is not None and float_value < min_value:
            return False, float_value, f"Value is too small (minimum: {min_value})"
        
        if max_value is not None and float_value > max_value:
            return False, float_value, f"Value is too large (maximum: {max_value})"
        
        return True, float_value, None
    
    @staticmethod
    def validate_boolean(value: Any) -> Tuple[bool, Optional[bool], Optional[str]]:
        """
        Validate a boolean input.
        
        Args:
            value: The value to validate
            
        Returns:
            (is_valid, sanitized_value, error_message)
        """
        # Direct boolean
        if isinstance(value, bool):
            return True, value, None
        
        # String representation
        if isinstance(value, str):
            value = value.strip().lower()
            if value in ('true', 't', 'yes', 'y', '1'):
                return True, True, None
            elif value in ('false', 'f', 'no', 'n', '0'):
                return True, False, None
        
        # Integer representation
        elif isinstance(value, int):
            if value == 1:
                return True, True, None
            elif value == 0:
                return True, False, None
        
        return False, None, f"Expected boolean, got {type(value).__name__}"
    
    @staticmethod
    def validate_url(url: str, allowed_schemes: List[str] = ['http', 'https'], 
                    allowed_domains: Optional[List[str]] = None) -> Tuple[bool, str, Optional[str]]:
        """
        Validate a URL.
        
        Args:
            url: The URL to validate
            allowed_schemes: List of allowed URL schemes
            allowed_domains: Optional list of allowed domains
            
        Returns:
            (is_valid, sanitized_value, error_message)
        """
        if not isinstance(url, str):
            return False, "", f"Expected string, got {type(url).__name__}"
        
        # Strip whitespace
        url = url.strip()
        
        try:
            # Parse URL
            parsed = urllib.parse.urlparse(url)
            
            # Check scheme
            if parsed.scheme not in allowed_schemes:
                return False, url, f"URL scheme not allowed (allowed: {', '.join(allowed_schemes)})"
            
            # Check domain if restricted
            if allowed_domains:
                domain = parsed.netloc.lower()
                if not any(domain.endswith(d.lower()) for d in allowed_domains):
                    return False, url, f"URL domain not allowed (allowed: {', '.join(allowed_domains)})"
            
            # Check if URL has at least a scheme and netloc
            if not (parsed.scheme and parsed.netloc):
                return False, url, "URL must include both scheme and domain"
            
            return True, url, None
        
        except Exception as e:
            return False, url, f"Invalid URL: {str(e)}"
    
    @staticmethod
    def validate_email(email: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate an email address.
        
        Args:
            email: The email to validate
            
        Returns:
            (is_valid, sanitized_value, error_message)
        """
        if not isinstance(email, str):
            return False, "", f"Expected string, got {type(email).__name__}"
        
        # Strip whitespace
        email = email.strip()
        
        # Simple regex for email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        if re.match(email_pattern, email):
            return True, email, None
        else:
            return False, email, "Invalid email format"
    
    @staticmethod
    def validate_ip_address(ip: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate an IP address (IPv4 or IPv6).
        
        Args:
            ip: The IP address to validate
            
        Returns:
            (is_valid, sanitized_value, error_message)
        """
        if not isinstance(ip, str):
            return False, "", f"Expected string, got {type(ip).__name__}"
        
        # Strip whitespace
        ip = ip.strip()
        
        try:
            ipaddress.ip_address(ip)
            return True, ip, None
        except ValueError:
            return False, ip, "Invalid IP address format"
    
    @staticmethod
    def sanitize_html(html: str) -> str:
        """
        Sanitize HTML content to prevent XSS attacks.
        Removes all tags except for a whitelist of safe tags.
        
        Args:
            html: The HTML content to sanitize
            
        Returns:
            Sanitized HTML
        """
        # Define allowed tags and attributes
        allowed_tags = [
            'a', 'abbr', 'acronym', 'b', 'blockquote', 'br', 'code', 'div', 'em',
            'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'hr', 'i', 'img', 'kbd', 'li', 'ol', 
            'p', 'pre', 'q', 's', 'span', 'strong', 'sub', 'sup', 'table', 'tbody', 
            'td', 'tfoot', 'th', 'thead', 'tr', 'tt', 'u', 'ul'
        ]
        
        allowed_attrs = {
            'a': ['href', 'title', 'target', 'rel'],
            'img': ['src', 'alt', 'title', 'width', 'height'],
            'div': ['class'],
            'span': ['class'],
            'p': ['class'],
            'pre': ['class'],
            'code': ['class'],
            'table': ['class', 'width'],
            'th': ['colspan', 'rowspan'],
            'td': ['colspan', 'rowspan']
        }
        
        # Simple tag stripping (a more robust solution would use a proper HTML parser)
        # Replace this with a proper HTML sanitizer library in production
        
        # First, replace < and > in non-tag context
        escaped_html = re.sub(r'&(?![a-z]+;)', '&amp;', html)
        
        # Strip all tags
        sanitized = re.sub(r'<[^>]*>', '', escaped_html)
        
        return sanitized
    
    @staticmethod
    def validate_json(json_str: str) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Validate a JSON string.
        
        Args:
            json_str: The JSON string to validate
            
        Returns:
            (is_valid, parsed_json, error_message)
        """
        import json
        
        if not isinstance(json_str, str):
            return False, None, f"Expected string, got {type(json_str).__name__}"
        
        try:
            parsed = json.loads(json_str)
            return True, parsed, None
        except json.JSONDecodeError as e:
            return False, None, f"Invalid JSON: {str(e)}"

class RateLimiter:
    """Rate limiting implementation to prevent abuse."""
    
    def __init__(self, limit: int, window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            limit: Maximum number of requests per window
            window: Time window in seconds
        """
        self.limit = limit
        self.window = window
        self.request_counts = {}  # key -> list of timestamps
        self.lock = Lock()
    
    def is_allowed(self, key: str) -> bool:
        """
        Check if a request is allowed based on the rate limit.
        
        Args:
            key: Identifier for the client (e.g., API key, IP address)
            
        Returns:
            True if request is allowed, False otherwise
        """
        with self.lock:
            now = time.time()
            
            # Initialize or clean up old entries
            if key not in self.request_counts:
                self.request_counts[key] = []
            
            # Remove entries outside the current window
            window_start = now - self.window
            self.request_counts[key] = [t for t in self.request_counts[key] if t >= window_start]
            
            # Check if limit has been reached
            if len(self.request_counts[key]) >= self.limit:
                return False
            
            # Record this request
            self.request_counts[key].append(now)
            return True
    
    def get_retry_after(self, key: str) -> int:
        """
        Get the number of seconds to wait before retrying.
        
        Args:
            key: Identifier for the client
            
        Returns:
            Seconds to wait before the next request would be allowed
        """
        with self.lock:
            if key not in self.request_counts or not self.request_counts[key]:
                return 0
            
            now = time.time()
            oldest_request = min(self.request_counts[key])
            next_allowed = oldest_request + self.window
            
            retry_after = max(0, int(next_allowed - now))
            return retry_after
    
    def reset(self, key: str):
        """
        Reset rate limit counter for a key.
        
        Args:
            key: Identifier for the client
        """
        with self.lock:
            if key in self.request_counts:
                del self.request_counts[key]

class PermissionSystem:
    """Permission system for controlling access to tools."""
    
    # Permission levels from lowest to highest
    PERMISSION_LEVELS = ["none", "read", "write", "admin"]
    
    def __init__(self):
        """Initialize the permission system."""
        self.role_permissions = {
            "guest": {
                "default": "none",
                "categories": {
                    "claude": "read"
                }
            },
            "user": {
                "default": "read",
                "categories": {
                    "github": "read",
                    "claude": "read",
                    "system": "read",
                    "rag": "read",
                    "image": "read",
                    "cookbook": "read"
                }
            },
            "power_user": {
                "default": "read",
                "categories": {
                    "github": "write",
                    "claude": "write",
                    "system": "write",
                    "rag": "write",
                    "image": "write",
                    "cookbook": "write"
                }
            },
            "admin": {
                "default": "admin",
                "categories": {}
            }
        }
        
        self.user_roles = {}  # user_id -> role
        self.user_permissions = {}  # user_id -> {category -> level}
        self.tool_permissions = {}  # tool_name -> required_level
    
    def assign_role(self, user_id: str, role: str):
        """
        Assign a role to a user.
        
        Args:
            user_id: User identifier
            role: Role to assign
        """
        if role not in self.role_permissions:
            raise ValueError(f"Unknown role: {role}")
        
        self.user_roles[user_id] = role
        self._update_user_permissions(user_id)
    
    def _update_user_permissions(self, user_id: str):
        """
        Update user permissions based on their role.
        
        Args:
            user_id: User identifier
        """
        if user_id not in self.user_roles:
            return
        
        role = self.user_roles[user_id]
        role_perms = self.role_permissions[role]
        
        # Initialize with default permission
        user_perms = {"default": role_perms["default"]}
        
        # Add category-specific permissions
        for category, level in role_perms["categories"].items():
            user_perms[category] = level
        
        self.user_permissions[user_id] = user_perms
    
    def set_tool_permission(self, tool_name: str, required_level: str):
        """
        Set the permission level required for a tool.
        
        Args:
            tool_name: Tool identifier
            required_level: Required permission level
        """
        if required_level not in self.PERMISSION_LEVELS:
            raise ValueError(f"Invalid permission level: {required_level}")
        
        self.tool_permissions[tool_name] = required_level
    
    def set_category_permissions(self, category: str, required_level: str):
        """
        Set the permission level required for all tools in a category.
        
        Args:
            category: Tool category
            required_level: Required permission level
        """
        if required_level not in self.PERMISSION_LEVELS:
            raise ValueError(f"Invalid permission level: {required_level}")
        
        # This is used when registering tools
        # The actual application happens in register_tool_permissions
        self.category_permission = {category: required_level}
    
    def register_tool_permissions(self, tools: Dict[str, Any], tool_categories: Dict[str, List[str]]):
        """
        Register permissions for a set of tools based on their categories.
        
        Args:
            tools: Dictionary of tools (name -> tool object)
            tool_categories: Dictionary of categories (category -> list of tool names)
        """
        # Set default permission for all tools to "read"
        for tool_name in tools:
            if tool_name not in self.tool_permissions:
                self.tool_permissions[tool_name] = "read"
        
        # Apply category-based permissions
        for category, tool_names in tool_categories.items():
            for tool_name in tool_names:
                if category in self.category_permission:
                    self.tool_permissions[tool_name] = self.category_permission[category]
    
    def has_permission(self, user_id: str, tool_name: str) -> bool:
        """
        Check if a user has permission to use a tool.
        
        Args:
            user_id: User identifier
            tool_name: Tool identifier
            
        Returns:
            True if user has permission, False otherwise
        """
        # Admin always has permission
        if self.get_user_role(user_id) == "admin":
            return True
        
        # Get user's permissions
        if user_id not in self.user_permissions:
            return False
        
        user_perms = self.user_permissions[user_id]
        
        # Get tool's required permission level
        tool_level = self.tool_permissions.get(tool_name, "read")
        
        # Find the tool's category
        tool_category = None
        for category, tool_names in tool_categories.items():
            if tool_name in tool_names:
                tool_category = category
                break
        
        # Use category-specific permission level if available, else use default
        user_level = user_perms.get(tool_category, user_perms["default"])
        
        # Check permission
        return self.PERMISSION_LEVELS.index(user_level) >= self.PERMISSION_LEVELS.index(tool_level)
    
    def get_user_role(self, user_id: str) -> str:
        """
        Get a user's role.
        
        Args:
            user_id: User identifier
            
        Returns:
            User's role, or "guest" if not assigned
        """
        return self.user_roles.get(user_id, "guest")
    
    def get_allowed_tools(self, user_id: str, available_tools: Dict[str, Any]) -> List[str]:
        """
        Get a list of tool names that a user has permission to use.
        
        Args:
            user_id: User identifier
            available_tools: Dictionary of available tools
            
        Returns:
            List of tool names the user can use
        """
        return [name for name in available_tools if self.has_permission(user_id, name)]

# Helper functions

def validate_tool_parameters(parameters: Dict[str, Any], tool_parameters: List[Any]) -> Tuple[bool, Dict[str, Any], List[str]]:
    """
    Validate parameters for a tool based on its parameter definitions.
    
    Args:
        parameters: Parameter values to validate
        tool_parameters: List of parameter definitions
        
    Returns:
        (is_valid, validated_parameters, error_messages)
    """
    # Initialize output
    validated = {}
    errors = []
    
    # Create lookup for parameter definitions
    param_defs = {p.name: p for p in tool_parameters}
    
    # Check for missing required parameters
    for name, param_def in param_defs.items():
        if param_def.required and name not in parameters:
            errors.append(f"Missing required parameter: {name}")
    
    # Validate each provided parameter
    for name, value in parameters.items():
        if name not in param_defs:
            errors.append(f"Unknown parameter: {name}")
            continue
        
        param_def = param_defs[name]
        param_type = param_def.type
        
        # Validate based on type
        if param_type == "string":
            is_valid, sanitized, error = InputValidator.validate_string(value)
            if not is_valid:
                errors.append(f"{name}: {error}")
            else:
                validated[name] = sanitized
        
        elif param_type == "integer":
            is_valid, sanitized, error = InputValidator.validate_integer(value)
            if not is_valid:
                errors.append(f"{name}: {error}")
            else:
                validated[name] = sanitized
        
        elif param_type == "number":
            is_valid, sanitized, error = InputValidator.validate_float(value)
            if not is_valid:
                errors.append(f"{name}: {error}")
            else:
                validated[name] = sanitized
        
        elif param_type == "boolean":
            is_valid, sanitized, error = InputValidator.validate_boolean(value)
            if not is_valid:
                errors.append(f"{name}: {error}")
            else:
                validated[name] = sanitized
        
        else:
            # For complex types, we'll do basic validation
            validated[name] = value
    
    # Use default values for non-provided optional parameters
    for name, param_def in param_defs.items():
        if not param_def.required and name not in parameters and param_def.default is not None:
            validated[name] = param_def.default
    
    return len(errors) == 0, validated, errors

def secure_hash(value: str) -> str:
    """
    Create a secure hash of a string.
    
    Args:
        value: String to hash
        
    Returns:
        Secure hash
    """
    return hashlib.sha256(value.encode()).hexdigest()

def rate_limit_decorator(limiter: RateLimiter, key_func: Callable = lambda *args, **kwargs: "default"):
    """
    Decorator for rate limiting function calls.
    
    Args:
        limiter: RateLimiter instance
        key_func: Function to extract key from arguments
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = key_func(*args, **kwargs)
            
            if not limiter.is_allowed(key):
                retry_after = limiter.get_retry_after(key)
                raise RateLimitExceeded(f"Rate limit exceeded. Try again in {retry_after} seconds.")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator

class SecurityException(Exception):
    """Base class for security-related exceptions."""
    pass

class ValidationError(SecurityException):
    """Exception raised for validation errors."""
    pass

class PermissionDenied(SecurityException):
    """Exception raised when permission is denied."""
    pass

class RateLimitExceeded(SecurityException):
    """Exception raised when rate limit is exceeded."""
    pass