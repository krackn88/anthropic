"""
Performance optimization module for the Anthropic-powered Agent
Handles response caching and parallel tool execution
"""

import os
import json
import time
import hashlib
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from functools import wraps
import pickle
import concurrent.futures

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResponseCache:
    """Caching system for API responses to reduce redundant calls."""
    
    def __init__(self, cache_dir: str = "cache", ttl: int = 3600, max_size_mb: int = 100):
        """
        Initialize response cache.
        
        Args:
            cache_dir: Directory to store cache files
            ttl: Time to live in seconds (default: 1 hour)
            max_size_mb: Maximum cache size in MB
        """
        self.cache_dir = cache_dir
        self.ttl = ttl
        self.max_size_mb = max_size_mb
        self.memory_cache = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize cache metadata
        self.metadata_path = os.path.join(cache_dir, "metadata.json")
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata from file."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Failed to load cache metadata, creating new")
        
        # Create default metadata
        return {
            "total_size_bytes": 0,
            "entries": {},
            "last_cleanup": datetime.now().isoformat()
        }
    
    def _save_metadata(self):
        """Save cache metadata to file."""
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f)
        except IOError:
            logger.error("Failed to save cache metadata")
    
    def _calculate_key(self, func_name: str, args: Tuple, kwargs: Dict[str, Any]) -> str:
        """
        Calculate cache key from function name and arguments.
        
        Args:
            func_name: Function name
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            Cache key
        """
        # Create a string representation of the function call
        arg_str = str(args) + str(sorted(kwargs.items()))
        
        # Hash it to get a fixed-length key
        key = hashlib.md5(f"{func_name}:{arg_str}".encode()).hexdigest()
        
        return key
    
    def has(self, key: str) -> bool:
        """
        Check if a key exists in the cache and is still valid.
        
        Args:
            key: Cache key
            
        Returns:
            True if valid cache entry exists, False otherwise
        """
        # Check memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if entry["expires"] > time.time():
                return True
            else:
                # Expired, remove from memory cache
                del self.memory_cache[key]
        
        # Check metadata
        if key in self.metadata["entries"]:
            entry = self.metadata["entries"][key]
            
            # Check if expired
            expires = datetime.fromisoformat(entry["expires"])
            if expires > datetime.now():
                return True
            else:
                # Expired, remove from metadata
                self._remove_entry(key)
        
        return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        # Check memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if entry["expires"] > time.time():
                return entry["value"]
            else:
                # Expired, remove from memory cache
                del self.memory_cache[key]
        
        # Check file cache
        if key in self.metadata["entries"]:
            entry = self.metadata["entries"][key]
            
            # Check if expired
            expires = datetime.fromisoformat(entry["expires"])
            if expires > datetime.now():
                try:
                    cache_path = os.path.join(self.cache_dir, f"{key}.cache")
                    with open(cache_path, 'rb') as f:
                        # Load and also store in memory cache for faster future access
                        value = pickle.load(f)
                        self.memory_cache[key] = {
                            "value": value,
                            "expires": time.time() + self.ttl
                        }
                        return value
                except (IOError, pickle.PickleError):
                    logger.error(f"Failed to load cache file for key {key}")
                    self._remove_entry(key)
            else:
                # Expired, remove from metadata
                self._remove_entry(key)
        
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: use instance ttl)
            
        Returns:
            True if successful, False otherwise
        """
        # Use instance TTL if not specified
        if ttl is None:
            ttl = self.ttl
        
        # Calculate expiration time
        expires = time.time() + ttl
        expires_iso = datetime.fromtimestamp(expires).isoformat()
        
        # Store in memory cache
        self.memory_cache[key] = {
            "value": value,
            "expires": expires
        }
        
        try:
            # Store in file cache
            cache_path = os.path.join(self.cache_dir, f"{key}.cache")
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Get file size
            size_bytes = os.path.getsize(cache_path)
            
            # Update metadata
            self.metadata["entries"][key] = {
                "size_bytes": size_bytes,
                "created": datetime.now().isoformat(),
                "expires": expires_iso,
                "type": type(value).__name__
            }
            
            # Update total size
            self.metadata["total_size_bytes"] += size_bytes
            
            # Save metadata
            self._save_metadata()
            
            # Check if cleanup needed
            self._check_size_limit()
            
            return True
        except Exception as e:
            logger.error(f"Failed to cache value: {str(e)}")
            return False
    
    def _remove_entry(self, key: str) -> bool:
        """
        Remove an entry from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        # Remove from memory cache
        if key in self.memory_cache:
            del self.memory_cache[key]
        
        # Remove from file cache
        cache_path = os.path.join(self.cache_dir, f"{key}.cache")
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except IOError:
                logger.error(f"Failed to remove cache file for key {key}")
                return False
        
        # Update metadata
        if key in self.metadata["entries"]:
            size = self.metadata["entries"][key]["size_bytes"]
            self.metadata["total_size_bytes"] -= size
            del self.metadata["entries"][key]
            self._save_metadata()
        
        return True
    
    def _check_size_limit(self):
        """Check cache size and clean up if needed."""
        current_size_mb = self.metadata["total_size_bytes"] / (1024 * 1024)
        
        if current_size_mb > self.max_size_mb:
            logger.info(f"Cache size {current_size_mb:.2f} MB exceeds limit {self.max_size_mb} MB, cleaning up")
            self._cleanup()
    
    def _cleanup(self):
        """Clean up expired and oldest entries to maintain size limit."""
        now = datetime.now()
        expired_keys = []
        entries_by_age = []
        
        # Find expired entries
        for key, entry in self.metadata["entries"].items():
            expires = datetime.fromisoformat(entry["expires"])
            if expires < now:
                expired_keys.append(key)
            else:
                # Add to list for potential age-based cleanup
                created = datetime.fromisoformat(entry["created"])
                entries_by_age.append((key, created))
        
        # Remove expired entries
        for key in expired_keys:
            self._remove_entry(key)
        
        # If still over limit, remove oldest entries
        current_size_mb = self.metadata["total_size_bytes"] / (1024 * 1024)
        if current_size_mb > self.max_size_mb:
            # Sort by creation time
            entries_by_age.sort(key=lambda x: x[1])
            
            # Remove oldest entries until under limit
            for key, _ in entries_by_age:
                self._remove_entry(key)
                
                current_size_mb = self.metadata["total_size_bytes"] / (1024 * 1024)
                if current_size_mb <= self.max_size_mb:
                    break
        
        # Update cleanup timestamp
        self.metadata["last_cleanup"] = now.isoformat()
        self._save_metadata()

# Create a global cache instance
global_response_cache = ResponseCache()

def cached(ttl: Optional[int] = None, cache_key_prefix: Optional[str] = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds (default: use cache instance ttl)
        cache_key_prefix: Optional prefix for cache keys
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            func_name = cache_key_prefix or func.__name__
            key = global_response_cache._calculate_key(func_name, args, kwargs)
            
            # Check cache
            cached_result = global_response_cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func_name}")
                return cached_result
            
            # Call original function
            result = await func(*args, **kwargs)
            
            # Cache result
            global_response_cache.set(key, result, ttl)
            
            return result
        
        return wrapper
    
    return decorator

class ParallelExecutor:
    """Execute tools and functions in parallel."""
    
    def __init__(self, max_workers: int = None):
        """
        Initialize parallel executor.
        
        Args:
            max_workers: Maximum number of worker threads or processes
        """
        self.max_workers = max_workers
    
    async def execute_parallel(self, functions: List[Tuple[Callable, List, Dict[str, Any]]]) -> List[Any]:
        """
        Execute multiple functions in parallel.
        
        Args:
            functions: List of (function, args, kwargs) tuples
            
        Returns:
            List of results in the same order as input functions
        """
        # Create task list
        tasks = []
        
        # Create event loop
        loop = asyncio.get_event_loop()
        
        for func, args, kwargs in functions:
            if asyncio.iscoroutinefunction(func):
                # Async function
                tasks.append(func(*args, **kwargs))
            else:
                # Sync function, run in executor
                tasks.append(loop.run_in_executor(
                    None, lambda f=func, a=args, k=kwargs: f(*a, **k)
                ))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in parallel execution: {str(result)}")
                processed_results.append({"error": str(result)})
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def execute_tool_chain(self, tools: List[Dict[str, Any]]) -> List[Any]:
        """
        Execute a chain of tools, potentially with parallel execution steps.
        
        Args:
            tools: List of tool definitions with dependencies
            
        Returns:
            List of results for each tool
        """
        # Tool format: {
        #   "name": "tool_name",
        #   "args": [...],
        #   "kwargs": {...},
        #   "depends_on": [indices of tools this depends on],
        #   "parallel_with": [indices of tools to run in parallel]
        # }
        
        results = [None] * len(tools)
        completed = [False] * len(tools)
        
        # Process tools in order, respecting dependencies
        while not all(completed):
            # Find eligible tools
            parallel_batch = []
            
            for i, tool in enumerate(tools):
                if completed[i]:
                    continue
                
                # Check dependencies
                dependencies_met = True
                for dep_idx in tool.get("depends_on", []):
                    if not completed[dep_idx]:
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    # Eligible for execution
                    parallel_batch.append((i, tool))
            
            if not parallel_batch:
                # No eligible tools, but not all completed - must be a cycle
                raise ValueError("Dependency cycle detected in tool chain")
            
            # Execute batch in parallel
            functions = []
            indices = []
            
            for i, tool in parallel_batch:
                indices.append(i)
                
                # Get tool function
                func = tool["function"]
                args = tool.get("args", [])
                kwargs = tool.get("kwargs", {})
                
                # Update args/kwargs with dependency results if needed
                for j, arg in enumerate(args):
                    if isinstance(arg, dict) and "_depends_on" in arg:
                        dep_idx = arg["_depends_on"]
                        args[j] = results[dep_idx]
                
                for key, value in kwargs.items():
                    if isinstance(value, dict) and "_depends_on" in value:
                        dep_idx = value["_depends_on"]
                        kwargs[key] = results[dep_idx]
                
                functions.append((func, args, kwargs))
            
            # Execute in parallel
            batch_results = await self.execute_parallel(functions)
            
            # Store results
            for i, result in zip(indices, batch_results):
                results[i] = result
                completed[i] = True
        
        return results

# Create a global parallel executor
global_parallel_executor = ParallelExecutor()