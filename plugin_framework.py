"""
Plugin framework for the Anthropic-powered Agent
Enables dynamic discovery and loading of external tools
"""

import os
import sys
import json
import logging
import importlib.util
import inspect
import hashlib
import re
import shutil
import tempfile
import zipfile
import requests
from typing import Dict, List, Any, Optional, Callable, Type, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    homepage: Optional[str] = None
    repository: Optional[str] = None
    license: Optional[str] = None
    min_agent_version: Optional[str] = None
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PluginMetadata':
        """Create from dictionary."""
        return cls(
            name=data.get("name", "Unknown"),
            version=data.get("version", "0.1.0"),
            description=data.get("description", ""),
            author=data.get("author", "Unknown"),
            dependencies=data.get("dependencies", []),
            homepage=data.get("homepage"),
            repository=data.get("repository"),
            license=data.get("license"),
            min_agent_version=data.get("min_agent_version")
        )

class PluginValidationError(Exception):
    """Error raised when plugin validation fails."""
    pass

class Plugin:
    """Plugin for the Anthropic-powered Agent."""
    
    def __init__(self, metadata: PluginMetadata, module: Any, tools: List[Any], plugin_path: str):
        """Initialize plugin."""
        self.metadata = metadata
        self.module = module
        self.tools = tools
        self.plugin_path = plugin_path
        self.enabled = True
        self.errors: List[str] = []
    
    def enable(self) -> None:
        """Enable the plugin."""
        self.enabled = True
        logger.info(f"Enabled plugin: {self.metadata.name}")
    
    def disable(self) -> None:
        """Disable the plugin."""
        self.enabled = False
        logger.info(f"Disabled plugin: {self.metadata.name}")
    
    def get_tools(self) -> List[Any]:
        """Get tools from the plugin."""
        return self.tools if self.enabled else []
    
    def reload(self) -> bool:
        """Reload the plugin."""
        try:
            # Re-import the module
            spec = importlib.util.spec_from_file_location(
                self.module.__name__, 
                os.path.join(self.plugin_path, "main.py")
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Get tools
            if hasattr(module, "get_tools"):
                tools = module.get_tools()
                
                # Validate tools
                validator = ToolValidator()
                for tool in tools:
                    validator.validate_tool(tool)
                
                # Update module and tools
                self.module = module
                self.tools = tools
                
                logger.info(f"Reloaded plugin: {self.metadata.name}")
                return True
            else:
                self.errors.append("Module does not have get_tools function")
                return False
        
        except Exception as e:
            self.errors.append(f"Failed to reload plugin: {str(e)}")
            logger.error(f"Error reloading plugin {self.metadata.name}: {str(e)}")
            return False

class ToolValidator:
    """Validate tools for the Anthropic-powered Agent."""
    
    def __init__(self):
        """Initialize tool validator."""
        pass
    
    def validate_tool(self, tool: Any) -> bool:
        """Validate a tool."""
        from anthropic_agent import Tool, ToolParameter
        
        if not isinstance(tool, Tool):
            raise PluginValidationError(f"Object is not a Tool instance: {type(tool)}")
        
        # Validate required fields
        if not tool.name:
            raise PluginValidationError("Tool name is required")
        
        if not tool.description:
            raise PluginValidationError("Tool description is required")
        
        if not tool.function:
            raise PluginValidationError("Tool function is required")
        
        if not callable(tool.function):
            raise PluginValidationError("Tool function must be callable")
        
        # Validate tool name format
        if not re.match(r'^[a-z][a-z0-9_]*$', tool.name):
            raise PluginValidationError(f"Tool name must start with a lowercase letter and contain only lowercase letters, numbers, and underscores: {tool.name}")
        
        # Validate parameters
        for param in tool.parameters:
            if not isinstance(param, ToolParameter):
                raise PluginValidationError(f"Parameter is not a ToolParameter instance: {type(param)}")
            
            if not param.name:
                raise PluginValidationError("Parameter name is required")
            
            if not param.type:
                raise PluginValidationError("Parameter type is required")
            
            if not param.description:
                raise PluginValidationError("Parameter description is required")
            
            # Validate parameter name format
            if not re.match(r'^[a-z][a-z0-9_]*$', param.name):
                raise PluginValidationError(f"Parameter name must start with a lowercase letter and contain only lowercase letters, numbers, and underscores: {param.name}")
            
            # Validate parameter type
            valid_types = ["string", "integer", "number", "boolean", "array", "object"]
            if param.type not in valid_types:
                raise PluginValidationError(f"Invalid parameter type: {param.type}. Must be one of: {', '.join(valid_types)}")
        
        return True
    
    def validate_tool_function(self, tool: Any) -> bool:
        """Validate a tool function."""
        # Check if function is async
        if not inspect.iscoroutinefunction(tool.function):
            raise PluginValidationError("Tool function must be async (defined with 'async def')")
        
        # Check function signature
        sig = inspect.signature(tool.function)
        
        # Verify parameters match function signature
        func_params = list(sig.parameters.keys())
        tool_params = [param.name for param in tool.parameters]
        
        # Check for missing required parameters in function signature
        for param in tool.parameters:
            if param.required and param.name not in func_params:
                raise PluginValidationError(f"Required parameter '{param.name}' not in function signature")
        
        # Check for return type hint
        return_annotation = sig.return_annotation
        if return_annotation is inspect.Signature.empty:
            raise PluginValidationError("Tool function should have a return type hint")
        
        # Check for Dict[str, Any] return type
        valid_return = False
        if hasattr(return_annotation, "__origin__") and return_annotation.__origin__ is dict:
            if len(return_annotation.__args__) == 2:
                if return_annotation.__args__[0] is str and return_annotation.__args__[1] is Any:
                    valid_return = True
        
        if not valid_return:
            raise PluginValidationError("Tool function should return Dict[str, Any]")
        
        return True

class PluginManager:
    """Manage plugins for the Anthropic-powered Agent."""
    
    def __init__(self, plugins_dir: str = "plugins", plugin_registry_url: Optional[str] = None):
        """Initialize plugin manager."""
        self.plugins_dir = plugins_dir
        self.plugin_registry_url = plugin_registry_url
        self.plugins: Dict[str, Plugin] = {}
        self.validator = ToolValidator()
        
        # Create plugins directory if it doesn't exist
        os.makedirs(plugins_dir, exist_ok=True)
    
    def discover_plugins(self) -> List[Dict[str, Any]]:
        """Discover available plugins."""
        discovered = []
        
        # Ensure plugins directory exists
        if not os.path.exists(self.plugins_dir):
            logger.warning(f"Plugins directory not found: {self.plugins_dir}")
            return discovered
        
        # Search for plugin.json files
        for item in os.listdir(self.plugins_dir):
            plugin_dir = os.path.join(self.plugins_dir, item)
            
            if os.path.isdir(plugin_dir):
                plugin_json = os.path.join(plugin_dir, "plugin.json")
                
                if os.path.exists(plugin_json):
                    try:
                        with open(plugin_json, "r") as f:
                            metadata = json.load(f)
                        
                        discovered.append({
                            "name": metadata.get("name", item),
                            "version": metadata.get("version", "0.1.0"),
                            "description": metadata.get("description", ""),
                            "author": metadata.get("author", "Unknown"),
                            "path": plugin_dir,
                            "loaded": metadata.get("name", item) in self.plugins
                        })
                    except Exception as e:
                        logger.error(f"Error reading plugin metadata: {str(e)}")
        
        return discovered
    
    def load_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """Load a plugin by name."""
        # Check if already loaded
        if plugin_name in self.plugins:
            logger.info(f"Plugin already loaded: {plugin_name}")
            return self.plugins[plugin_name]
        
        # Find plugin directory
        plugin_dir = os.path.join(self.plugins_dir, plugin_name)
        
        if not os.path.exists(plugin_dir) or not os.path.isdir(plugin_dir):
            plugin_found = False
            
            # Try to find by plugin.json name field
            for item in os.listdir(self.plugins_dir):
                item_dir = os.path.join(self.plugins_dir, item)
                plugin_json = os.path.join(item_dir, "plugin.json")
                
                if os.path.exists(plugin_json) and os.path.isdir(item_dir):
                    try:
                        with open(plugin_json, "r") as f:
                            metadata = json.load(f)
                        
                        if metadata.get("name") == plugin_name:
                            plugin_dir = item_dir
                            plugin_found = True
                            break
                    except:
                        continue
            
            if not plugin_found:
                logger.error(f"Plugin not found: {plugin_name}")
                return None
        
        # Load plugin metadata
        plugin_json = os.path.join(plugin_dir, "plugin.json")
        
        if not os.path.exists(plugin_json):
            logger.error(f"Plugin metadata not found: {plugin_json}")
            return None
        
        try:
            with open(plugin_json, "r") as f:
                metadata_dict = json.load(f)
            
            metadata = PluginMetadata.from_dict(metadata_dict)
        except Exception as e:
            logger.error(f"Error loading plugin metadata: {str(e)}")
            return None
        
        # Check for main module
        main_module = os.path.join(plugin_dir, "main.py")
        
        if not os.path.exists(main_module):
            logger.error(f"Plugin main module not found: {main_module}")
            return None
        
        # Check dependencies
        missing_deps = []
        for dep in metadata.dependencies:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(dep)
        
        if missing_deps:
            logger.error(f"Missing dependencies for plugin {plugin_name}: {', '.join(missing_deps)}")
            return None
        
        # Load the plugin module
        try:
            # Create a unique module name
            module_name = f"plugin_{metadata.name.replace('-', '_')}"
            
            # Load the module
            spec = importlib.util.spec_from_file_location(module_name, main_module)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module  # Add to sys.modules to make imports work
            spec.loader.exec_module(module)
            
            # Get tools from the module
            if not hasattr(module, "get_tools"):
                logger.error(f"Plugin {plugin_name} does not have a get_tools function")
                return None
            
            tools = module.get_tools()
            
            # Validate tools
            for tool in tools:
                try:
                    self.validator.validate_tool(tool)
                    self.validator.validate_tool_function(tool)
                except PluginValidationError as e:
                    logger.error(f"Tool validation error in plugin {plugin_name}: {str(e)}")
                    return None
            
            # Create plugin
            plugin = Plugin(metadata, module, tools, plugin_dir)
            self.plugins[plugin_name] = plugin
            
            logger.info(f"Loaded plugin: {plugin_name} v{metadata.version} by {metadata.author}")
            
            return plugin
        
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_name}: {str(e)}")
            return None
    
    def load_all_plugins(self) -> Dict[str, Plugin]:
        """Load all available plugins."""
        discovered = self.discover_plugins()
        
        for plugin_info in discovered:
            plugin_name = plugin_info["name"]
            if not plugin_info["loaded"]:
                self.load_plugin(plugin_name)
        
        return self.plugins
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin by name."""
        if plugin_name in self.plugins:
            # Remove from sys.modules if it was added
            module_name = f"plugin_{self.plugins[plugin_name].metadata.name.replace('-', '_')}"
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            # Remove from plugins dict
            del self.plugins[plugin_name]
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
        else:
            logger.warning(f"Plugin not loaded: {plugin_name}")
            return False
    
    def get_all_tools(self) -> List[Any]:
        """Get all tools from loaded plugins."""
        tools = []
        
        for plugin in self.plugins.values():
            tools.extend(plugin.get_tools())
        
        return tools
    
    def install_plugin(self, source: str) -> Optional[str]:
        """Install a plugin from a source (local zip file or URL)."""
        try:
            plugin_data = None
            plugin_name = None
            
            # Handle local file
            if os.path.exists(source) and source.endswith(".zip"):
                with zipfile.ZipFile(source, "r") as zip_ref:
                    # Extract to temp directory
                    temp_dir = tempfile.mkdtemp()
                    zip_ref.extractall(temp_dir)
                    
                    # Look for plugin.json
                    plugin_json = os.path.join(temp_dir, "plugin.json")
                    if not os.path.exists(plugin_json):
                        # Try looking in a subdirectory
                        for item in os.listdir(temp_dir):
                            sub_dir = os.path.join(temp_dir, item)
                            if os.path.isdir(sub_dir):
                                plugin_json = os.path.join(sub_dir, "plugin.json")
                                if os.path.exists(plugin_json):
                                    temp_dir = sub_dir
                                    break
                    
                    if os.path.exists(plugin_json):
                        with open(plugin_json, "r") as f:
                            plugin_data = json.load(f)
                            plugin_name = plugin_data.get("name")
                    
                    if not plugin_name:
                        logger.error("Invalid plugin: plugin.json not found or missing name")
                        shutil.rmtree(temp_dir)
                        return None
                    
                    # Copy to plugins directory
                    plugin_dir = os.path.join(self.plugins_dir, plugin_name)
                    if os.path.exists(plugin_dir):
                        logger.info(f"Removing existing plugin: {plugin_name}")
                        shutil.rmtree(plugin_dir)
                    
                    shutil.copytree(temp_dir, plugin_dir)
                    shutil.rmtree(temp_dir)
            
            # Handle URL
            elif source.startswith(("http://", "https://")) and source.endswith(".zip"):
                # Download to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
                    response = requests.get(source, stream=True)
                    response.raise_for_status()
                    
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)
                    
                    temp_file_path = temp_file.name
                
                # Install from temp file
                plugin_name = self.install_plugin(temp_file_path)
                
                # Clean up
                os.unlink(temp_file_path)
                
                return plugin_name
            
            else:
                logger.error(f"Invalid plugin source: {source}")
                return None
            
            logger.info(f"Installed plugin: {plugin_name}")
            return plugin_name
        
        except Exception as e:
            logger.error(f"Error installing plugin: {str(e)}")
            return None
    
    def uninstall_plugin(self, plugin_name: str) -> bool:
        """Uninstall a plugin by name."""
        # Unload the plugin first
        if plugin_name in self.plugins:
            self.unload_plugin(plugin_name)
        
        # Find plugin directory
        plugin_dir = os.path.join(self.plugins_dir, plugin_name)
        
        if not os.path.exists(plugin_dir) or not os.path.isdir(plugin_dir):
            # Try to find by plugin.json name field
            for item in os.listdir(self.plugins_dir):
                item_dir = os.path.join(self.plugins_dir, item)
                plugin_json = os.path.join(item_dir, "plugin.json")
                
                if os.path.exists(plugin_json) and os.path.isdir(item_dir):
                    try:
                        with open(plugin_json, "r") as f:
                            metadata = json.load(f)
                        
                        if metadata.get("name") == plugin_name:
                            plugin_dir = item_dir
                            break
                    except:
                        continue
        
        if os.path.exists(plugin_dir) and os.path.isdir(plugin_dir):
            try:
                shutil.rmtree(plugin_dir)
                logger.info(f"Uninstalled plugin: {plugin_name}")
                return True
            except Exception as e:
                logger.error(f"Error removing plugin directory: {str(e)}")
                return False
        else:
            logger.error(f"Plugin directory not found: {plugin_name}")
            return False
    
    def list_available_plugins(self) -> List[Dict[str, Any]]:
        """List available plugins from the registry."""
        if not self.plugin_registry_url:
            logger.warning("Plugin registry URL not set")
            return []
        
        try:
            response = requests.get(self.plugin_registry_url)
            response.raise_for_status()
            
            plugins = response.json()
            
            # Compare with installed plugins
            installed_plugins = self.discover_plugins()
            installed_names = [p["name"] for p in installed_plugins]
            
            for plugin in plugins:
                plugin["installed"] = plugin["name"] in installed_names
            
            return plugins
        
        except Exception as e:
            logger.error(f"Error listing available plugins: {str(e)}")
            return []

# Tool discovery functions
def discover_tools_from_module(module: Any) -> List[Any]:
    """Discover tools from a module based on naming convention."""
    from anthropic_agent import Tool, ToolParameter
    
    tools = []
    
    # Find all Tool instances in the module
    for name, obj in inspect.getmembers(module):
        if isinstance(obj, Tool):
            tools.append(obj)
    
    # Find functions that start with "tool_" or end with "_tool"
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if name.startswith("tool_") or name.endswith("_tool"):
            # Get function signature
            sig = inspect.signature(obj)
            parameters = []
            
            # Create parameters from function signature
            for param_name, param in sig.parameters.items():
                # Determine parameter type
                param_type = "string"  # Default
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation in [str, "str"]:
                        param_type = "string"
                    elif param.annotation in [int, "int"]:
                        param_type = "integer"
                    elif param.annotation in [float, "float"]:
                        param_type = "number"
                    elif param.annotation in [bool, "bool"]:
                        param_type = "boolean"
                    elif param.annotation in [list, "list"]:
                        param_type = "array"
                    elif param.annotation in [dict, "dict"]:
                        param_type = "object"
                
                # Determine if required
                required = param.default == inspect.Parameter.empty
                
                # Create parameter
                parameter = ToolParameter(
                    name=param_name,
                    type=param_type,
                    description=f"Parameter: {param_name}",
                    required=required,
                    default=None if param.default == inspect.Parameter.empty else param.default
                )
                
                parameters.append(parameter)
            
            # Create a Tool from the function
            tool = Tool(
                name=name,
                description=obj.__doc__ or f"Tool: {name}",
                parameters=parameters,
                function=obj,
                category="plugin"
            )
            
            tools.append(tool)
    
    return tools

# Example plugin structure
def create_plugin_template(plugin_dir: str, plugin_name: str, author: str, description: str) -> bool:
    """Create a plugin template."""
    try:
        # Create plugin directory
        plugin_path = os.path.join(plugin_dir, plugin_name)
        os.makedirs(plugin_path, exist_ok=True)
        
        # Create plugin.json
        plugin_json = {
            "name": plugin_name,
            "version": "0.1.0",
            "description": description,
            "author": author,
            "dependencies": []
        }
        
        with open(os.path.join(plugin_path, "plugin.json"), "w") as f:
            json.dump(plugin_json, f, indent=2)
        
        # Create main.py
        main_py = """
from typing import List, Dict, Any
from anthropic_agent import Tool, ToolParameter

async def example_tool(text: str) -> Dict[str, Any]:
    \"\"\"An example tool that echoes the input text.\"\"\"
    return {
        "echo": text,
        "status": "success"
    }

def get_tools() -> List[Tool]:
    \"\"\"Get tools provided by this plugin.\"\"\"
    return [
        Tool(
            name="example_tool",
            description="An example tool that echoes the input text",
            parameters=[
                ToolParameter(name="text", type="string", description="Text to echo")
            ],
            function=example_tool,
            category="example"
        )
    ]
"""
        
        with open(os.path.join(plugin_path, "main.py"), "w") as f:
            f.write(main_py.strip())
        
        # Create README.md
        readme_md = f"""# {plugin_name}

{description}

## Author

{author}

## Usage

This plugin provides the following tools:

- `example_tool`: An example tool that echoes the input text

## Development

To modify this plugin, edit the `main.py` file and update the `get_tools()` function to return your custom tools.

## License

MIT
"""
        
        with open(os.path.join(plugin_path, "README.md"), "w") as f:
            f.write(readme_md)
        
        return True
    
    except Exception as e:
        logger.error(f"Error creating plugin template: {str(e)}")
        return False