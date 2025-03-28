def plugin_commands(self, args: Optional[List[str]] = None):
    """Handle plugin-related commands."""
    if not args or args[0] == "help":
        print("\nPlugin Commands:")
        print("  plugins list                    - List installed plugins")
        print("  plugins available               - List available plugins from registry")
        print("  plugins load <name>             - Load a plugin")
        print("  plugins unload <name>           - Unload a plugin")
        print("  plugins install <source>        - Install a plugin (local zip or URL)")
        print("  plugins uninstall <name>        - Uninstall a plugin")
        print("  plugins create <name> <author>  - Create a plugin template")
        print("  plugins info <name>             - Show plugin information")
        print("  plugins reload <name>           - Reload a plugin")
        print("  plugins tools                   - List all tools from loaded plugins")
        return
    
    # Initialize plugin manager if not already done
    if not hasattr(self, 'plugin_manager'):
        from plugin_framework import PluginManager
        self.plugin_manager = PluginManager()
    
    command = args[0]
    
    if command == "list":
        plugins = self.plugin_manager.discover_plugins()
        
        if not plugins:
            print("No plugins installed.")
            return
        
        print("\nInstalled Plugins:")
        for plugin in plugins:
            loaded = plugin["loaded"]
            status = "loaded" if loaded else "not loaded"
            print(f"  {plugin['name']} v{plugin['version']} by {plugin['author']} ({status})")
            print(f"    {plugin['description']}")
            print(f"    Path: {plugin['path']}")
            print()
    
    elif command == "available":
        plugins = self.plugin_manager.list_available_plugins()
        
        if not plugins:
            print("No plugins available from registry or registry URL not set.")
            return
        
        print("\nAvailable Plugins:")
        for plugin in plugins:
            installed = plugin.get("installed", False)
            status = "installed" if installed else "not installed"
            print(f"  {plugin['name']} v{plugin['version']} by {plugin['author']} ({status})")
            print(f"    {plugin['description']}")
            print()
    
    elif command == "load" and len(args) >= 2:
        plugin_name = args[1]
        plugin = self.plugin_manager.load_plugin(plugin_name)
        
        if plugin:
            # Register tools with agent
            self.agent.register_tools(plugin.get_tools())
            print(f"Loaded plugin: {plugin.metadata.name} v{plugin.metadata.version} by {plugin.metadata.author}")
            print(f"Registered {len(plugin.get_tools())} tools")
        else:
            print(f"Failed to load plugin: {plugin_name}")
    
    elif command == "unload" and len(args) >= 2:
        plugin_name = args[1]
        success = self.plugin_manager.unload_plugin(plugin_name)
        
        if success:
            print(f"Unloaded plugin: {plugin_name}")
            print("Note: Tools from unloaded plugins remain registered with the agent until restart.")
        else:
            print(f"Failed to unload plugin: {plugin_name}")
    
    elif command == "install" and len(args) >= 2:
        source = args[1]
        plugin_name = self.plugin_manager.install_plugin(source)
        
        if plugin_name:
            print(f"Installed plugin: {plugin_name}")
            print("Use 'plugins load <name>' to load the plugin")
        else:
            print(f"Failed to install plugin from: {source}")
    
    elif command == "uninstall" and len(args) >= 2:
        plugin_name = args[1]
        success = self.plugin_manager.uninstall_plugin(plugin_name)
        
        if success:
            print(f"Uninstalled plugin: {plugin_name}")
        else:
            print(f"Failed to uninstall plugin: {plugin_name}")
    
    elif command == "create" and len(args) >= 3:
        plugin_name = args[1]
        author = args[2]
        description = " ".join(args[3:]) if len(args) > 3 else f"A plugin named {plugin_name}"
        
        from plugin_framework import create_plugin_template
        success = create_plugin_template(self.plugin_manager.plugins_dir, plugin_name, author, description)
        
        if success:
            print(f"Created plugin template: {plugin_name}")
            print(f"Plugin directory: {os.path.join(self.plugin_manager.plugins_dir, plugin_name)}")
        else:
            print(f"Failed to create plugin template: {plugin_name}")
    
    elif command == "info" and len(args) >= 2:
        plugin_name = args[1]
        
        if plugin_name in self.plugin_manager.plugins:
            plugin = self.plugin_manager.plugins[plugin_name]
            metadata = plugin.metadata
            
            print(f"\nPlugin: {metadata.name} v{metadata.version}")
            print(f"Description: {metadata.description}")
            print(f"Author: {metadata.author}")
            
            if metadata.homepage:
                print(f"Homepage: {metadata.homepage}")
            
            if metadata.repository:
                print(f"Repository: {metadata.repository}")
            
            if metadata.license:
                print(f"License: {metadata.license}")
            
            if metadata.dependencies:
                print(f"Dependencies: {', '.join(metadata.dependencies)}")
            
            print(f"\nTools: {len(plugin.get_tools())}")
            for tool in plugin.get_tools():
                print(f"  {tool.name} - {tool.description}")
        else:
            print(f"Plugin not loaded: {plugin_name}")
            print("Use 'plugins load <name>' to load the plugin")
    
    elif command == "reload" and len(args) >= 2:
        plugin_name = args[1]
        
        if plugin_name in self.plugin_manager.plugins:
            plugin = self.plugin_manager.plugins[plugin_name]
            success = plugin.reload()
            
            if success:
                print(f"Reloaded plugin: {plugin_name}")
            else:
                print(f"Failed to reload plugin: {plugin_name}")
                if plugin.errors:
                    print("Errors:")
                    for error in plugin.errors:
                        print(f"  {error}")
        else:
            print(f"Plugin not loaded: {plugin_name}")
    
    elif command == "tools":
        tools = self.plugin_manager.get_all_tools()
        
        if not tools:
            print("No tools available from plugins.")
            return
        
        print("\nTools from Plugins:")
        for tool in tools:
            print(f"  {tool.name} - {tool.description}")
            print(f"    Category: {tool.category}")
            
            for param in tool.parameters:
                required = " (required)" if param.required else " (optional)"
                default = f", default: {param.default}" if param.default is not None and not param.required else ""
                print(f"    - {param.name}: {param.type}{required}{default} - {param.description}")
            
            print()
    
    else:
        print(f"Unknown plugin command: {command}")
        print("Use 'plugins help' to see available commands.")