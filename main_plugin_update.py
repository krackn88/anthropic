def setup_agent(model=None, use_rag=False, load_plugins=True):
    """Set up the agent with all available tools."""
    print("Initializing Anthropic Agent...")
    
    # Import here to avoid circular imports
    from anthropic_agent import Agent
    from github_tools import get_github_tools
    from claude_tools import get_claude_tools
    from system_tools import get_system_tools
    from anthropic_cookbook import get_cookbook_tools
    
    # Create agent with specified model or default
    agent = Agent(model=model if model else "claude-3-opus-20240229", use_rag=use_rag)
    
    # Register tools
    print("Loading tools...")
    
    github_tools = get_github_tools()
    agent.register_tools(github_tools)
    print(f"Registered {len(github_tools)} GitHub tools")
    
    claude_tools = get_claude_tools()
    agent.register_tools(claude_tools)
    print(f"Registered {len(claude_tools)} Claude language tools")
    
    system_tools = get_system_tools()
    agent.register_tools(system_tools)
    print(f"Registered {len(system_tools)} system tools")
    
    cookbook_tools = get_cookbook_tools()
    agent.register_tools(cookbook_tools)
    print(f"Registered {len(cookbook_tools)} cookbook tools")
    
    # Register RAG tools if enabled
    if use_rag:
        from rag import get_rag_tools
        rag_tools = get_rag_tools()
        agent.register_tools(rag_tools)
        print(f"Registered {len(rag_tools)} RAG tools")
    
    # Register image tools
    from image_processing import get_image_tools
    image_tools = get_image_tools()
    agent.register_tools(image_tools)
    print(f"Registered {len(image_tools)} image tools")
    
    # Load plugins if enabled
    if load_plugins:
        print("Loading plugins...")
        try:
            from plugin_framework import PluginManager
            plugin_manager = PluginManager()
            plugins = plugin_manager.load_all_plugins()
            
            for plugin_name, plugin in plugins.items():
                plugin_tools = plugin.get_tools()
                agent.register_tools(plugin_tools)
                print(f"Registered {len(plugin_tools)} tools from plugin: {plugin_name}")
            
        except Exception as e:
            logger.error(f"Error loading plugins: {str(e)}")
            print(f"Warning: Failed to load plugins: {str(e)}")
    
    print(f"Total tools available: {len(agent.tools)}")
    print()
    
    return agent

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Anthropic-powered Agent CLI")
    
    parser.add_argument("--model", type=str, default=None,
                      help="Claude model to use (default: claude-3-opus-20240229)")
    
    parser.add_argument("--no-streaming", action="store_true", default=False,
                      help="Disable streaming responses")
    
    parser.add_argument("--structured", type=str, default=None,
                      help="Use structured output with the specified schema name")
    
    parser.add_argument("--query", type=str, default=None,
                      help="Query to process (when using with --structured)")
    
    parser.add_argument("--output", type=str, default=None,
                      help="Output file for structured response (JSON)")
    
    parser.add_argument("--use-rag", action="store_true", default=False,
                      help="Enable Retrieval-Augmented Generation (RAG)")
    
    parser.add_argument("--rag-command", action="store_true", default=False,
                      help="Run the RAG CLI instead of the agent CLI")
    
    