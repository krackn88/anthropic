#!/usr/bin/env python3
"""
Main entry point for the Anthropic-powered Agent
"""

import os
import sys
import logging
import asyncio
import argparse
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check required environment variables."""
    required_vars = ["ANTHROPIC_API_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set these variables and try again.")
        print("For example:")
        print("  export ANTHROPIC_API_KEY=your_api_key")
        sys.exit(1)
    
    # Optional variables
    if not os.environ.get("GITHUB_TOKEN"):
        print("Warning: GITHUB_TOKEN environment variable not set.")
        print("GitHub API requests may be subject to rate limiting.")
        print("For better performance, consider setting this variable.")
        print()

def setup_agent(model=None, use_rag=False):
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
    
    parser.add_argument("--image", type=str, default=None,
                      help="Path to an image to analyze")
    
    parser.add_argument("--image-prompt", type=str, default=None,
                      help="Prompt to use when analyzing an image")
    
    parser.add_argument("--web", action="store_true", default=False,
                      help="Start the web API server")
    
    parser.add_argument("--web-host", type=str, default="127.0.0.1",
                      help="Host to listen on for web API server")
    
    parser.add_argument("--web-port", type=int, default=8000,
                      help="Port to listen on for web API server")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Check environment
    check_environment()
    
    # Start web API server if requested
    if args.web:
        import uvicorn
        print(f"Starting web API server on {args.web_host}:{args.web_port}")
        uvicorn.run("web_api:app", host=args.web_host, port=args.web_port, reload=False)
        return
    
    # Run RAG CLI if requested
    if args.rag_command:
        import rag_cli
        import sys
        sys.argv = sys.argv[1:]  # Remove the --rag-command argument
        asyncio.run(rag_cli.main())
        return
    
    # Set up agent with specified model
    agent = setup_agent(model=args.model, use_rag=args.use_rag)
    
    # Handle image analysis mode
    if args.image:
        if not os.path.exists(args.image):
            print(f"Error: Image file not found: {args.image}")
            sys.exit(1)
        
        prompt = args.image_prompt or "Please describe this image in detail."
        
        print(f"Analyzing image: {args.image}")
        print(f"Prompt: {prompt}")
        
        response = asyncio.run(agent.process_message_with_image(prompt, args.image))
        print("\nAnalysis:")
        print(response)
        return
    
    # Handle structured output mode
    if args.structured:
        # Import here to avoid circular imports
        from structured_schemas import get_schema_by_id
        
        schema = get_schema_by_id(args.structured)
        if not schema:
            print(f"Error: Unknown schema '{args.structured}'")
            print("Use --structured list to see available schemas")
            sys.exit(1)
        
        if not args.query:
            print("Error: Query is required when using structured output mode")
            print("Use --query \"Your query here\"")
            sys.exit(1)
        
        # Get the structured response
        print(f"Getting structured response using schema: {args.structured}")
        schema_class = schema["schema"]
        result = asyncio.run(agent.get_structured_response(args.query, schema_class))
        
        # Output the result
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"Result saved to: {args.output}")
        else:
            import json
            print(json.dumps(result, indent=2))
        
        return
    
    # Run CLI
    from agent_cli import AgentCLI
    cli = AgentCLI(agent, use_streaming=not args.no_streaming)
    cli.run()

if __name__ == "__main__":
    main()