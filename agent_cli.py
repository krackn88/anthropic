"""
Command Line Interface for the Anthropic-powered Agent
"""

import os
import sys
import json
import shlex
import asyncio
import signal
import readline
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgentCLI:
    """Command Line Interface for the Anthropic Agent."""
    
    def __init__(self, agent, use_streaming: bool = True):
        """Initialize with an agent."""
        self.agent = agent
        self.running = True
        self.history = []
        self.use_streaming = use_streaming
        self.commands = {
            "help": self.show_help,
            "exit": self.exit,
            "quit": self.exit,
            "tools": self.list_tools,
            "clear": self.clear_screen,
            "history": self.show_history,
            "save": self.save_conversation,
            "load": self.load_conversation,
            "github": self.github_commands,
            "execute": self.execute_tool,
            "streaming": self.toggle_streaming,
            "models": self.model_commands,
            "usage": self.show_usage_stats,
            "structured": self.structured_commands,
            "rag": self.rag_commands,
            "image": self.image_commands
        }
    
    def run(self):
        """Run the CLI."""
        # Set up signal handler for graceful exit
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print("\n=== Anthropic-Powered Agent CLI ===")
        print("Type 'help' to see available commands. Type 'exit' to quit.")
        
        while self.running:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Parse as command or query
                parts = shlex.split(user_input)
                command = parts[0].lower()
                
                if command in self.commands:
                    self.commands[command](parts[1:] if len(parts) > 1 else None)
                else:
                    self.process_query(user_input)
            
            except EOFError:
                self.exit()
            except KeyboardInterrupt:
                print("\nKeyboard interrupt.")
                continue
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                print(f"Error: {str(e)}")
    
    def signal_handler(self, sig, frame):
        """Handle signals."""
        print("\nReceived signal to terminate. Exiting gracefully...")
        self.exit()
    
    def show_help(self, args: Optional[List[str]] = None):
        """Show help information."""
        print("\n===== Anthropic Agent CLI Help =====")
        print("\nAvailable Commands:")
        print("  help              - Show this help message")
        print("  exit, quit        - Exit the program")
        print("  tools [category]  - List available tools, optionally filtered by category")
        print("  clear             - Clear the screen")
        print("  history [n]       - Show conversation history (last n messages)")
        print("  save <filename>   - Save conversation to a file")
        print("  load <filename>   - Load conversation from a file")
        print("  github help       - Show GitHub commands")
        print("  execute <tool>    - Execute a tool directly")
        print("  streaming on|off  - Toggle streaming mode")
        print("  models            - Show model information and commands")
        print("  usage             - Show current usage statistics")
        print("  structured        - Show structured output commands")
        print("  image             - Show image processing commands")
        
        if self.agent.use_rag:
            print("  rag               - Show RAG system commands")
        
        print("\nAny other input will be processed as a query to the agent.")
        print("\n===================================\n")
    
    def exit(self, args: Optional[List[str]] = None):
        """Exit the program."""
        print("Exiting. Goodbye!")
        self.running = False
        sys.exit(0)
    
    def list_tools(self, args: Optional[List[str]] = None):
        """List available tools."""
        category_filter = args[0] if args else None
        
        if category_filter and category_filter not in self.agent.tool_categories:
            print(f"Unknown category: {category_filter}")
            print("Available categories: " + ", ".join(self.agent.tool_categories.keys()))
            return
        
        print("\nAvailable Tools:")
        
        if category_filter:
            print(f"\nCategory: {category_filter}")
            for tool_name in self.agent.tool_categories[category_filter]:
                tool = self.agent.tools[tool_name]
                print(f"  {tool_name} - {tool.description}")
                for param in tool.parameters:
                    required = " (required)" if param.required else " (optional)"
                    default = f", default: {param.default}" if param.default is not None and not param.required else ""
                    print(f"    - {param.name}: {param.type}{required}{default} - {param.description}")
        else:
            for category, tool_names in self.agent.tool_categories.items():
                print(f"\nCategory: {category}")
                for tool_name in tool_names:
                    tool = self.agent.tools[tool_name]
                    print(f"  {tool_name} - {tool.description}")
    
    def clear_screen(self, args: Optional[List[str]] = None):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def show_history(self, args: Optional[List[str]] = None):
        """Show conversation history."""
        try:
            n = int(args[0]) if args else len(self.history)
        except (ValueError, IndexError):
            n = len(self.history)
        
        print("\nConversation History:")
        for i, (role, message) in enumerate(self.history[-n:]):
            print(f"\n[{i+1}] {role.capitalize()}:")
            print(message)
    
    def save_conversation(self, args: Optional[List[str]] = None):
        """Save conversation to a file."""
        if not args:
            print("Error: Please specify a filename.")
            return
        
        filename = args[0]
        
        try:
            # Add timestamp and model info
            data = {
                "timestamp": datetime.now().isoformat(),
                "model": self.agent.model,
                "history": self.history,
                "usage_stats": self.agent.get_usage_stats()
            }
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"Conversation saved to {filename}")
        
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")
            print(f"Error saving conversation: {str(e)}")
    
    def load_conversation(self, args: Optional[List[str]] = None):
        """Load conversation from a file."""
        if not args:
            print("Error: Please specify a filename.")
            return
        
        filename = args[0]
        
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            if "history" in data:
                self.history = data["history"]
                
                # Update agent memory
                self.agent.memory.clear()
                for role, content in self.history:
                    self.agent.memory.add(role, content)
                
                print(f"Conversation loaded from {filename}")
                
                # Show model info if available
                if "model" in data:
                    print(f"Model: {data['model']}")
                if "timestamp" in data:
                    print(f"Original timestamp: {data['timestamp']}")
            else:
                print("Error: Invalid conversation file format.")
        
        except Exception as e:
            logger.error(f"Error loading conversation: {str(e)}")
            print(f"Error loading conversation: {str(e)}")
    
    def github_commands(self, args: Optional[List[str]] = None):
        """Handle GitHub-related commands."""
        if not args or args[0] == "help":
            print("\nGitHub Commands:")
            print("\n--- Basic GitHub Commands ---")
            print("  github repo <owner> <repo>           - Get information about a repository")
            print("  github file <owner> <repo> <path>    - Get file contents")
            print("  github search <query>                - Search code")
            print("  github issues <owner> <repo>         - List issues in a repository")
            print("  github prs <owner> <repo>            - List pull requests in a repository")
            
            print("\n--- GitHub Actions Commands ---")
            print("  github workflows <owner> <repo>      - List workflows in a repository")
            print("  github workflow <owner> <repo> <id>  - Get a specific workflow")
            print("  github runs <owner> <repo> [wf_id]   - List workflow runs")
            print("  github trigger <owner> <repo> <id> <ref> - Trigger a workflow")
            print("  github visualize-runs <owner> <repo> <id> - Visualize workflow history")
            
            print("\n--- PR Management Commands ---")
            print("  github analyze-pr <owner> <repo> <number> - Analyze a pull request")
            print("  github create-pr <owner> <repo> <title> <head> <base> - Create a pull request")
            
            print("\n--- Code Quality Commands ---")
            print("  github code-quality <owner> <repo> <path> - Analyze code complexity")
            print("  github dependencies <owner> <repo>    - Scan dependencies")
            
            print("\n--- Repository Visualization Commands ---")
            print("  github commit-history <owner> <repo> [days] - Visualize commit history")
            print("  github contributors <owner> <repo>   - Visualize contributors")
            print("  github languages <owner> <repo>      - Visualize language distribution")
            return
        
        command = args[0]
        
        # Basic GitHub commands
        if command == "repo" and len(args) >= 3:
            owner, repo = args[1], args[2]
            self.execute_tool(["get_repo_info", f"owner={owner}", f"repo={repo}"])
        
        elif command == "file" and len(args) >= 4:
            owner, repo, path = args[1], args[2], args[3]
            ref = args[4] if len(args) >= 5 else None
            ref_param = f"ref={ref}" if ref else ""
            self.execute_tool(["get_file_contents", f"owner={owner}", f"repo={repo}", f"path={path}", ref_param])
        
        elif command == "search" and len(args) >= 2:
            query = args[1]
            self.execute_tool(["search_code", f"query={query}"])
        
        elif command == "issues" and len(args) >= 3:
            owner, repo = args[1], args[2]
            state = args[3] if len(args) >= 4 else "open"
            self.execute_tool(["list_issues", f"owner={owner}", f"repo={repo}", f"state={state}"])
        
        elif command == "prs" and len(args) >= 3:
            owner, repo = args[1], args[2]
            state = args[3] if len(args) >= 4 else "open"
            self.execute_tool(["list_pull_requests", f"owner={owner}", f"repo={repo}", f"state={state}"])
        
        # GitHub Actions commands
        elif command == "workflows" and len(args) >= 3:
            owner, repo = args[1], args[2]
            self.execute_tool(["list_github_workflows", f"owner={owner}", f"repo={repo}"])
        
        elif command == "workflow" and len(args) >= 4:
            owner, repo, workflow_id = args[1], args[2], args[3]
            self.execute_tool(["get_github_workflow", f"owner={owner}", f"repo={repo}", f"workflow_id={workflow_id}"])
        
        elif command == "runs" and len(args) >= 3:
            owner, repo = args[1], args[2]
            workflow_id = args[3] if len(args) >= 4 else None
            status = args[4] if len(args) >= 5 else None
            
            tool_args = [f"owner={owner}", f"repo={repo}"]
            if workflow_id:
                tool_args.append(f"workflow_id={workflow_id}")
            if status:
                tool_args.append(f"status={status}")
            
            self.execute_tool(["list_workflow_runs"] + tool_args)
        
        elif command == "trigger" and len(args) >= 5:
            owner, repo, workflow_id, ref = args[1], args[2], args[3], args[4]
            self.execute_tool(["trigger_workflow", f"owner={owner}", f"repo={repo}", f"workflow_id={workflow_id}", f"ref={ref}"])
        
        elif command == "visualize-runs" and len(args) >= 4:
            owner, repo, workflow_id = args[1], args[2], args[3]
            self.execute_tool(["visualize_workflow_history", f"owner={owner}", f"repo={repo}", f"workflow_id={workflow_id}"])
        
        # PR Management commands
        elif command == "analyze-pr" and len(args) >= 4:
            owner, repo, pull_number = args[1], args[2], args[3]
            self.execute_tool(["analyze_pull_request", f"owner={owner}", f"repo={repo}", f"pull_number={pull_number}"])
        
        elif command == "create-pr" and len(args) >= 6:
            owner, repo, title, head, base = args[1], args[2], args[3], args[4], args[5]
            
            # Optionally get body and draft status
            body = None
            draft = False
            
            for i in range(6, len(args)):
                if args[i].startswith("body="):
                    body = args[i][5:]
                elif args[i] == "draft":
                    draft = True
            
            tool_args = [
                f"owner={owner}", 
                f"repo={repo}", 
                f"title={title}", 
                f"head={head}", 
                f"base={base}"
            ]
            
            if body:
                tool_args.append(f"body={body}")
            
            if draft:
                tool_args.append(f"draft=true")
            
            self.execute_tool(["create_pull_request"] + tool_args)
        
        # Code Quality commands
        elif command == "code-quality" and len(args) >= 4:
            owner, repo, path = args[1], args[2], args[3]
            ref = args[4] if len(args) >= 5 else None
            
            tool_args = [f"owner={owner}", f"repo={repo}", f"path={path}"]
            if ref:
                tool_args.append(f"ref={ref}")
            
            self.execute_tool(["analyze_code_complexity"] + tool_args)
        
        elif command == "dependencies" and len(args) >= 3:
            owner, repo = args[1], args[2]
            ref = args[3] if len(args) >= 4 else None
            
            tool_args = [f"owner={owner}", f"repo={repo}"]
            if ref:
                tool_args.append(f"ref={ref}")
            
            self.execute_tool(["scan_dependencies"] + tool_args)
        
        # Repository visualization commands
        elif command == "commit-history" and len(args) >= 3:
            owner, repo = args[1], args[2]
            days = args[3] if len(args) >= 4 else "30"
            
            self.execute_tool(["visualize_commit_history", f"owner={owner}", f"repo={repo}", f"days={days}"])
        
        elif command == "contributors" and len(args) >= 3:
            owner, repo = args[1], args[2]
            self.execute_tool(["visualize_contributors", f"owner={owner}", f"repo={repo}"])
        
        elif command == "languages" and len(args) >= 3:
            owner, repo = args[1], args[2]
            self.execute_tool(["visualize_languages", f"owner={owner}", f"repo={repo}"])
        
        else:
            print(f"Unknown GitHub command: {command}")
            print("Use 'github help' to see available commands.")
    
    def execute_tool(self, args: Optional[List[str]] = None):
        """Execute a tool directly."""
        if not args or not args[0]:
            print("Error: Please specify a tool name and parameters.")
            print("Usage: execute <tool_name> param1=value1 param2=value2 ...")
            return
        
        tool_name = args[0]
        
        if tool_name not in self.agent.tools:
            print(f"Error: Tool '{tool_name}' not found.")
            print("Use 'tools' command to see available tools.")
            return
        
        # Parse parameters
        params = {}
        for arg in args[1:]:
            if "=" in arg:
                key, value = arg.split("=", 1)
                # Try to parse as JSON if it looks like one
                if (value.startswith('{') and value.endswith('}')) or \
                   (value.startswith('[') and value.endswith(']')):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                params[key] = value
        
        # Get tool
        tool = self.agent.tools[tool_name]
        
        # Check required parameters
        missing_params = []
        for param in tool.parameters:
            if param.required and param.name not in params:
                missing_params.append(param.name)
        
        if missing_params:
            print(f"Error: Missing required parameters: {', '.join(missing_params)}")
            return
        
        # Add default values for missing optional parameters
        for param in tool.parameters:
            if not param.required and param.name not in params and param.default is not None:
                params[param.name] = param.default
        
        # Execute tool
        print(f"Executing tool: {tool_name}")
        print(f"Parameters: {json.dumps(params, indent=2)}")
        
        try:
            if asyncio.iscoroutinefunction(tool.function):
                result = asyncio.run(tool.function(**params))
            else:
                result = tool.function(**params)
            
            print("\nResult:")
            print(json.dumps(result, indent=2))
        
        except Exception as e:
            logger.error(f"Error executing tool: {str(e)}")
            print(f"Error executing tool: {str(e)}")
    
    def toggle_streaming(self, args: Optional[List[str]] = None):
        """Toggle streaming mode."""
        if args and args[0].lower() in ["on", "true", "1", "yes"]:
            self.use_streaming = True
            print("Streaming mode enabled.")
        elif args and args[0].lower() in ["off", "false", "0", "no"]:
            self.use_streaming = False
            print("Streaming mode disabled.")
        else:
            self.use_streaming = not self.use_streaming
            print(f"Streaming mode {'enabled' if self.use_streaming else 'disabled'}.")
    
    def model_commands(self, args: Optional[List[str]] = None):
        """Handle model-related commands."""
        if not args or args[0] == "help":
            print("\nModel Commands:")
            print("  models list             - List available models")
            print("  models info <model_id>  - Show detailed model information")
            print("  models current          - Show current model information")
            print("  models set <model_id>   - Switch to a different model")
            print("  models recommend <task> - Get model recommendations for a task")
            return
        
        command = args[0]
        
        if command == "list":
            from model_config import list_available_models
            models = list_available_models()
            
            print("\nAvailable Models:")
            for model in models:
                current = " (current)" if model["id"] == self.agent.model else ""
                print(f"  {model['id']} - {model['name']}{current}")
                print(f"    {model['description']}")
        
        elif command == "info" and len(args) >= 2:
            model_id = args[1]
            from model_config import get_model_info
            model_info = get_model_info(model_id)
            
            if model_info:
                print(f"\nModel: {model_info['name']} ({model_info['id']})")
                print(f"Description: {model_info['description']}")
                print(f"Context Window: {model_info['context_window']} tokens")
                print(f"Training Data: {model_info['training_data']}")
                print("\nCapabilities:")
                for capability, supported in model_info["capabilities"].items():
                    print(f"  {capability}: {'✓' if supported else '✗'}")
                
                print("\nUse Cases:")
                for use_case in model_info["use_cases"]:
                    print(f"  - {use_case}")
                
                print("\nPricing:")
                print(f"  Input: ${model_info['pricing']['input_per_million'] / 1000000:.6f} per token")
                print(f"  Output: ${model_info['pricing']['output_per_million'] / 1000000:.6f} per token")
            else:
                print(f"Error: Model '{model_id}' not found.")
        
        elif command == "current":
            from model_config import get_model_info
            model_info = get_model_info(self.agent.model)
            
            if model_info:
                print(f"\nCurrent Model: {model_info['name']} ({model_info['id']})")
                print(f"Description: {model_info['description']}")
                print(f"Context Window: {model_info['context_window']} tokens")
            else:
                print(f"Current Model: {self.agent.model}")
        
        elif command == "set" and len(args) >= 2:
            new_model = args[1]
            from model_config import get_model_info
            model_info = get_model_info(new_model)
            
            if model_info:
                self.agent.model = new_model
                print(f"Model changed to {model_info['name']} ({new_model}).")
            else:
                print(f"Error: Model '{new_model}' not found.")
        
        elif command == "recommend" and len(args) >= 2:
            task_description = " ".join(args[1:])
            from model_config import recommend_model
            recommendations = recommend_model(task_description)
            
            print(f"\nRecommended models for: {task_description}")
            for i, rec in enumerate(recommendations):
                print(f"\n{i+1}. {rec['model']['name']} ({rec['model']['id']})")
                print(f"   Suitability: {rec['suitability']} / 10")
                print(f"   Reason: {rec['reason']}")
        
        else:
            print(f"Unknown model command: {command}")
            print("Use 'models help' to see available commands.")
    
    def show_usage_stats(self, args: Optional[List[str]] = None):
        """Show current usage statistics."""
        stats = self.agent.get_usage_stats()
        
        print("\nUsage Statistics:")
        print(f"API Calls: {stats['api_calls']}")
        print(f"Total Prompt Tokens: {stats['total_prompt_tokens']:,}")
        print(f"Total Completion Tokens: {stats['total_completion_tokens']:,}")
        print(f"Total Tokens: {stats['total_tokens']:,}")
        print(f"Estimated Cost: ${stats['estimated_cost_usd']:.4f} USD")
        
        if stats['last_call_timestamp']:
            print(f"Last Call: {stats['last_call_timestamp']}")
    
    def structured_commands(self, args: Optional[List[str]] = None):
        """Handle structured output commands."""
        if not args or args[0] == "help":
            print("\nStructured Output Commands:")
            print("  structured list               - List available schemas")
            print("  structured analyze <schema>   - Process query with structured output")
            print("  structured save <file>        - Save last structured result to a file")
            return
        
        command = args[0]
        
        if command == "list":
            from structured_schemas import list_available_schemas
            schemas = list_available_schemas()
            
            print("\nAvailable Schemas:")
            for schema in schemas:
                print(f"  {schema['id']} - {schema['name']}")
                print(f"    {schema['description']}")
        
        elif command == "analyze" and len(args) >= 2:
            schema_id = args[1]
            from structured_schemas import get_schema_by_id
            schema = get_schema_by_id(schema_id)
            
            if not schema:
                print(f"Error: Schema '{schema_id}' not found.")
                return
            
            # Get user query
            query = input(f"Enter query for {schema['name']} analysis: ")
            
            print(f"Processing with schema: {schema['name']}...")
            result = asyncio.run(self.agent.get_structured_response(query, schema["schema"]))
            
            # Store result for potential saving
            self.last_structured_result = result
            
            print("\nResult:")
            print(json.dumps(result, indent=2))
        
        elif command == "save" and len(args) >= 2:
            filename = args[1]
            
            if hasattr(self, 'last_structured_result'):
                with open(filename, 'w') as f:
                    json.dump(self.last_structured_result, f, indent=2)
                print(f"Structured result saved to {filename}")
            else:
                print("Error: No structured result available to save.")
        
        else:
            print(f"Unknown structured command: {command}")
            print("Use 'structured help' to see available commands.")
    
    def rag_commands(self, args: Optional[List[str]] = None):
        """Handle RAG-related commands."""
        if not self.agent.use_rag:
            print("Error: RAG system is not enabled.")
            print("Start the agent with --use-rag to enable RAG capabilities.")
            return
        
        if not args or args[0] == "help":
            print("\nRAG Commands:")
            print("  rag add <file_path>       - Add a document to the RAG system")
            print("  rag query <query>         - Query the RAG system directly")
            print("  rag list                  - List all documents in the RAG system")
            print("  rag get <document_id>     - Get a specific document from the RAG system")
            print("  rag delete <document_id>  - Delete a document from the RAG system")
            print("  rag stats                 - Show RAG system statistics")
            return
        
        command = args[0]
        
        if command == "add" and len(args) >= 2:
            file_path = args[1]
            metadata = {}
            
            # Optional metadata as key=value pairs
            for arg in args[2:]:
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    metadata[key] = value
            
            try:
                result = asyncio.run(self.agent.rag_system.add_document_from_file(file_path, metadata))
                print("\nDocument added to RAG system:")
                print(f"Document ID: {result.id}")
                print(f"Chunks: {len(result.chunks)}")
                print(f"Metadata: {json.dumps(result.metadata, indent=2)}")
            except Exception as e:
                logger.error(f"Error adding document: {str(e)}")
                print(f"Error adding document: {str(e)}")
        
        elif command == "query" and len(args) >= 2:
            query = " ".join(args[1:])
            try:
                results = asyncio.run(self.agent.rag_system.query(query))
                print("\nRAG Query Results:")
                print(f"Query: {results['query']}")
                print(f"Found {len(results['results'])} results")
                
                for i, result in enumerate(results["results"]):
                    print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
                    print(f"Content: {result['content'][:200]}...")
                    print(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
            except Exception as e:
                logger.error(f"Error querying RAG system: {str(e)}")
                print(f"Error querying RAG system: {str(e)}")
        
        elif command == "list":
            try:
                documents = self.agent.rag_system.get_all_documents()
                print("\nDocuments in RAG System:")
                print(f"Total: {len(documents)}")
                
                for doc in documents:
                    print(f"\nID: {doc.id}")
                    print(f"Chunks: {len(doc.chunks)}")
                    print(f"Metadata: {json.dumps(doc.metadata, indent=2)}")
                    print(f"Content Preview: {doc.content[:100]}...")
            except Exception as e:
                logger.error(f"Error listing documents: {str(e)}")
                print(f"Error listing documents: {str(e)}")
        
        elif command == "get" and len(args) >= 2:
            document_id = args[1]
            try:
                document = self.agent.rag_system.get_document(document_id)
                if document:
                    print(f"\nDocument ID: {document.id}")
                    print(f"Chunks: {len(document.chunks)}")
                    print(f"Metadata: {json.dumps(document.metadata, indent=2)}")
                    
                    # Ask if user wants to see full content
                    show_content = input("\nShow full content? (y/n): ").lower().startswith('y')
                    if show_content:
                        print("\nContent:")
                        print(document.content)
                else:
                    print(f"Document {document_id} not found.")
            except Exception as e:
                logger.error(f"Error getting document: {str(e)}")
                print(f"Error getting document: {str(e)}")
        
        elif command == "delete" and len(args) >= 2:
            document_id = args[1]
            try:
                success = self.agent.rag_system.delete_document(document_id)
                if success:
                    print(f"Document {document_id} deleted successfully.")
                else:
                    print(f"Document {document_id} not found.")
            except Exception as e:
                logger.error(f"Error deleting document: {str(e)}")
                print(f"Error deleting document: {str(e)}")
        
        elif command == "stats":
            try:
                stats = self.agent.rag_system.get_stats()
                print("\nRAG System Statistics