# Additional method to add to the AgentCLI class

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