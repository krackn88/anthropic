# Continue from previous implementation
    def visualize_code_complexity(self, metrics: Dict[str, Any], output_path: str = "code_complexity.png") -> Dict[str, Any]:
        """Visualize code complexity metrics."""
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(12, 8))
            plt.suptitle(f"Code Complexity Analysis: {metrics.get('file', 'Unknown File')}", fontsize=16)
            
            # Line breakdown
            ax1 = plt.subplot(2, 2, 1)
            line_data = {
                "Code": metrics.get("metrics", {}).get("code_lines", 0),
                "Comments": metrics.get("metrics", {}).get("comment_lines", 0),
                "Empty": metrics.get("metrics", {}).get("empty_lines", 0)
            }
            
            if sum(line_data.values()) > 0:
                labels = list(line_data.keys())
                values = list(line_data.values())
                ax1.pie(values, labels=labels, autopct='%1.1f%%')
                ax1.set_title("Line Breakdown")
            else:
                ax1.text(0.5, 0.5, "No line data available", ha="center", va="center")
                ax1.axis("off")
            
            # Complexity metrics
            ax2 = plt.subplot(2, 2, 2)
            complexity_data = {
                "Functions": metrics.get("metrics", {}).get("function_count", 0),
                "Classes": metrics.get("metrics", {}).get("class_count", 0),
                "Nesting": metrics.get("metrics", {}).get("max_nesting_depth", 0),
            }
            
            if sum(complexity_data.values()) > 0:
                ax2.bar(complexity_data.keys(), complexity_data.values(), color="skyblue")
                ax2.set_title("Code Structure Metrics")
                ax2.set_ylabel("Count")
            else:
                ax2.text(0.5, 0.5, "No complexity data available", ha="center", va="center")
                ax2.axis("off")
            
            # Cyclomatic complexity
            ax3 = plt.subplot(2, 2, 3)
            cyclomatic = metrics.get("metrics", {}).get("avg_cyclomatic_complexity", 0)
            
            if cyclomatic > 0:
                # Create a gauge-like visualization
                complexity_ranges = [
                    {"range": (0, 5), "label": "Low", "color": "green"},
                    {"range": (5, 10), "label": "Medium", "color": "orange"},
                    {"range": (10, 100), "label": "High", "color": "red"}
                ]
                
                # Determine level
                level_color = "gray"
                level_label = "Unknown"
                for level in complexity_ranges:
                    if level["range"][0] <= cyclomatic < level["range"][1]:
                        level_color = level["color"]
                        level_label = level["label"]
                
                # Draw a gauge
                ax3.text(0.5, 0.5, f"{cyclomatic:.1f}", ha="center", va="center", fontsize=24)
                ax3.text(0.5, 0.2, f"Cyclomatic Complexity ({level_label})", ha="center", va="center")
                ax3.add_patch(plt.Circle((0.5, 0.5), 0.4, color=level_color, alpha=0.3))
                ax3.axis("off")
            else:
                ax3.text(0.5, 0.5, "No cyclomatic complexity data", ha="center", va="center")
                ax3.axis("off")
            
            # Overall complexity score
            ax4 = plt.subplot(2, 2, 4)
            complexity_score = metrics.get("metrics", {}).get("complexity_score", 0)
            complexity_level = metrics.get("metrics", {}).get("complexity_level", "Unknown")
            
            if complexity_score > 0:
                # Color mapping
                level_colors = {
                    "Low": "green",
                    "Medium": "orange",
                    "High": "red",
                    "Unknown": "gray"
                }
                
                # Create a simple visualization
                ax4.text(0.5, 0.6, f"{complexity_score} / 10", ha="center", va="center", fontsize=24)
                ax4.text(0.5, 0.4, f"Overall Complexity: {complexity_level}", ha="center", va="center")
                ax4.add_patch(plt.Rectangle((0.2, 0.2), 0.6, 0.1, color=level_colors.get(complexity_level, "gray"), alpha=0.7))
                ax4.axis("off")
            else:
                ax4.text(0.5, 0.5, "No overall complexity data", ha="center", va="center")
                ax4.axis("off")
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            return {
                "success": True,
                "visualization_path": output_path,
                "metrics": metrics.get("metrics", {})
            }
        
        except Exception as e:
            logger.error(f"Error visualizing code complexity: {str(e)}")
            return {"error": str(e)}

class RepositoryVisualizer:
    """Generate visualizations for GitHub repositories."""
    
    def __init__(self, github_api: GitHubAPI):
        """Initialize with GitHub API."""
        self.github_api = github_api
    
    def create_commit_history_visualization(self, owner: str, repo: str, days: int = 30, 
                                          output_path: str = "commit_history.png") -> Dict[str, Any]:
        """Create a visualization of commit history."""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for GitHub API
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            
            # Get commits
            params = {
                "since": f"{start_date_str}T00:00:00Z",
                "until": f"{end_date_str}T23:59:59Z",
                "per_page": 100
            }
            
            commits = self.github_api.get(f"/repos/{owner}/{repo}/commits", params=params)
            
            if not commits:
                return {"error": "No commits found in the specified date range"}
            
            # Extract dates and count commits per day
            commit_dates = []
            for commit in commits:
                if "commit" in commit and "author" in commit["commit"] and "date" in commit["commit"]["author"]:
                    date_str = commit["commit"]["author"]["date"]
                    commit_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    commit_dates.append(commit_date.date())
            
            # Count commits per day
            date_counts = {}
            for date in commit_dates:
                date_str = date.isoformat()
                if date_str not in date_counts:
                    date_counts[date_str] = 0
                date_counts[date_str] += 1
            
            # Fill in missing dates
            current_date = start_date.date()
            while current_date <= end_date.date():
                date_str = current_date.isoformat()
                if date_str not in date_counts:
                    date_counts[date_str] = 0
                current_date += timedelta(days=1)
            
            # Sort dates
            sorted_dates = sorted(date_counts.keys())
            counts = [date_counts[date] for date in sorted_dates]
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            plt.bar(sorted_dates, counts, color="skyblue")
            plt.title(f"Commit History for {owner}/{repo} (Last {days} Days)")
            plt.xlabel("Date")
            plt.ylabel("Commits")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            # Add trend line
            if len(sorted_dates) > 1:
                z = np.polyfit(range(len(sorted_dates)), counts, 1)
                p = np.poly1d(z)
                plt.plot(sorted_dates, p(range(len(sorted_dates))), "r--", alpha=0.7)
            
            plt.savefig(output_path)
            plt.close()
            
            return {
                "success": True,
                "visualization_path": output_path,
                "commit_count": len(commit_dates),
                "date_range": {
                    "start": start_date_str,
                    "end": end_date_str
                }
            }
        
        except Exception as e:
            logger.error(f"Error creating commit history visualization: {str(e)}")
            return {"error": str(e)}
    
    def create_contributor_visualization(self, owner: str, repo: str, 
                                       output_path: str = "contributors.png") -> Dict[str, Any]:
        """Create a visualization of repository contributors."""
        try:
            # Get contributors
            contributors = self.github_api.get(f"/repos/{owner}/{repo}/contributors")
            
            if not contributors:
                return {"error": "No contributors found"}
            
            # Extract data
            usernames = []
            contributions = []
            
            for contributor in contributors[:20]:  # Limit to top 20
                usernames.append(contributor.get("login", "Unknown"))
                contributions.append(contributor.get("contributions", 0))
            
            # Sort by contribution count (descending)
            sorted_data = sorted(zip(usernames, contributions), key=lambda x: x[1], reverse=True)
            usernames = [item[0] for item in sorted_data]
            contributions = [item[1] for item in sorted_data]
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            
            # Horizontal bar chart
            plt.barh(usernames, contributions, color="lightgreen")
            plt.title(f"Top Contributors for {owner}/{repo}")
            plt.xlabel("Contributions")
            plt.ylabel("Username")
            plt.tight_layout()
            
            plt.savefig(output_path)
            plt.close()
            
            return {
                "success": True,
                "visualization_path": output_path,
                "contributor_count": len(contributors),
                "top_contributors": [{"username": user, "contributions": count} 
                                    for user, count in zip(usernames, contributions)]
            }
        
        except Exception as e:
            logger.error(f"Error creating contributor visualization: {str(e)}")
            return {"error": str(e)}
    
    def create_language_visualization(self, owner: str, repo: str,
                                    output_path: str = "languages.png") -> Dict[str, Any]:
        """Create a visualization of repository languages."""
        try:
            # Get languages
            languages = self.github_api.get(f"/repos/{owner}/{repo}/languages")
            
            if not languages:
                return {"error": "No language data found"}
            
            # Extract data
            labels = list(languages.keys())
            sizes = list(languages.values())
            
            # Calculate percentages
            total = sum(sizes)
            percentages = [size / total * 100 for size in sizes]
            
            # Create visualization
            plt.figure(figsize=(10, 8))
            
            # Create color map
            colormap = plt.cm.get_cmap("tab20")
            colors = [colormap(i % 20) for i in range(len(labels))]
            
            # Create pie chart
            patches, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%1.1f%%',
                                              startangle=90, colors=colors)
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            plt.axis('equal')
            plt.title(f"Language Distribution for {owner}/{repo}")
            plt.tight_layout()
            
            plt.savefig(output_path)
            plt.close()
            
            return {
                "success": True,
                "visualization_path": output_path,
                "languages": {label: {"bytes": size, "percentage": pct} 
                             for label, size, pct in zip(labels, sizes, percentages)}
            }
        
        except Exception as e:
            logger.error(f"Error creating language visualization: {str(e)}")
            return {"error": str(e)}

# Define GitHub tools
def get_advanced_github_tools() -> List:
    """Get tools for advanced GitHub features."""
    from anthropic_agent import Tool, ToolParameter
    
    # Initialize GitHub API
    github_api = GitHubAPI()
    
    # Initialize managers and analyzers
    actions_manager = GitHubActionsManager(github_api)
    pr_manager = PRManager(github_api)
    code_quality_analyzer = CodeQualityAnalyzer(github_api)
    dependency_analyzer = DependencyAnalyzer(github_api)
    repo_visualizer = RepositoryVisualizer(github_api)
    
    # GitHub Actions tools
    async def list_github_workflows(owner: str, repo: str) -> Dict[str, Any]:
        """List GitHub Actions workflows in a repository."""
        return actions_manager.list_workflows(owner, repo)
    
    async def get_github_workflow(owner: str, repo: str, workflow_id: Union[int, str]) -> Dict[str, Any]:
        """Get a specific GitHub Actions workflow."""
        return actions_manager.get_workflow(owner, repo, workflow_id)
    
    async def list_workflow_runs(owner: str, repo: str, workflow_id: Optional[Union[int, str]] = None, 
                               status: Optional[str] = None) -> Dict[str, Any]:
        """List GitHub Actions workflow runs."""
        return actions_manager.list_workflow_runs(owner, repo, workflow_id, status)
    
    async def trigger_workflow(owner: str, repo: str, workflow_id: Union[int, str], 
                             ref: str, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Trigger a GitHub Actions workflow."""
        return actions_manager.trigger_workflow(owner, repo, workflow_id, ref, inputs)
    
    async def visualize_workflow_history(owner: str, repo: str, workflow_id: Union[int, str]) -> Dict[str, Any]:
        """Visualize GitHub Actions workflow run history."""
        output_path = f"workflow_history_{owner}_{repo}_{workflow_id}.png"
        return actions_manager.visualize_workflow_history(owner, repo, workflow_id, output_path)
    
    # PR Management tools
    async def list_pull_requests(owner: str, repo: str, state: str = "open") -> Dict[str, Any]:
        """List pull requests in a repository."""
        return pr_manager.list_pull_requests(owner, repo, state)
    
    async def get_pull_request(owner: str, repo: str, pull_number: int) -> Dict[str, Any]:
        """Get a specific pull request."""
        return pr_manager.get_pull_request(owner, repo, pull_number)
    
    async def create_pull_request(owner: str, repo: str, title: str, head: str, base: str, 
                                body: Optional[str] = None, draft: bool = False) -> Dict[str, Any]:
        """Create a pull request."""
        return pr_manager.create_pull_request(owner, repo, title, head, base, body, draft)
    
    async def analyze_pull_request(owner: str, repo: str, pull_number: int) -> Dict[str, Any]:
        """Analyze a pull request."""
        analysis = pr_manager.analyze_pull_request(owner, repo, pull_number)
        
        # Optionally create a visualization
        if "error" not in analysis:
            output_path = f"pr_analysis_{owner}_{repo}_{pull_number}.png"
            visualization = pr_manager.visualize_pull_request(analysis, output_path)
            if "success" in visualization:
                analysis["visualization"] = visualization
        
        return analysis
    
    # Code Quality tools
    async def analyze_code_complexity(owner: str, repo: str, path: str, ref: Optional[str] = None) -> Dict[str, Any]:
        """Analyze code complexity."""
        analysis = code_quality_analyzer.analyze_code_complexity(owner, repo, path, ref)
        
        # Optionally create a visualization
        if "error" not in analysis:
            output_path = f"code_complexity_{owner}_{repo}_{path.replace('/', '_')}.png"
            visualization = code_quality_analyzer.visualize_code_complexity(analysis, output_path)
            if "success" in visualization:
                analysis["visualization"] = visualization
        
        return analysis
    
    # Dependency tools
    async def scan_dependencies(owner: str, repo: str, ref: Optional[str] = None) -> Dict[str, Any]:
        """Scan dependencies in a repository."""
        dependencies = dependency_analyzer.scan_dependencies(owner, repo, ref)
        
        # Optionally create a visualization
        if "error" not in dependencies and "dependencies" in dependencies:
            output_path = f"dependency_graph_{owner}_{repo}.png"
            visualization = dependency_analyzer.generate_dependency_graph(dependencies["dependencies"], output_path)
            if "success" in visualization:
                dependencies["visualization"] = visualization
        
        return dependencies
    
    # Repository visualization tools
    async def visualize_commit_history(owner: str, repo: str, days: int = 30) -> Dict[str, Any]:
        """Visualize commit history."""
        output_path = f"commit_history_{owner}_{repo}.png"
        return repo_visualizer.create_commit_history_visualization(owner, repo, days, output_path)
    
    async def visualize_contributors(owner: str, repo: str) -> Dict[str, Any]:
        """Visualize repository contributors."""
        output_path = f"contributors_{owner}_{repo}.png"
        return repo_visualizer.create_contributor_visualization(owner, repo, output_path)
    
    async def visualize_languages(owner: str, repo: str) -> Dict[str, Any]:
        """Visualize repository languages."""
        output_path = f"languages_{owner}_{repo}.png"
        return repo_visualizer.create_language_visualization(owner, repo, output_path)
    
    # Define tools list
    return [
        # GitHub Actions tools
        Tool(
            name="list_github_workflows",
            description="List GitHub Actions workflows in a repository",
            parameters=[
                ToolParameter(name="owner", type="string", description="Repository owner"),
                ToolParameter(name="repo", type="string", description="Repository name")
            ],
            function=list_github_workflows,
            category="github_actions"
        ),
        
        Tool(
            name="get_github_workflow",
            description="Get a specific GitHub Actions workflow",
            parameters=[
                ToolParameter(name="owner", type="string", description="Repository owner"),
                ToolParameter(name="repo", type="string", description="Repository name"),
                ToolParameter(name="workflow_id", type="string", description="Workflow ID or file name")
            ],
            function=get_github_workflow,
            category="github_actions"
        ),
        
        Tool(
            name="list_workflow_runs",
            description="List GitHub Actions workflow runs",
            parameters=[
                ToolParameter(name="owner", type="string", description="Repository owner"),
                ToolParameter(name="repo", type="string", description="Repository name"),
                ToolParameter(name="workflow_id", type="string", description="Workflow ID or file name", required=False),
                ToolParameter(name="status", type="string", description="Filter by status (completed, success, failure)", required=False)
            ],
            function=list_workflow_runs,
            category="github_actions"
        ),
        
        Tool(
            name="trigger_workflow",
            description="Trigger a GitHub Actions workflow",
            parameters=[
                ToolParameter(name="owner", type="string", description="Repository owner"),
                ToolParameter(name="repo", type="string", description="Repository name"),
                ToolParameter(name="workflow_id", type="string", description="Workflow ID or file name"),
                ToolParameter(name="ref", type="string", description="Git reference (branch, tag, or SHA)"),
                ToolParameter(name="inputs", type="object", description="Workflow inputs", required=False)
            ],
            function=trigger_workflow,
            category="github_actions"
        ),
        
        Tool(
            name="visualize_workflow_history",
            description="Visualize GitHub Actions workflow run history",
            parameters=[
                ToolParameter(name="owner", type="string", description="Repository owner"),
                ToolParameter(name="repo", type="string", description="Repository name"),
                ToolParameter(name="workflow_id", type="string", description="Workflow ID or file name")
            ],
            function=visualize_workflow_history,
            category="github_actions"
        ),
        
        # PR Management tools
        Tool(
            name="list_pull_requests",
            description="List pull requests in a repository",
            parameters=[
                ToolParameter(name="owner", type="string", description="Repository owner"),
                ToolParameter(name="repo", type="string", description="Repository name"),
                ToolParameter(name="state", type="string", description="PR state (open, closed, all)", required=False, default="open")
            ],
            function=list_pull_requests,
            category="github_pr"
        ),
        
        Tool(
            name="get_pull_request",
            description="Get a specific pull request",
            parameters=[
                ToolParameter(name="owner", type="string", description="Repository owner"),
                ToolParameter(name="repo", type="string", description="Repository name"),
                ToolParameter(name="pull_number", type="integer", description="Pull request number")
            ],
            function=get_pull_request,
            category="github_pr"
        ),
        
        Tool(
            name="create_pull_request",
            description="Create a pull request",
            parameters=[
                ToolParameter(name="owner", type="string", description="Repository owner"),
                ToolParameter(name="repo", type="string", description="Repository name"),
                ToolParameter(name="title", type="string", description="Pull request title"),
                ToolParameter(name="head", type="string", description="Name of the branch where changes are implemented"),
                ToolParameter(name="base", type="string", description="Name of the branch to merge changes into"),
                ToolParameter(name="body", type="string", description="Pull request body", required=False),
                ToolParameter(name="draft", type="boolean", description="Create as draft PR", required=False, default=False)
            ],
            function=create_pull_request,
            category="github_pr"
        ),
        
        Tool(
            name="analyze_pull_request",
            description="Analyze a pull request with detailed metrics and visualization",
            parameters=[
                ToolParameter(name="owner", type="string", description="Repository owner"),
                ToolParameter(name="repo", type="string", description="Repository name"),
                ToolParameter(name="pull_number", type="integer", description="Pull request number")
            ],
            function=analyze_pull_request,
            category="github_pr"
        ),
        
        # Code Quality tools
        Tool(
            name="analyze_code_complexity",
            description="Analyze code complexity with detailed metrics and visualization",
            parameters=[
                ToolParameter(name="owner", type="string", description="Repository owner"),
                ToolParameter(name="repo", type="string", description="Repository name"),
                ToolParameter(name="path", type="string", description="File path"),
                ToolParameter(name="ref", type="string", description="Branch, tag, or commit SHA", required=False)
            ],
            function=analyze_code_complexity,
            category="github_code_quality"
        ),
        
        # Dependency tools
        Tool(
            name="scan_dependencies",
            description="Scan dependencies in a repository with visualization",
            parameters=[
                ToolParameter(name="owner", type="string", description="Repository owner"),
                ToolParameter(name="repo", type="string", description="Repository name"),
                ToolParameter(name="ref", type="string", description="Branch, tag, or commit SHA", required=False)
            ],
            function=scan_dependencies,
            category="github_dependencies"
        ),
        
        # Repository visualization tools
        Tool(
            name="visualize_commit_history",
            description="Visualize commit history",
            parameters=[
                ToolParameter(name="owner", type="string", description="Repository owner"),
                ToolParameter(name="repo", type="string", description="Repository name"),
                ToolParameter(name="days", type="integer", description="Number of days to include", required=False, default=30)
            ],
            function=visualize_commit_history,
            category="github_visualization"
        ),
        
        Tool(
            name="visualize_contributors",
            description="Visualize repository contributors",
            parameters=[
                ToolParameter(name="owner", type="string", description="Repository owner"),
                ToolParameter(name="repo", type="string", description="Repository name")
            ],
            function=visualize_contributors,
            category="github_visualization"
        ),
        
        Tool(
            name="visualize_languages",
            description="Visualize repository languages",
            parameters=[
                ToolParameter(name="owner", type="string", description="Repository owner"),
                ToolParameter(name="repo", type="string", description="Repository name")
            ],
            function=visualize_languages,
            category="github_visualization"
        )
    ]

# Combined get_github_tools function that includes basic and advanced tools
def get_github_tools() -> List:
    """Get all GitHub tools including basic and advanced features."""
    from anthropic_agent import Tool, ToolParameter
    
    # Get basic GitHub tools
    basic_tools = [
        # Re-implement basic tools here or import from the existing module
        Tool(
            name="get_repo_info",
            description="Get information about a GitHub repository",
            parameters=[
                ToolParameter(name="owner", type="string", description="Repository owner"),
                ToolParameter(name="repo", type="string", description="Repository name")
            ],
            function=lambda owner, repo: GitHubAPI().get(f"/repos/{owner}/{repo}"),
            category="github"
        ),
        
        Tool(
            name="get_file_contents",
            description="Get the contents of a file in a GitHub repository",
            parameters=[
                ToolParameter(name="owner", type="string", description="Repository owner"),
                ToolParameter(name="repo", type="string", description="Repository name"),
                ToolParameter(name="path", type="string", description="File path"),
                ToolParameter(name="ref", type="string", description="Branch, tag, or commit SHA", required=False)
            ],
            function=lambda owner, repo, path, ref=None: GitHubAPI().get(
                f"/repos/{owner}/{repo}/contents/{path}",
                params={"ref": ref} if ref else None
            ),
            category="github"
        ),
        
        Tool(
            name="search_code",
            description="Search code in GitHub repositories",
            parameters=[
                ToolParameter(name="query", type="string", description="Search query")
            ],
            function=lambda query: GitHubAPI().get("/search/code", {"q": query}),
            category="github"
        )
    ]
    
    # Get advanced GitHub tools
    advanced_tools = get_advanced_github_tools()
    
    # Combine all tools
    return basic_tools + advanced_tools