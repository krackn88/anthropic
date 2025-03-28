"""
Predefined schemas for structured outputs
"""

from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field

# Text Analysis Response
class SentimentAnalysis(BaseModel):
    """Sentiment analysis result."""
    sentiment: str = Field(..., description="Overall sentiment (positive, negative, neutral, mixed)")
    confidence: float = Field(..., description="Confidence score (0-1)", ge=0, le=1)
    explanation: str = Field(..., description="Explanation of the sentiment analysis")

class NamedEntity(BaseModel):
    """Named entity found in text."""
    text: str = Field(..., description="Entity text")
    type: str = Field(..., description="Entity type (person, organization, location, etc.)")
    start_index: int = Field(..., description="Starting character index in the original text")
    end_index: int = Field(..., description="Ending character index in the original text")

class TopicAnalysis(BaseModel):
    """Topic analysis result."""
    topic: str = Field(..., description="Topic name")
    relevance: float = Field(..., description="Topic relevance score (0-1)", ge=0, le=1)
    keywords: List[str] = Field(..., description="Keywords associated with the topic")

class KeyPoint(BaseModel):
    """Key point from text."""
    point: str = Field(..., description="The key point")
    importance: float = Field(..., description="Importance score (0-1)", ge=0, le=1)

class TextAnalysisResponse(BaseModel):
    """Structured response for text analysis."""
    text_summary: str = Field(..., description="Brief summary of the text")
    sentiment: SentimentAnalysis = Field(..., description="Sentiment analysis")
    entities: List[NamedEntity] = Field(default_factory=list, description="Named entities found in the text")
    topics: List[TopicAnalysis] = Field(default_factory=list, description="Topics identified in the text")
    key_points: List[KeyPoint] = Field(default_factory=list, description="Key points extracted from the text")
    word_count: int = Field(..., description="Total word count")
    readability_score: float = Field(..., description="Readability score (0-100)", ge=0, le=100)
    language_detected: str = Field(..., description="Detected language")

# Search Results Response
class SearchResult(BaseModel):
    """Individual search result."""
    title: str = Field(..., description="Result title")
    url: Optional[str] = Field(None, description="Result URL if available")
    snippet: str = Field(..., description="Text snippet from the result")
    relevance_score: float = Field(..., description="Relevance score (0-1)", ge=0, le=1)
    source: str = Field(..., description="Source of the result")

class SearchResultsResponse(BaseModel):
    """Structured response for search results."""
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., description="Total number of results found")
    results: List[SearchResult] = Field(..., description="Search results")
    suggested_queries: List[str] = Field(default_factory=list, description="Suggested related queries")
    response_time_ms: int = Field(..., description="Response time in milliseconds")

# GitHub Repository Analysis
class ContributorInfo(BaseModel):
    """Information about a repository contributor."""
    username: str = Field(..., description="GitHub username")
    contributions: int = Field(..., description="Number of contributions")
    role: Optional[str] = Field(None, description="Role in the project if determinable")

class LanguageBreakdown(BaseModel):
    """Programming language breakdown."""
    language: str = Field(..., description="Programming language name")
    percentage: float = Field(..., description="Percentage of code (0-100)", ge=0, le=100)
    bytes: Optional[int] = Field(None, description="Number of bytes")

class ActivityMetrics(BaseModel):
    """Repository activity metrics."""
    commits_last_month: int = Field(..., description="Number of commits in the last month")
    open_issues: int = Field(..., description="Number of open issues")
    closed_issues_last_month: int = Field(..., description="Number of issues closed in the last month")
    pull_requests_open: int = Field(..., description="Number of open pull requests")
    pull_requests_merged_last_month: int = Field(..., description="Number of PRs merged in the last month")

class GitHubRepositoryAnalysis(BaseModel):
    """Structured response for GitHub repository analysis."""
    repo_name: str = Field(..., description="Repository name")
    owner: str = Field(..., description="Repository owner")
    description: str = Field(..., description="Repository description")
    stars: int = Field(..., description="Number of stars")
    forks: int = Field(..., description="Number of forks")
    created_at: str = Field(..., description="Creation date")
    last_updated: str = Field(..., description="Last update date")
    homepage: Optional[str] = Field(None, description="Homepage URL")
    top_contributors: List[ContributorInfo] = Field(..., description="Top contributors")
    language_breakdown: List[LanguageBreakdown] = Field(..., description="Programming language breakdown")
    activity: ActivityMetrics = Field(..., description="Repository activity metrics")
    key_topics: List[str] = Field(..., description="Key topics/tags")
    summary: str = Field(..., description="Overall summary of the repository")

# Code Analysis Response
class CodeFunction(BaseModel):
    """Information about a function in the code."""
    name: str = Field(..., description="Function name")
    signature: str = Field(..., description="Function signature")
    description: str = Field(..., description="Description of what the function does")
    parameters: List[Dict[str, str]] = Field(..., description="Parameters with descriptions")
    returns: str = Field(..., description="Return value description")
    complexity: Optional[str] = Field(None, description="Complexity assessment")
    line_range: Dict[str, int] = Field(..., description="Start and end line numbers")

class CodeIssue(BaseModel):
    """Potential code issue."""
    type: str = Field(..., description="Issue type (security, performance, style, etc.)")
    severity: str = Field(..., description="Severity (high, medium, low)")
    description: str = Field(..., description="Description of the issue")
    location: Optional[Dict[str, int]] = Field(None, description="Location in the code")
    suggestion: str = Field(..., description="Suggested fix")

class CodeAnalysisResponse(BaseModel):
    """Structured response for code analysis."""
    language: str = Field(..., description="Programming language")
    code_summary: str = Field(..., description="Summary of what the code does")
    functions: List[CodeFunction] = Field(..., description="Functions identified in the code")
    classes: Optional[List[Dict[str, Any]]] = Field(None, description="Classes identified in the code")
    dependencies: List[str] = Field(default_factory=list, description="External dependencies identified")
    potential_issues: List[CodeIssue] = Field(default_factory=list, description="Potential issues identified")
    complexity_assessment: str = Field(..., description="Overall complexity assessment")
    best_practices: Dict[str, List[str]] = Field(..., description="Best practices followed and violated")
    improvement_suggestions: List[str] = Field(..., description="Suggestions for improvement")

# Plan Response
class PlanStep(BaseModel):
    """Step in a plan."""
    step_number: int = Field(..., description="Step number")
    title: str = Field(..., description="Step title")
    description: str = Field(..., description="Detailed description of the step")
    estimated_time: Optional[str] = Field(None, description="Estimated time to complete")
    dependencies: List[int] = Field(default_factory=list, description="Step numbers this step depends on")

class ResourceItem(BaseModel):
    """Resource for a plan."""
    name: str = Field(..., description="Resource name")
    type: str = Field(..., description="Resource type (tool, library, service, etc.)")
    url: Optional[str] = Field(None, description="Resource URL if applicable")
    description: str = Field(..., description="Description of the resource")

class PlanResponse(BaseModel):
    """Structured response for a plan."""
    plan_title: str = Field(..., description="Plan title")
    objective: str = Field(..., description="Plan objective")
    steps: List[PlanStep] = Field(..., description="Plan steps")
    estimated_total_time: Optional[str] = Field(None, description="Estimated total time")
    prerequisites: List[str] = Field(default_factory=list, description="Prerequisites for the plan")
    resources: List[ResourceItem] = Field(..., description="Resources needed for the plan")
    risks: List[Dict[str, str]] = Field(..., description="Potential risks and mitigations")
    success_criteria: List[str] = Field(..., description="Success criteria for the plan")

# All available schemas
AVAILABLE_SCHEMAS = {
    "text_analysis": {
        "id": "text_analysis",
        "name": "Text Analysis",
        "description": "Comprehensive analysis of text content",
        "schema": TextAnalysisResponse
    },
    "search_results": {
        "id": "search_results",
        "name": "Search Results",
        "description": "Structured search results with metadata",
        "schema": SearchResultsResponse
    },
    "github_repo": {
        "id": "github_repo",
        "name": "GitHub Repository Analysis",
        "description": "Detailed analysis of a GitHub repository",
        "schema": GitHubRepositoryAnalysis
    },
    "code_analysis": {
        "id": "code_analysis",
        "name": "Code Analysis",
        "description": "In-depth analysis of code with function detection and issue identification",
        "schema": CodeAnalysisResponse
    },
    "plan": {
        "id": "plan",
        "name": "Plan",
        "description": "Structured plan with steps, resources, and success criteria",
        "schema": PlanResponse
    }
}

def list_available_schemas() -> List[Dict[str, Any]]:
    """List all available schemas."""
    return [
        {
            "id": schema_id,
            "name": schema_info["name"],
            "description": schema_info["description"]
        }
        for schema_id, schema_info in AVAILABLE_SCHEMAS.items()
    ]

def get_schema_by_id(schema_id: str) -> Optional[Dict[str, Any]]:
    """Get schema information by ID."""
    return AVAILABLE_SCHEMAS.get(schema_id)