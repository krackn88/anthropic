"""
Model configuration and selection for the Anthropic-powered Agent
"""

from typing import List, Dict, Any, Optional

# Claude model information
CLAUDE_MODELS = [
    {
        "id": "claude-3-opus-20240229",
        "name": "Claude 3 Opus",
        "description": "Most powerful model for complex tasks requiring deep understanding and expert-level problem solving.",
        "context_window": 200000,
        "training_data": "Up to early 2023",
        "capabilities": {
            "reasoning": True,
            "code": True,
            "vision": True,
            "tool_use": True,
            "multilingual": True,
            "creative_writing": True
        },
        "use_cases": [
            "Complex reasoning tasks",
            "Scientific and technical writing",
            "Advanced code generation and analysis",
            "Multimodal understanding (text + images)",
            "Tasks requiring exceptional precision"
        ],
        "pricing": {
            "input_per_million": 15000,  # $15 per million tokens
            "output_per_million": 75000  # $75 per million tokens
        }
    },
    {
        "id": "claude-3-sonnet-20240229",
        "name": "Claude 3 Sonnet",
        "description": "Balanced model with excellent performance across a wide range of tasks at a lower cost.",
        "context_window": 200000,
        "training_data": "Up to early 2023",
        "capabilities": {
            "reasoning": True,
            "code": True,
            "vision": True,
            "tool_use": True,
            "multilingual": True,
            "creative_writing": True
        },
        "use_cases": [
            "General purpose tasks",
            "Everyday code assistance",
            "Content analysis and generation",
            "Multimodal understanding (text + images)",
            "Business applications and customer support"
        ],
        "pricing": {
            "input_per_million": 3000,  # $3 per million tokens
            "output_per_million": 15000  # $15 per million tokens
        }
    },
    {
        "id": "claude-3-haiku-20240307",
        "name": "Claude 3 Haiku",
        "description": "Fastest, most compact model for high-volume, lower complexity tasks requiring quick responses.",
        "context_window": 200000,
        "training_data": "Up to early 2023",
        "capabilities": {
            "reasoning": True,
            "code": True,
            "vision": True,
            "tool_use": True,
            "multilingual": True,
            "creative_writing": True
        },
        "use_cases": [
            "Real-time chat applications",
            "Quick responses to straightforward questions",
            "Simple content moderation",
            "Basic image understanding",
            "High-volume processing"
        ],
        "pricing": {
            "input_per_million": 250,  # $0.25 per million tokens
            "output_per_million": 1250  # $1.25 per million tokens
        }
    },
    {
        "id": "claude-2.1",
        "name": "Claude 2.1",
        "description": "Advanced model with improved reasoning, comprehension, and task performance.",
        "context_window": 100000,
        "training_data": "Up to late 2022",
        "capabilities": {
            "reasoning": True,
            "code": True,
            "vision": False,
            "tool_use": True,
            "multilingual": True,
            "creative_writing": True
        },
        "use_cases": [
            "Long document analysis",
            "Complex reasoning tasks",
            "Code assistance",
            "Content generation",
            "Customer support automation"
        ],
        "pricing": {
            "input_per_million": 8000,  # $8 per million tokens
            "output_per_million": 24000  # $24 per million tokens
        }
    },
    {
        "id": "claude-2.0",
        "name": "Claude 2.0",
        "description": "Solid, reliable model with good performance across various tasks.",
        "context_window": 100000,
        "training_data": "Up to late 2022",
        "capabilities": {
            "reasoning": True,
            "code": True,
            "vision": False,
            "tool_use": True,
            "multilingual": True,
            "creative_writing": True
        },
        "use_cases": [
            "Content summarization",
            "Document analysis",
            "Question answering",
            "Code assistance",
            "General purpose tasks"
        ],
        "pricing": {
            "input_per_million": 8000,  # $8 per million tokens
            "output_per_million": 24000  # $24 per million tokens
        }
    },
    {
        "id": "claude-instant-1.2",
        "name": "Claude Instant 1.2",
        "description": "Fast, economical model for simpler tasks requiring quick responses.",
        "context_window": 100000,
        "training_data": "Up to early 2023",
        "capabilities": {
            "reasoning": True,
            "code": True,
            "vision": False,
            "tool_use": True,
            "multilingual": True,
            "creative_writing": True
        },
        "use_cases": [
            "High-volume processing",
            "Simple content creation",
            "Quick responses to straightforward questions",
            "Basic customer support automation",
            "Text classification"
        ],
        "pricing": {
            "input_per_million": 1630,  # $1.63 per million tokens
            "output_per_million": 5510  # $5.51 per million tokens
        }
    }
]

def list_available_models() -> List[Dict[str, Any]]:
    """List all available Claude models."""
    return CLAUDE_MODELS

def get_model_info(model_id: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific model."""
    for model in CLAUDE_MODELS:
        if model["id"] == model_id:
            return model
    return None

def recommend_model(task_description: str) -> List[Dict[str, Any]]:
    """Recommend models based on a task description."""
    # Task categories and keywords
    task_keywords = {
        "complex_reasoning": [
            "complex", "reasoning", "difficult", "expert", "advanced", "nuanced", 
            "deep analysis", "thorough", "comprehensive", "sophisticated"
        ],
        "code": [
            "code", "programming", "algorithm", "function", "debug", "software", 
            "development", "engineering", "repository", "git", "repository"
        ],
        "creative": [
            "creative", "write", "story", "poem", "novel", "fiction", "narrative", 
            "content creation", "copywriting", "marketing"
        ],
        "fast": [
            "quick", "fast", "rapid", "speed", "immediately", "instant", "real-time",
            "responsive", "high volume", "efficient"
        ],
        "vision": [
            "image", "picture", "photo", "visual", "diagram", "chart", "graph",
            "screenshot", "scan", "camera", "multimodal"
        ],
        "simple": [
            "simple", "basic", "straightforward", "easy", "elementary", "uncomplicated",
            "standard", "routine"
        ]
    }
    
    # Score task based on keywords
    task_scores = {category: 0 for category in task_keywords}
    task_lower = task_description.lower()
    
    for category, keywords in task_keywords.items():
        for keyword in keywords:
            if keyword.lower() in task_lower:
                task_scores[category] += 1
    
    # Model recommendations
    recommendations = []
    
    # Logic for recommendations
    if task_scores["vision"] > 0:
        # For vision tasks, only Claude 3 models are applicable
        if task_scores["complex_reasoning"] > 1 or task_scores["code"] > 1:
            recommendations.append({
                "model": get_model_info("claude-3-opus-20240229"),
                "suitability": 9,
                "reason": "Best choice for complex vision tasks requiring detailed analysis"
            })
        
        recommendations.append({
            "model": get_model_info("claude-3-sonnet-20240229"),
            "suitability": 8,
            "reason": "Excellent balance of performance and cost for vision tasks"
        })
        
        if task_scores["fast"] > 1 or task_scores["simple"] > 1:
            recommendations.append({
                "model": get_model_info("claude-3-haiku-20240307"),
                "suitability": 7,
                "reason": "Fastest model for simple vision tasks requiring quick responses"
            })
    else:
        # Non-vision tasks
        if task_scores["complex_reasoning"] > 1 or task_scores["code"] > 1:
            recommendations.append({
                "model": get_model_info("claude-3-opus-20240229"),
                "suitability": 9,
                "reason": "Most powerful model for complex reasoning and code tasks"
            })
            
            recommendations.append({
                "model": get_model_info("claude-3-sonnet-20240229"),
                "suitability": 8,
                "reason": "Excellent balance of performance and cost for advanced tasks"
            })
            
            recommendations.append({
                "model": get_model_info("claude-2.1"),
                "suitability": 7,
                "reason": "Strong performance on complex tasks with a large context window"
            })
        
        elif task_scores["creative"] > 1:
            recommendations.append({
                "model": get_model_info("claude-3-opus-20240229"),
                "suitability": 8,
                "reason": "Superior creative capabilities for high-quality content"
            })
            
            recommendations.append({
                "model": get_model_info("claude-3-sonnet-20240229"),
                "suitability": 8,
                "reason": "Excellent creative abilities with good cost efficiency"
            })
        
        elif task_scores["fast"] > 1 or task_scores["simple"] > 1:
            recommendations.append({
                "model": get_model_info("claude-3-haiku-20240307"),
                "suitability": 9,
                "reason": "Fastest model for simple tasks requiring quick responses"
            })
            
            recommendations.append({
                "model": get_model_info("claude-instant-1.2"),
                "suitability": 7,
                "reason": "Economical option for high-volume simple tasks"
            })
        
        else:
            # Default recommendation for general tasks
            recommendations.append({
                "model": get_model_info("claude-3-sonnet-20240229"),
                "suitability": 9,
                "reason": "Best all-around model for general tasks with excellent performance"
            })
            
            recommendations.append({
                "model": get_model_info("claude-3-opus-20240229"),
                "suitability": 8,
                "reason": "Most powerful model for any task, but at a higher cost"
            })
            
            recommendations.append({
                "model": get_model_info("claude-3-haiku-20240307"),
                "suitability": 7,
                "reason": "Quick, cost-effective model for general tasks"
            })
    
    # Sort by suitability
    recommendations.sort(key=lambda x: x["suitability"], reverse=True)
    
    return recommendations