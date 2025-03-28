"""
Advanced Claude techniques from Anthropic's cookbook
"""

import os
import logging
from typing import List, Dict, Any
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Anthropic client
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def get_cookbook_tools() -> List:
    """Get tools for advanced techniques from Anthropic's cookbook."""
    from anthropic_agent import Tool, ToolParameter
    
    async def few_shot_prompting(prompt: str, examples: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate a response using few-shot prompting."""
        try:
            # Construct few-shot prompt
            few_shot_prompt = prompt + "\n\nHere are some examples:\n\n"
            
            for i, example in enumerate(examples):
                if "input" in example and "output" in example:
                    few_shot_prompt += f"Example {i+1}:\nInput: {example['input']}\nOutput: {example['output']}\n\n"
            
            few_shot_prompt += "Now, please respond with the same format."
            
            # Call Claude
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": few_shot_prompt
                    }
                ]
            )
            
            return {"response": response.content[0].text}
        
        except Exception as e:
            logger.error(f"Error with few-shot prompting: {str(e)}")
            return {"error": str(e)}
    
    async def chain_of_thought(prompt: str) -> Dict[str, Any]:
        """Generate a response using chain-of-thought reasoning."""
        try:
            # Construct chain-of-thought prompt
            cot_prompt = f"I need you to solve this step-by-step, showing your reasoning:\n\n{prompt}\n\nLet's think about this systematically:"
            
            # Call Claude
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": cot_prompt
                    }
                ]
            )
            
            return {"response": response.content[0].text}
        
        except Exception as e:
            logger.error(f"Error with chain-of-thought: {str(e)}")
            return {"error": str(e)}
    
    async def self_critique(prompt: str) -> Dict[str, Any]:
        """Generate a response with self-critique."""
        try:
            # Construct self-critique prompt
            sc_prompt = f"""Please respond to the following prompt:

{prompt}

After providing your response, review your own work and provide a critique of any errors, biases, or areas for improvement. Then provide an improved final answer that addresses these issues.

Format your response as:

Initial Response:
[Your initial response]

Self-Critique:
[Your critique of your initial response]

Improved Response:
[Your improved response]"""
            
            # Call Claude
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": sc_prompt
                    }
                ]
            )
            
            return {"response": response.content[0].text}
        
        except Exception as e:
            logger.error(f"Error with self-critique: {str(e)}")
            return {"error": str(e)}
    
    async def role_prompting(prompt: str, role: str) -> Dict[str, Any]:
        """Generate a response with role prompting."""
        try:
            # Construct role prompt
            role_prompt = f"I'd like you to take on the role of {role} and respond to the following:\n\n{prompt}"
            
            # Call Claude
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": role_prompt
                    }
                ]
            )
            
            return {"response": response.content[0].text}
        
        except Exception as e:
            logger.error(f"Error with role prompting: {str(e)}")
            return {"error": str(e)}
    
    async def generate_analogies(concept: str, domains: List[str] = None) -> Dict[str, Any]:
        """Generate analogies for a concept across different domains."""
        try:
            # Construct prompt
            if domains:
                domains_text = ", ".join(domains)
                prompt = f"Generate analogies for the concept of '{concept}' in the following domains: {domains_text}."
            else:
                prompt = f"Generate analogies for the concept of '{concept}' across different domains (such as technology, nature, education, business, etc.)."
            
            prompt += " For each analogy, explain the similarities and how the analogy helps understand the original concept."
            
            # Call Claude
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return {"response": response.content[0].text}
        
        except Exception as e:
            logger.error(f"Error generating analogies: {str(e)}")
            return {"error": str(e)}
    
    async def structured_format_generation(prompt: str, format: str = "json", schema: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a response in a structured format (JSON, markdown table, etc.)."""
        try:
            # Construct format prompt
            if format.lower() == "json" and schema:
                format_prompt = f"""Please respond to the following prompt:

{prompt}

Provide your response in JSON format according to this schema:
{str(schema)}"""
            
            elif format.lower() == "json":
                format_prompt = f"""Please respond to the following prompt:

{prompt}

Provide your response in JSON format."""
            
            elif format.lower() == "table":
                format_prompt = f"""Please respond to the following prompt:

{prompt}

Provide your response as a markdown table with appropriate columns."""
            
            elif format.lower() == "bullet":
                format_prompt = f"""Please respond to the following prompt:

{prompt}

Provide your response as a bullet-point list."""
            
            else:
                format_prompt = f"""Please respond to the following prompt:

{prompt}

Provide your response in {format} format."""
            
            # Call Claude
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": format_prompt
                    }
                ]
            )
            
            return {"response": response.content[0].text}
        
        except Exception as e:
            logger.error(f"Error with structured format generation: {str(e)}")
            return {"error": str(e)}
    
    # Define tools
    return [
        Tool(
            name="few_shot_prompting",
            description="Generate a response using few-shot prompting with examples",
            parameters=[
                ToolParameter(name="prompt", type="string", description="Main prompt for Claude"),
                ToolParameter(name="examples", type="array", description="List of example dictionaries with 'input' and 'output' keys")
            ],
            function=few_shot_prompting,
            category="cookbook"
        ),
        
        Tool(
            name="chain_of_thought",
            description="Generate a response using chain-of-thought reasoning",
            parameters=[
                ToolParameter(name="prompt", type="string", description="Prompt for Claude to reason through step-by-step")
            ],
            function=chain_of_thought,
            category="cookbook"
        ),
        
        Tool(
            name="self_critique",
            description="Generate a response with self-critique and improvement",
            parameters=[
                ToolParameter(name="prompt", type="string", description="Prompt for Claude")
            ],
            function=self_critique,
            category="cookbook"
        ),
        
        Tool(
            name="role_prompting",
            description="Generate a response with role prompting",
            parameters=[
                ToolParameter(name="prompt", type="string", description="Prompt for Claude"),
                ToolParameter(name="role", type="string", description="Role for Claude to adopt (e.g., 'expert physicist', 'experienced programmer')")
            ],
            function=role_prompting,
            category="cookbook"
        ),
        
        Tool(
            name="generate_analogies",
            description="Generate analogies for a concept across different domains",
            parameters=[
                ToolParameter(name="concept", type="string", description="Concept to generate analogies for"),
                ToolParameter(name="domains", type="array", description="Specific domains for analogies", required=False)
            ],
            function=generate_analogies,
            category="cookbook"
        ),
        
        Tool(
            name="structured_format_generation",
            description="Generate a response in a structured format (JSON, markdown table, etc.)",
            parameters=[
                ToolParameter(name="prompt", type="string", description="Prompt for Claude"),
                ToolParameter(name="format", type="string", description="Desired format (json, table, bullet)", required=False, default="json"),
                ToolParameter(name="schema", type="object", description="JSON schema (when format is json)", required=False)
            ],
            function=structured_format_generation,
            category="cookbook"
        )
    ]