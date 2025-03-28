"""
Claude language capabilities for the Anthropic-powered Agent
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Anthropic client
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def get_claude_tools() -> List:
    """Get tools for Claude language capabilities."""
    from anthropic_agent import Tool, ToolParameter
    
    async def summarize_text(text: str, max_length: Optional[int] = None, format: Optional[str] = None) -> Dict[str, Any]:
        """Summarize a given text using Claude."""
        try:
            prompt = f"Summarize the following text:"
            
            if max_length:
                prompt += f" Limit your summary to approximately {max_length} words."
            
            if format:
                if format.lower() == "bullet":
                    prompt += " Format your summary as bullet points."
                elif format.lower() == "paragraph":
                    prompt += " Format your summary as a single paragraph."
                elif format.lower() == "detailed":
                    prompt += " Provide a detailed summary with sections for key points."
            
            prompt += f"\n\n{text}"
            
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
            
            return {"summary": response.content[0].text}
        
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return {"error": str(e)}
    
    async def translate_text(text: str, target_language: str, preserve_formatting: bool = True) -> Dict[str, Any]:
        """Translate a given text to the target language using Claude."""
        try:
            prompt = f"Translate the following text to {target_language}:"
            
            if preserve_formatting:
                prompt += " Preserve the original formatting, such as paragraphs, bullet points, etc."
            
            prompt += f"\n\n{text}"
            
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
            
            return {
                "translation": response.content[0].text,
                "source_text": text,
                "target_language": target_language
            }
        
        except Exception as e:
            logger.error(f"Error translating text: {str(e)}")
            return {"error": str(e)}
    
    async def complete_code(prompt: str, language: Optional[str] = None) -> Dict[str, Any]:
        """Generate code completion for a given prompt using Claude."""
        try:
            full_prompt = "Complete the following code:"
            
            if language:
                full_prompt += f" The code is written in {language}."
            
            full_prompt += f"\n\n{prompt}"
            
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ]
            )
            
            return {"completion": response.content[0].text}
        
        except Exception as e:
            logger.error(f"Error completing code: {str(e)}")
            return {"error": str(e)}
    
    async def explain_code(code: str, detail_level: str = "standard") -> Dict[str, Any]:
        """Explain a given piece of code using Claude."""
        try:
            prompt = "Explain the following code:"
            
            if detail_level == "simple":
                prompt += " Provide a simple, high-level explanation for beginners."
            elif detail_level == "detailed":
                prompt += " Provide a detailed explanation of all components, logic, and potential issues."
            elif detail_level == "step-by-step":
                prompt += " Walk through the code step-by-step, explaining what happens at each line or block."
            
            prompt += f"\n\n{code}"
            
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
            
            return {"explanation": response.content[0].text}
        
        except Exception as e:
            logger.error(f"Error explaining code: {str(e)}")
            return {"error": str(e)}
    
    async def analyze_text(text: str, analysis_type: str = "sentiment") -> Dict[str, Any]:
        """Analyze a given text using Claude."""
        try:
            if analysis_type == "sentiment":
                prompt = "Analyze the sentiment of the following text. Determine if it's positive, negative, or neutral, and explain your reasoning:\n\n"
            elif analysis_type == "entities":
                prompt = "Extract the named entities (people, organizations, locations, etc.) from the following text:\n\n"
            elif analysis_type == "topics":
                prompt = "Identify the main topics discussed in the following text:\n\n"
            elif analysis_type == "summary":
                prompt = "Provide a concise summary of the following text:\n\n"
            else:
                prompt = f"Analyze the following text for {analysis_type}:\n\n"
            
            prompt += text
            
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
            
            return {
                "analysis": response.content[0].text,
                "analysis_type": analysis_type
            }
        
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {"error": str(e)}
    
    async def generate_creative_text(prompt: str, style: Optional[str] = None, length: Optional[str] = "medium") -> Dict[str, Any]:
        """Generate creative text using Claude."""
        try:
            full_prompt = prompt
            
            if style:
                full_prompt = f"Write in the style of {style}: {prompt}"
            
            length_guide = {
                "short": "Write a brief response (100-200 words).",
                "medium": "Write a moderate-length response (300-500 words).",
                "long": "Write a detailed response (800-1000 words)."
            }
            
            if length in length_guide:
                full_prompt = f"{full_prompt}\n\n{length_guide[length]}"
            
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ]
            )
            
            return {"text": response.content[0].text}
        
        except Exception as e:
            logger.error(f"Error generating creative text: {str(e)}")
            return {"error": str(e)}
    
    # Define tools
    return [
        Tool(
            name="summarize_text",
            description="Summarize a given text using Claude",
            parameters=[
                ToolParameter(name="text", type="string", description="Text to summarize"),
                ToolParameter(name="max_length", type="integer", description="Maximum length of summary (in words)", required=False),
                ToolParameter(name="format", type="string", description="Format of summary (bullet, paragraph, detailed)", required=False)
            ],
            function=summarize_text,
            category="claude"
        ),
        
        Tool(
            name="translate_text",
            description="Translate a given text to the target language using Claude",
            parameters=[
                ToolParameter(name="text", type="string", description="Text to translate"),
                ToolParameter(name="target_language", type="string", description="Target language"),
                ToolParameter(name="preserve_formatting", type="boolean", description="Whether to preserve original formatting", required=False, default=True)
            ],
            function=translate_text,
            category="claude"
        ),
        
        Tool(
            name="complete_code",
            description="Generate code completion for a given prompt using Claude",
            parameters=[
                ToolParameter(name="prompt", type="string", description="Code prompt"),
                ToolParameter(name="language", type="string", description="Programming language", required=False)
            ],
            function=complete_code,
            category="claude"
        ),
        
        Tool(
            name="explain_code",
            description="Explain a given piece of code using Claude",
            parameters=[
                ToolParameter(name="code", type="string", description="Code to explain"),
                ToolParameter(name="detail_level", type="string", description="Level of detail (simple, standard, detailed, step-by-step)", required=False, default="standard")
            ],
            function=explain_code,
            category="claude"
        ),
        
        Tool(
            name="analyze_text",
            description="Analyze a given text using Claude",
            parameters=[
                ToolParameter(name="text", type="string", description="Text to analyze"),
                ToolParameter(name="analysis_type", type="string", description="Type of analysis (sentiment, entities, topics, summary)", required=False, default="sentiment")
            ],
            function=analyze_text,
            category="claude"
        ),
        
        Tool(
            name="generate_creative_text",
            description="Generate creative text using Claude",
            parameters=[
                ToolParameter(name="prompt", type="string", description="Creative prompt"),
                ToolParameter(name="style", type="string", description="Writing style", required=False),
                ToolParameter(name="length", type="string", description="Length of response (short, medium, long)", required=False, default="medium")
            ],
            function=generate_creative_text,
            category="claude"
        )
    ]