"""
Image processing module for the Anthropic-powered Agent
Enables multimodal capabilities with Claude 3 models
"""

import os
import json
import logging
import base64
import mimetypes
import requests
from typing import Dict, List, Optional, Any, Union, BinaryIO
from pathlib import Path
from PIL import Image
import io
from anthropic import Anthropic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Anthropic client
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

class ImageProcessor:
    """Process images for use with Claude 3 models."""
    
    @staticmethod
    def encode_image_base64(image_path: str) -> Dict[str, Any]:
        """
        Encode an image as base64 for Claude API
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict containing image type and base64-encoded data
        """
        try:
            with open(image_path, "rb") as image_file:
                # Get MIME type
                mime_type, _ = mimetypes.guess_type(image_path)
                if mime_type is None:
                    # Default to JPEG if type can't be determined
                    mime_type = "image/jpeg"
                
                # Encode as base64
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": encoded_string
                    }
                }
        except Exception as e:
            logger.error(f"Error encoding image: {str(e)}")
            raise
    
    @staticmethod
    def encode_image_url(image_url: str) -> Dict[str, Any]:
        """
        Create image object from URL for Claude API
        
        Args:
            image_url: URL of the image
            
        Returns:
            Dict containing image type and URL source
        """
        return {
            "type": "image",
            "source": {
                "type": "url",
                "url": image_url
            }
        }
    
    @staticmethod
    def optimize_image(image_path: str, max_size_mb: float = 5.0, 
                       max_width: int = 2000, max_height: int = 2000) -> str:
        """
        Optimize an image for Claude API by resizing and compressing
        
        Args:
            image_path: Path to the image file
            max_size_mb: Maximum file size in MB
            max_width: Maximum width in pixels
            max_height: Maximum height in pixels
            
        Returns:
            Path to the optimized image
        """
        try:
            # Open the image
            with Image.open(image_path) as img:
                # Get original format
                original_format = img.format
                if not original_format:
                    original_format = "JPEG"  # Default format
                
                # Check if resizing is needed
                width, height = img.size
                if width > max_width or height > max_height:
                    # Calculate aspect ratio
                    aspect_ratio = width / height
                    
                    if width > max_width:
                        width = max_width
                        height = int(width / aspect_ratio)
                    
                    if height > max_height:
                        height = max_height
                        width = int(height * aspect_ratio)
                    
                    # Resize image
                    img = img.resize((width, height), Image.LANCZOS)
                
                # Create output path
                output_path = f"{os.path.splitext(image_path)[0]}_optimized{os.path.splitext(image_path)[1]}"
                
                # Save with quality adjustment if JPEG
                if original_format == "JPEG":
                    # Start with high quality and reduce until size is acceptable
                    quality = 95
                    while quality > 70:  # Don't go below quality 70
                        img.save(output_path, format=original_format, quality=quality)
                        
                        # Check file size
                        size_mb = os.path.getsize(output_path) / (1024 * 1024)
                        if size_mb <= max_size_mb:
                            break
                        
                        # Reduce quality and try again
                        quality -= 5
                else:
                    # For other formats, just save normally
                    img.save(output_path, format=original_format)
                
                return output_path
        
        except Exception as e:
            logger.error(f"Error optimizing image: {str(e)}")
            return image_path  # Return original path if optimization fails
    
    @staticmethod
    def download_image(image_url: str, output_dir: str = "downloaded_images") -> str:
        """
        Download an image from a URL
        
        Args:
            image_url: URL of the image to download
            output_dir: Directory to save the downloaded image
            
        Returns:
            Path to the downloaded image
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Get filename from URL
            filename = os.path.basename(image_url.split("?")[0])  # Remove query params
            if not filename or "." not in filename:
                # Generate a filename if none is available
                filename = f"image_{hash(image_url) % 10000}.jpg"
            
            output_path = os.path.join(output_dir, filename)
            
            # Download the image
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return output_path
        
        except Exception as e:
            logger.error(f"Error downloading image: {str(e)}")
            raise

class ImageAnalyzer:
    """Analyze images using Claude 3 models."""
    
    def __init__(self, model: str = "claude-3-opus-20240229"):
        """Initialize the image analyzer."""
        self.client = client
        self.model = model
    
    async def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Analyze an image using Claude
        
        Args:
            image_path: Path to the image file
            prompt: Prompt for Claude describing what to analyze
            
        Returns:
            Claude's analysis of the image
        """
        try:
            # Optimize image if needed
            optimizer = ImageProcessor()
            optimized_path = optimizer.optimize_image(image_path)
            
            # Encode image
            image_content = optimizer.encode_image_base64(optimized_path)
            
            # Create message
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            image_content
                        ]
                    }
                ]
            )
            
            # Clean up optimized image if it's different from original
            if optimized_path != image_path and os.path.exists(optimized_path):
                try:
                    os.remove(optimized_path)
                except:
                    pass
            
            return response.content[0].text
        
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return f"Error analyzing image: {str(e)}"
    
    async def analyze_image_url(self, image_url: str, prompt: str) -> str:
        """
        Analyze an image from a URL using Claude
        
        Args:
            image_url: URL of the image
            prompt: Prompt for Claude describing what to analyze
            
        Returns:
            Claude's analysis of the image
        """
        try:
            # Create image content
            image_content = ImageProcessor.encode_image_url(image_url)
            
            # Create message
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            image_content
                        ]
                    }
                ]
            )
            
            return response.content[0].text
        
        except Exception as e:
            logger.error(f"Error analyzing image from URL: {str(e)}")
            return f"Error analyzing image from URL: {str(e)}"
    
    async def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from an image (OCR)
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Text extracted from the image
        """
        prompt = "Please extract all text from this image. Only include the text you can see, formatted exactly as it appears."
        return await self.analyze_image(image_path, prompt)
    
    async def describe_image(self, image_path: str, detail_level: str = "detailed") -> str:
        """
        Generate a description of an image
        
        Args:
            image_path: Path to the image file
            detail_level: Level of detail (brief, detailed, comprehensive)
            
        Returns:
            Description of the image
        """
        detail_prompts = {
            "brief": "Please provide a brief description of this image in 1-2 sentences.",
            "detailed": "Please provide a detailed description of this image. Include main subjects, actions, setting, and notable elements.",
            "comprehensive": "Please provide a comprehensive analysis of this image. Include main subjects, background elements, visual composition, colors, lighting, mood, possible context, and any text visible."
        }
        
        prompt = detail_prompts.get(detail_level, detail_prompts["detailed"])
        return await self.analyze_image(image_path, prompt)

# Image tools for the agent
def get_image_tools() -> List:
    """Get tools for image processing and analysis."""
    from anthropic_agent import Tool, ToolParameter
    
    # Initialize image analyzer
    image_analyzer = ImageAnalyzer()
    
    async def analyze_image_tool(image_path: str, prompt: str) -> Dict[str, Any]:
        """Analyze an image with a specific prompt."""
        result = await image_analyzer.analyze_image(image_path, prompt)
        return {
            "analysis": result
        }
    
    async def analyze_image_url_tool(image_url: str, prompt: str) -> Dict[str, Any]:
        """Analyze an image from a URL with a specific prompt."""
        result = await image_analyzer.analyze_image_url(image_url, prompt)
        return {
            "analysis": result
        }
    
    async def extract_text_from_image_tool(image_path: str) -> Dict[str, Any]:
        """Extract text from an image."""
        result = await image_analyzer.extract_text_from_image(image_path)
        return {
            "extracted_text": result
        }
    
    async def describe_image_tool(image_path: str, detail_level: str = "detailed") -> Dict[str, Any]:
        """Generate a description of an image."""
        result = await image_analyzer.describe_image(image_path, detail_level)
        return {
            "description": result
        }
    
    def optimize_image_tool(image_path: str, max_size_mb: float = 5.0) -> Dict[str, Any]:
        """Optimize an image for use with Claude."""
        try:
            optimized_path = ImageProcessor.optimize_image(
                image_path, 
                max_size_mb=max_size_mb
            )
            
            original_size = os.path.getsize(image_path) / (1024 * 1024)
            optimized_size = os.path.getsize(optimized_path) / (1024 * 1024)
            
            return {
                "original_path": image_path,
                "optimized_path": optimized_path,
                "original_size_mb": original_size,
                "optimized_size_mb": optimized_size,
                "reduction_percent": (1 - (optimized_size / original_size)) * 100 if original_size > 0 else 0
            }
        except Exception as e:
            return {
                "error": str(e)
            }
    
    def download_image_tool(image_url: str) -> Dict[str, Any]:
        """Download an image from a URL."""
        try:
            downloaded_path = ImageProcessor.download_image(image_url)
            
            return {
                "image_url": image_url,
                "downloaded_path": downloaded_path,
                "size_kb": os.path.getsize(downloaded_path) / 1024
            }
        except Exception as e:
            return {
                "error": str(e)
            }
    
    # Define tools
    return [
        Tool(
            name="analyze_image",
            description="Analyze an image with a specific prompt",
            parameters=[
                ToolParameter(name="image_path", type="string", description="Path to the image file"),
                ToolParameter(name="prompt", type="string", description="Prompt for Claude describing what to analyze")
            ],
            function=analyze_image_tool,
            category="image"
        ),
        
        Tool(
            name="analyze_image_url",
            description="Analyze an image from a URL with a specific prompt",
            parameters=[
                ToolParameter(name="image_url", type="string", description="URL of the image"),
                ToolParameter(name="prompt", type="string", description="Prompt for Claude describing what to analyze")
            ],
            function=analyze_image_url_tool,
            category="image"
        ),
        
        Tool(
            name="extract_text_from_image",
            description="Extract text from an image (OCR)",
            parameters=[
                ToolParameter(name="image_path", type="string", description="Path to the image file")
            ],
            function=extract_text_from_image_tool,
            category="image"
        ),
        
        Tool(
            name="describe_image",
            description="Generate a description of an image",
            parameters=[
                ToolParameter(name="image_path", type="string", description="Path to the image file"),
                ToolParameter(name="detail_level", type="string", description="Level of detail (brief, detailed, comprehensive)", required=False, default="detailed")
            ],
            function=describe_image_tool,
            category="image"
        ),
        
        Tool(
            name="optimize_image",
            description="Optimize an image for use with Claude",
            parameters=[
                ToolParameter(name="image_path", type="string", description="Path to the image file"),
                ToolParameter(name="max_size_mb", type="number", description="Maximum file size in MB", required=False, default=5.0)
            ],
            function=optimize_image_tool,
            category="image"
        ),
        
        Tool(
            name="download_image",
            description="Download an image from a URL",
            parameters=[
                ToolParameter(name="image_url", type="string", description="URL of the image to download")
            ],
            function=download_image_tool,
            category="image"
        )
    ]