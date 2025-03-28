"""
Unit tests for image processing
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import json
import tempfile
from pathlib import Path
import base64
import asyncio

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from image_processing import ImageProcessor, ImageAnalyzer, get_image_tools

class TestImageProcessor(unittest.TestCase):
    """Test cases for the ImageProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'ANTHROPIC_API_KEY': 'mock-api-key',
        })
        self.env_patcher.start()
        
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(b'fake image data')
            self.temp_image_path = f.name
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
        if os.path.exists(self.temp_image_path):
            os.unlink(self.temp_image_path)
    
    def test_encode_image_base64(self):
        """Test encoding an image as base64."""
        # Encode the image
        image_content = ImageProcessor.encode_image_base64(self.temp_image_path)
        
        # Check the result
        self.assertEqual(image_content["type"], "image")
        self.assertEqual(image_content["source"]["type"], "base64")
        self.assertEqual(image_content["source"]["media_type"], "image/jpeg")
        self.assertIsNotNone(image_content["source"]["data"])
        
        # Decode the base64 data
        decoded_data = base64.b64decode(image_content["source"]["data"])
        
        # Check that the decoded data matches the original
        with open(self.temp_image_path, 'rb') as f:
            original_data = f.read()
        self.assertEqual(decoded_data, original_data)
    
    def test_encode_image_url(self):
        """Test creating an image object from a URL."""
        # Create an image object from a URL
        image_url = "https://example.com/image.jpg"
        image_content = ImageProcessor.encode_image_url(image_url)
        
        # Check the result
        self.assertEqual(image_content["type"], "image")
        self.assertEqual(image_content["source"]["type"], "url")
        self.assertEqual(image_content["source"]["url"], image_url)
    
    @patch('PIL.Image.open')
    def test_optimize_image(self, mock_open):
        """Test optimizing an image."""
        # Mock PIL.Image.open
        mock_img = MagicMock()
        mock_img.format = "JPEG"
        mock 