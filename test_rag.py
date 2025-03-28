"""
Unit tests for the RAG system
"""

import os
import sys
import unittest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import json
import tempfile
from pathlib import Path
import numpy as np

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag import RAGSystem, Document, EmbeddingGenerator, VectorStore, DocumentProcessor, RAGRetriever
from rag_enhancements import DocumentProcessor as EnhancedDocumentProcessor
from rag_enhancements import HybridSearcher, DocumentCollection, CollectionManager

class TestRAGSystem(unittest.TestCase):
    """Test cases for the RAG system."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock environment variables
        self.env_patcher = patch.dict('os.environ', {
            'ANTHROPIC_API_KEY': 'mock-api-key',
        })
        self.env_patcher.start()
        
        # Create temporary directory for storage
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create mock RAG components
        self.mock_embedding_generator = MagicMock(spec=EmbeddingGenerator)
        self.mock_embedding_generator.generate_embedding.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        self.mock_vector_store = MagicMock(spec=VectorStore)
        self.mock_vector_store.storage_dir = Path(self.temp_dir.name)
        self.mock_vector_store.documents_dir = Path(self.temp_dir.name) / "documents"
        self.mock_vector_store.chunks_dir = Path(self.temp_dir.name) / "chunks"
        
        # Create directories
        self.mock_vector_store.documents_dir.mkdir(parents=True, exist_ok=True)
        self.mock_vector_store.chunks_dir.mkdir(parents=True, exist_ok=True)
        
        self.mock_document_processor = MagicMock(spec=DocumentProcessor)
        self.mock_retriever = MagicMock(spec=RAGRetriever)
        
        # Create RAG system with mocked components
        self.rag_system = RAGSystem()
        self.rag_system.embedding_generator = self.mock_embedding_generator
        self.rag_system.vector_store = self.mock_vector_store
        self.rag_system.document_processor = self.mock_document_processor
        self.rag_system.retriever = self.mock_retriever
    
    def tearDown(self):
        """Clean up after tests."""
        self.env_patcher.stop()
        self.temp_dir.cleanup()
    
    async def test_add_document(self):
        """Test adding a document."""
        # Mock the document processor
        mock_document = Document(
            id="doc123",
            content="This is a test document.",
            metadata={"source": "test"},
            chunks=[{"id": "chunk1", "content": "This is a test", "embedding": [0.1, 0.2, 0.3]}]
        )
        self.mock_document_processor.process_document.return_value = mock_document
        
        # Add a document
        document = await self.rag_system.add_document(
            content="This is a test document.",
            metadata={"source": "test"},
            chunk_strategy="tokens"
        )
        
        # Check that the document processor was called
        self.mock_document_processor.process_document.assert_called_once()
        
        # Check the document
        self.assertEqual(document.id, "doc123")
        self.assertEqual(document.content, "This is a test document.")
        self.assertEqual(document.metadata, {"source": "test"})
        self.assertEqual(len(document.chunks), 1)
    
    async def test_query(self):
        """Test querying the RAG system."""
        # Mock retriever results
        mock_results = [
            {"content": "Result 1", "score": 0.9, "metadata": {"source": "doc1"}},
            {"content": "Result 2", "score": 0.8, "metadata": {"source": "doc2"}}
        ]
        self.mock_retriever.retrieve.return_value = mock_results
        
        # Query the system
        results = await self.rag_system.query("test query", top_k=2)
        
        # Check that the retriever was called
        self.mock_retriever.retrieve.assert_called_once()
        
        # Check the results
        self.assertEqual(results["query"], "test query")
        self.assertEqual(len(results["results"]), 2)
        self.assertEqual(results["results"][0]["content"], "Result 1")
        self.assertEqual(results["results"][0]["score"], 0.9)
        self.assertEqual(results["results"][0]["metadata"]["source"], "doc1")
        self.assertEqual(results["results"][1]["content"], "Result 2")
        self.assertEqual(results["results"][1]["score"], 0.8)
        self.assertEqual(results["results"][1]["metadata"]["source"], "doc2")
    
    def test_get_document(self):
        """Test getting a document."""
        # Mock the vector store
        mock_document = {
            "id": "doc123",
            "content": "This is a test document.",
            "metadata": {"source": "test"},
            "chunks": [{"id": "chunk1", "content": "This is a test", "embedding": [0.1, 0.2, 0.3]}]
        }
        self.mock_vector_store.get_document.return_value = mock_document
        
        # Get a document
        document = self.rag_system.get_document("doc123")
        
        # Check that the vector store was called
        self.mock_vector_store.get_document.assert_called_once_with("doc123")
        
        # Check the document
        self.assertEqual(document["id"], "doc123")
        self.assertEqual(document["content"], "This is a test document.")
        self.assertEqual(document["metadata"], {"source": "test"})
        self.assertEqual(len(document["chunks"]), 1)
    
    def test_delete_document(self):
        """Test deleting a document."""
        # Mock the vector store
        self.mock_vector_store.delete_document.return_value = True
        
        # Delete a document
        success = self.rag_system.delete_document("doc123")
        
        # Check that the vector store was called
        self.mock_vector_store.delete_document.assert_called_once_with("doc123")
        
        # Check the result
        self.assertTrue(success)
    
    def test_get_all_documents(self):
        """Test getting all documents."""
        # Mock the vector store
        mock_documents = [
            {
                "id": "doc1",
                "content": "Document 1",
                "metadata": {"source": "test1"},
                "chunks": []
            },
            {
                "id": "doc2",
                "content": "Document 2",
                "metadata": {"source": "test2"},
                "chunks": []
            }
        ]
        self.mock_vector_store.get_all_documents.return_value = mock_documents
        
        # Get all documents
        documents = self.rag_system.get_all_documents()
        
        # Check that the vector store was called
        self.mock_vector_store.get_all_documents.assert_called_once()
        
        # Check the documents
        self.assertEqual(len(documents), 2)
        self.assertEqual(documents[0]["id"], "doc1")
        self.assertEqual(documents[0]["content"], "Document 1")
        self.assertEqual(documents[1]["id"], "doc2")
        self.assertEqual(documents[1]["content"], "Document 2")

class TestEnhancedDocumentProcessor(unittest.TestCase):
    """Test cases for the enhanced document processor."""
    
    def test_extract_text_from_file(self):
        """Test extracting text from different file types."""
        # Create temporary text file
        with tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False) as temp_file:
            temp_file.write("This is a test file.")
            temp_path = temp_file.name
        
        try:
            # Extract text from the file
            text, metadata = EnhancedDocumentProcessor.extract_text_from_file(temp_path)
            
            # Check the extracted text
            self.assertEqual(text, "This is a test file.")
            
            # Check metadata
            self.assertEqual(metadata["filename"], os.path.basename(temp_path))
            self.assertEqual(metadata["extension"], ".txt")
            self.assertEqual(metadata["type"], "text")
            
            # Check word and character counts
            self.assertEqual(metadata["word_count"], 5)
            self.assertEqual(metadata["char_count"], 19)
        
        finally:
            # Clean up
            os.unlink(temp_path)

class TestHybridSearcher(unittest.TestCase):
    """Test cases for the hybrid search functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create mock vector store
        self.mock_vector_store = MagicMock()
        
        # Create hybrid searcher
        self.hybrid_searcher = HybridSearcher(self.mock_vector_store)
    
    def test_keyword_search(self):
        """Test keyword search functionality."""
        # Create test chunks
        chunks = [
            {"id": "chunk1", "content": "This is a test document about AI and machine learning.", "metadata": {}},
            {"id": "chunk2", "content": "Machine learning is a subset of artificial intelligence.", "metadata": {}},
            {"id": "chunk3", "content": "This document has nothing relevant.", "metadata": {}}
        ]
        
        # Perform keyword search
        results = self.hybrid_searcher._keyword_search("machine learning AI", chunks, top_k=2)
        
        # Check results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk"]["id"], "chunk2")  # Should be first due to more keyword matches
        self.assertEqual(results[1]["chunk"]["id"], "chunk1")

class TestDocumentCollection(unittest.TestCase):
    """Test cases for document collections."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create collection
        self.collection = DocumentCollection("test_collection", self.temp_dir.name)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_add_document(self):
        """Test adding a document to a collection."""
        # Create a mock document
        mock_document = MagicMock()
        mock_document.id = "doc123"
        mock_document.dict.return_value = {
            "id": "doc123",
            "content": "Test document",
            "metadata": {},
            "chunks": []
        }
        
        # Add document to collection
        doc_id = self.collection.add_document(mock_document)
        
        # Check result
        self.assertEqual(doc_id, "doc123")
        
        # Check that the document was saved
        document_path = self.collection.documents_dir / "doc123.json"
        self.assertTrue(document_path.exists())
        
        # Check metadata
        self.assertEqual(self.collection.metadata["document_count"], 1)
    
    def test_get_document(self):
        """Test getting a document from a collection."""
        # Create and add a mock document
        document_data = {
            "id": "doc123",
            "content": "Test document",
            "metadata": {},
            "chunks": []
        }
        
        document_path = self.collection.documents_dir / "doc123.json"
        with open(document_path, 'w') as f:
            json.dump(document_data, f)
        
        # Get the document
        document = self.collection.get_document("doc123")
        
        # Check the document
        self.assertEqual(document["id"], "doc123")
        self.assertEqual(document["content"], "Test document")
    
    def test_delete_document(self):
        """Test deleting a document from a collection."""
        # Create and add a mock document
        document_data = {
            "id": "doc123",
            "content": "Test document",
            "metadata": {},
            "chunks": []
        }
        
        document_path = self.collection.documents_dir / "doc123.json"
        with open(document_path, 'w') as f:
            json.dump(document_data, f)
        
        # Update metadata
        self.collection.metadata["document_count"] = 1
        self.collection._save_metadata()
        
        # Delete the document
        success = self.collection.delete_document("doc123")
        
        # Check result
        self.assertTrue(success)
        
        # Check that the document was deleted
        self.assertFalse(document_path.exists())
        
        # Check metadata
        self.assertEqual(self.collection.metadata["document_count"], 0)

class TestCollectionManager(unittest.TestCase):
    """Test cases for the collection manager."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create collection manager
        self.manager = CollectionManager(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def test_create_collection(self):
        """Test creating a collection."""
        # Create a new collection
        collection = self.manager.create_collection("test_collection")
        
        # Check the collection
        self.assertEqual(collection.name, "test_collection")
        
        # Check that the collection was added to the manager
        self.assertIn("test_collection", self.manager.collections)
        
        # Check metadata
        self.assertIn("test_collection", self.manager.metadata["collections"])
    
    def test_get_collection(self):
        """Test getting a collection."""
        # Create a new collection
        self.manager.create_collection("test_collection")
        
        # Get the collection
        collection = self.manager.get_collection("test_collection")
        
        # Check the collection
        self.assertEqual(collection.name, "test_collection")
    
    def test_delete_collection(self):
        """Test deleting a collection."""
        # Create a new collection
        self.manager.create_collection("test_collection")
        
        # Delete the collection
        success = self.manager.delete_collection("test_collection")
        
        # Check result
        self.assertTrue(success)
        
        # Check that the collection was removed
        self.assertNotIn("test_collection", self.manager.collections)
        
        # Check metadata
        self.assertNotIn("test_collection", self.manager.metadata["collections"])
    
    def test_list_collections(self):
        """Test listing collections."""
        # Create collections
        self.manager.create_collection("collection1")
        self.manager.create_collection("collection2")
        
        # List collections
        collections = self.manager.list_collections()
        
        # Check the list
        self.assertEqual(len(collections), 3)  # default + 2 new ones
        
        # Check collection names
        collection_names = [c["name"] for c in collections]
        self.assertIn("default", collection_names)
        self.assertIn("collection1", collection_names)
        self.assertIn("collection2", collection_names)
    
    def test_set_default_collection(self):
        """Test setting the default collection."""
        # Create a new collection
        self.manager.create_collection("test_collection")
        
        # Set as default
        success = self.manager.set_default_collection("test_collection")
        
        # Check result
        self.assertTrue(success)
        
        # Check that the default was updated
        self.assertEqual(self.manager.metadata["default_collection"], "test_collection")
        
        # Get default collection
        default = self.manager.get_default_collection()
        
        # Check the default collection
        self.assertEqual(default.name, "test_collection")