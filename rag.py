"""
Retrieval-Augmented Generation (RAG) system for the Anthropic-powered Agent
"""

import os
import json
import uuid
import hashlib
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Document:
    """Document for the RAG system."""
    
    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None, id: Optional[str] = None):
        """Initialize a document."""
        self.id = id or str(uuid.uuid4())
        self.content = content
        self.metadata = metadata or {}
        self.chunks = []
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "chunks": self.chunks
        }

class Chunk:
    """Chunk of a document."""
    
    def __init__(self, content: str, document_id: str, metadata: Optional[Dict[str, Any]] = None, 
                 embedding: Optional[List[float]] = None, id: Optional[str] = None, index: int = 0):
        """Initialize a chunk."""
        self.id = id or str(uuid.uuid4())
        self.content = content
        self.document_id = document_id
        self.metadata = metadata or {}
        self.embedding = embedding
        self.index = index
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "document_id": self.document_id,
            "metadata": self.metadata,
            "embedding": self.embedding,
            "index": self.index
        }

class VectorStore:
    """Store for document vectors."""
    
    def __init__(self, storage_dir: str = "vector_store"):
        """Initialize vector store."""
        self.storage_dir = storage_dir
        self.documents_dir = os.path.join(storage_dir, "documents")
        self.chunks_dir = os.path.join(storage_dir, "chunks")
        
        # Create directories if they don't exist
        os.makedirs(self.documents_dir, exist_ok=True)
        os.makedirs(self.chunks_dir, exist_ok=True)
    
    def save_document(self, document: Document) -> str:
        """Save a document."""
        document_path = os.path.join(self.documents_dir, f"{document.id}.json")
        with open(document_path, 'w') as f:
            json.dump(document.dict(), f, indent=2)
        return document.id
    
    def save_chunk(self, chunk: Chunk) -> str:
        """Save a chunk."""
        chunk_path = os.path.join(self.chunks_dir, f"{chunk.id}.json")
        with open(chunk_path, 'w') as f:
            json.dump(chunk.dict(), f, indent=2)
        return chunk.id
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document."""
        document_path = os.path.join(self.documents_dir, f"{document_id}.json")
        if not os.path.exists(document_path):
            return None
        
        with open(document_path, 'r') as f:
            document_dict = json.load(f)
            document = Document(
                content=document_dict["content"],
                metadata=document_dict["metadata"],
                id=document_dict["id"]
            )
            document.chunks = document_dict["chunks"]
            return document
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk."""
        chunk_path = os.path.join(self.chunks_dir, f"{chunk_id}.json")
        if not os.path.exists(chunk_path):
            return None
        
        with open(chunk_path, 'r') as f:
            chunk_dict = json.load(f)
            return Chunk(
                content=chunk_dict["content"],
                document_id=chunk_dict["document_id"],
                metadata=chunk_dict["metadata"],
                embedding=chunk_dict["embedding"],
                id=chunk_dict["id"],
                index=chunk_dict["index"]
            )
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents."""
        documents = []
        for document_file in os.listdir(self.documents_dir):
            if document_file.endswith(".json"):
                document_id = document_file[:-5]  # Remove .json extension
                document = self.get_document(document_id)
                if document:
                    documents.append(document)
        return documents
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        document = self.get_document(document_id)
        if not document:
            return False
        
        # Delete document file
        document_path = os.path.join(self.documents_dir, f"{document_id}.json")
        if os.path.exists(document_path):
            os.remove(document_path)
        
        # Delete chunks
        for chunk_info in document.chunks:
            chunk_id = chunk_info["id"]
            chunk_path = os.path.join(self.chunks_dir, f"{chunk_id}.json")
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
        
        return True
    
    def search(self, query_embedding: List[float], top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar chunks using cosine similarity."""
        results = []
        
        # Load all chunks
        for chunk_file in os.listdir(self.chunks_dir):
            if chunk_file.endswith(".json"):
                chunk_path = os.path.join(self.chunks_dir, chunk_file)
                with open(chunk_path, 'r') as f:
                    chunk_dict = json.load(f)
                    
                    # Apply filters if provided
                    if filters:
                        metadata = chunk_dict.get("metadata", {})
                        if not all(metadata.get(key) == value for key, value in filters.items()):
                            continue
                    
                    # Calculate similarity if embedding exists
                    if "embedding" in chunk_dict and chunk_dict["embedding"]:
                        chunk_embedding = chunk_dict["embedding"]
                        similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                        
                        chunk = Chunk(
                            content=chunk_dict["content"],
                            document_id=chunk_dict["document_id"],
                            metadata=chunk_dict["metadata"],
                            embedding=chunk_dict["embedding"],
                            id=chunk_dict["id"],
                            index=chunk_dict["index"]
                        )
                        
                        results.append({
                            "chunk": chunk,
                            "score": similarity
                        })
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top k results
        return results[:top_k]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)

class EmbeddingGenerator:
    """Generate embeddings for text using Claude API."""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229"):
        """Initialize embedding generator."""
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            response = await self.client.embeddings.create(
                model="claude-3-sonnet-20240229-embedding",
                input=text,
                dimensions=1536
            )
            
            return response.embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * 1536

class DocumentProcessor:
    """Process documents for the RAG system."""
    
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator):
        """Initialize document processor."""
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
    
    async def process_document(self, document: Document, chunk_strategy: str = "tokens") -> Document:
        """Process a document by chunking and generating embeddings."""
        # Chunk the document
        chunks = await self.chunk_document(document, chunk_strategy)
        
        # Save chunks to vector store
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = await self.embedding_generator.generate_embedding(chunk.content)
            chunk.embedding = embedding
            
            # Save chunk
            self.vector_store.save_chunk(chunk)
            
            # Add chunk info to document
            document.chunks.append({
                "id": chunk.id,
                "index": i
            })
        
        # Save document
        self.vector_store.save_document(document)
        
        return document
    
    async def chunk_document(self, document: Document, strategy: str = "tokens") -> List[Chunk]:
        """Chunk a document using the specified strategy."""
        content = document.content
        
        if strategy == "tokens":
            # Chunk by approximate token count (rough estimation)
            chunk_size = 1000  # tokens
            words = content.split()
            chunks_text = []
            
            current_chunk = []
            current_size = 0
            
            for word in words:
                # Rough estimation: 1 token ≈ 0.75 words
                word_tokens = len(word) // 3 + 1
                
                if current_size + word_tokens > chunk_size and current_chunk:
                    chunks_text.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_size = word_tokens
                else:
                    current_chunk.append(word)
                    current_size += word_tokens
            
            if current_chunk:
                chunks_text.append(" ".join(current_chunk))
        
        elif strategy == "paragraphs":
            # Chunk by paragraphs
            paragraphs = [p for p in content.split("\n\n") if p.strip()]
            
            # Combine short paragraphs
            chunks_text = []
            current_chunk = ""
            
            for paragraph in paragraphs:
                if len(current_chunk) + len(paragraph) < 2000:
                    current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                else:
                    if current_chunk:
                        chunks_text.append(current_chunk)
                    current_chunk = paragraph
            
            if current_chunk:
                chunks_text.append(current_chunk)
        
        elif strategy == "sentences":
            # Chunk by sentences (approximate)
            import re
            sentences = re.split(r'(?<=[.!?])\s+', content)
            
            # Combine short sentences
            chunks_text = []
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < 1000:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        chunks_text.append(current_chunk)
                    current_chunk = sentence
            
            if current_chunk:
                chunks_text.append(current_chunk)
        
        else:
            # Default to tokens strategy
            return await self.chunk_document(document, "tokens")
        
        # Create chunk objects
        chunks = []
        for i, chunk_text in enumerate(chunks_text):
            # Create a unique but deterministic ID based on content
            content_hash = hashlib.md5(chunk_text.encode()).hexdigest()
            chunk_id = f"{document.id}_{content_hash}_{i}"
            
            chunk = Chunk(
                content=chunk_text,
                document_id=document.id,
                metadata=document.metadata,
                id=chunk_id,
                index=i
            )
            
            chunks.append(chunk)
        
        return chunks

class RAGRetriever:
    """Retrieve relevant chunks for a query."""
    
    def __init__(self, vector_store: VectorStore, embedding_generator: EmbeddingGenerator):
        """Initialize RAG retriever."""
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
    
    async def search(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for relevant chunks for a query."""
        # Generate embedding for query
        query_embedding = await self.embedding_generator.generate_embedding(query)
        
        # Search vector store
        search_results = self.vector_store.search(query_embedding, top_k, filters)
        
        # Format results
        formatted_results = []
        for result in search_results:
            chunk = result["chunk"]
            formatted_results.append({
                "content": chunk.content,
                "score": result["score"],
                "metadata": {
                    **chunk.metadata,
                    "document_id": chunk.document_id
                }
            })
        
        return formatted_results

class RAGSystem:
    """Complete RAG system combining all components."""
    
    def __init__(self, storage_dir: str = "vector_store", embedding_model: str = "claude-3-sonnet-20240229"):
        """Initialize the RAG system."""
        self.vector_store = VectorStore(storage_dir)
        self.embedding_generator = EmbeddingGenerator(model=embedding_model)
        self.document_processor = DocumentProcessor(self.vector_store, self.embedding_generator)
        self.retriever = RAGRetriever(self.vector_store, self.embedding_generator)
    
    async def add_document(self, content: str, metadata: Dict[str, Any] = None, 
                      chunk_strategy: str = "tokens") -> Document:
        """Add a document to the RAG system."""
        document = Document(content=content, metadata=metadata)
        return await self.document_processor.process_document(document, chunk_strategy)
    
    async def add_document_from_file(self, file_path: str, metadata: Dict[str, Any] = None, 
                                chunk_strategy: str = "tokens") -> Document:
        """Add a document from a file to the RAG system."""
        # Read file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Add file metadata
        file_metadata = {
            "source": file_path,
            "filename": os.path.basename(file_path),
            "filetype": os.path.splitext(file_path)[1],
            "added_at": datetime.now().isoformat()
        }
        
        combined_metadata = {**file_metadata}
        if metadata:
            combined_metadata.update(metadata)
        
        # Process document
        return await self.add_document(content, combined_metadata, chunk_strategy)
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get a document from the RAG system."""
        return self.vector_store.get_document(document_id)
    
    def get_all_documents(self) -> List[Document]:
        """Get all documents from the RAG system."""
        return self.vector_store.get_all_documents()
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from the RAG system."""
        return self.vector_store.delete_document(document_id)
    
    async def query(self, query: str, top_k: int = 5, 
                  filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the RAG system."""
        results = await self.retriever.search(query, top_k, filters)
        
        return {
            "query": query,
            "results": results
        }
    
    async def get_query_context(self, query: str, top_k: int = 5, 
                              filters: Optional[Dict[str, Any]] = None) -> str:
        """Get formatted context for a query to inject into a prompt."""
        results = await self.query(query, top_k, filters)
        
        formatted_text = "RELEVANT CONTEXT:\n\n"
        
        for i, result in enumerate(results["results"]):
            formatted_text += f"[Document {i+1}] (Relevance: {result['score']:.4f})\n"
            formatted_text += f"{result['content']}\n\n"
        
        return formatted_text
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        documents = self.get_all_documents()
        
        # Count chunks
        chunk_count = sum(len(doc.chunks) for doc in documents)
        
        # Calculate storage size
        documents_dir = self.vector_store.documents_dir
        chunks_dir = self.vector_store.chunks_dir
        
        documents_size = sum(os.path.getsize(os.path.join(documents_dir, f)) for f in os.listdir(documents_dir) if os.path.isfile(os.path.join(documents_dir, f)))
        chunks_size = sum(os.path.getsize(os.path.join(chunks_dir, f)) for f in os.listdir(chunks_dir) if os.path.isfile(os.path.join(chunks_dir, f)))
        total_size = documents_size + chunks_size
        
        # Find last updated
        last_updated = max([datetime.fromtimestamp(os.path.getmtime(os.path.join(documents_dir, f))).isoformat() 
                          for f in os.listdir(documents_dir) if os.path.isfile(os.path.join(documents_dir, f)) and os.listdir(documents_dir)], 
                         default=None)
        
        return {
            "document_count": len(documents),
            "chunk_count": chunk_count,
            "storage_size_bytes": total_size,
            "last_updated": last_updated
        }

def get_rag_tools() -> List:
    """Get tools for the RAG system."""
    from anthropic_agent import Tool, ToolParameter
    
    # Initialize RAG system
    rag_system = RAGSystem()
    
    async def add_document_to_rag(content: str, metadata: Optional[Dict[str, Any]] = None, 
                                 chunk_strategy: str = "tokens") -> Dict[str, Any]:
        """Add a document to the RAG system."""
        try:
            document = await rag_system.add_document(content, metadata, chunk_strategy)
            
            return {
                "document_id": document.id,
                "chunk_count": len(document.chunks),
                "metadata": document.metadata
            }
        except Exception as e:
            logger.error(f"Error adding document to RAG: {str(e)}")
            return {"error": str(e)}
    
    async def add_file_to_rag(file_path: str, metadata: Optional[Dict[str, Any]] = None, 
                             chunk_strategy: str = "tokens") -> Dict[str, Any]:
        """Add a file to the RAG system."""
        try:
            document = await rag_system.add_document_from_file(file_path, metadata, chunk_strategy)
            
            return {
                "document_id": document.id,
                "file": os.path.basename(file_path),
                "chunk_count": len(document.chunks),
                "metadata": document.metadata
            }
        except Exception as e:
            logger.error(f"Error adding file to RAG: {str(e)}")
            return {"error": str(e)}
    
    def get_document_from_rag(document_id: str) -> Dict[str, Any]:
        """Get a document from the RAG system."""
        try:
            document = rag_system.get_document(document_id)
            
            if document:
                return {
                    "document_id": document.id,
                    "content": document.content,
                    "chunk_count": len(document.chunks),
                    "metadata": document.metadata
                }
            else:
                return {"error": f"Document {document_id} not found"}
        
        except Exception as e:
            logger.error(f"Error getting document from RAG: {str(e)}")
            return {"error": str(e)}
    
    def list_rag_documents() -> Dict[str, Any]:
        """List all documents in the RAG system."""
        try:
            documents = rag_system.get_all_documents()
            
            return {
                "document_count": len(documents),
                "documents": [
                    {
                        "id": doc.id,
                        "content_preview": doc.content[:100] + "..." if len(doc.content) > 100 else doc.content,
                        "chunk_count": len(doc.chunks),
                        "metadata": doc.metadata
                    }
                    for doc in documents
                ]
            }
        
        except Exception as e:
            logger.error(f"Error listing RAG documents: {str(e)}")
            return {"error": str(e)}
    
    def delete_rag_document(document_id: str) -> Dict[str, Any]:
        """Delete a document from the RAG system."""
        try:
            success = rag_system.delete_document(document_id)
            
            if success:
                return {"success": True, "document_id": document_id}
            else:
                return {"error": f"Document {document_id} not found"}
        
        except Exception as e:
            logger.error(f"Error deleting RAG document: {str(e)}")
            return {"error": str(e)}
    
    async def query_rag(query: str, top_k: int = 5, 
                      filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the RAG system."""
        try:
            results = await rag_system.query(query, top_k, filters)
            return results
        
        except Exception as e:
            logger.error(f"Error querying RAG: {str(e)}")
            return {"error": str(e)}
    
    def get_rag_stats() -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        try:
            return rag_system.get_stats()
        
        except Exception as e:
            logger.error(f"Error getting RAG stats: {str(e)}")
            return {"error": str(e)}
    
    # Define tools
    return [
        Tool(
            name="add_document_to_rag",
            description="Add a document to the RAG system",
            parameters=[
                ToolParameter(name="content", type="string", description="Document content"),
                ToolParameter(name="metadata", type="object", description="Document metadata", required=False),
                ToolParameter(name="chunk_strategy", type="string", description="Chunking strategy (tokens, paragraphs, sentences)", required=False, default="tokens")
            ],
            function=add_document_to_rag,
            category="rag"
        ),
        
        Tool(
            name="add_file_to_rag",
            description="Add a file to the RAG system",
            parameters=[
                ToolParameter(name="file_path", type="string", description="Path to the file"),
                ToolParameter(name="metadata", type="object", description="Additional metadata", required=False),
                ToolParameter(name="chunk_strategy", type="string", description="Chunking strategy (tokens, paragraphs, sentences)", required=False, default="tokens")
            ],
            function=add_file_to_rag,
            category="rag"
        ),
        
        Tool(
            name="get_document_from_rag",
            description="Get a document from the RAG system",
            parameters=[
                ToolParameter(name="document_id", type="string", description="Document ID")
            ],
            function=get_document_from_rag,
            category="rag"
        ),
        
        Tool(
            name="list_rag_documents",
            description="List all documents in the RAG system",
            parameters=[],
            function=list_rag_documents,
            category="rag"
        ),
        
        Tool(
            name="delete_rag_document",
            description="Delete a document from the RAG system",
            parameters=[
                ToolParameter(name="document_id", type="string", description="Document ID")
            ],
            function=delete_rag_document,
            category="rag"
        ),
        
        Tool(
            name="query_rag",
            description="Query the RAG system",
            parameters=[
                ToolParameter(name="query", type="string", description="Query text"),
                ToolParameter(name="top_k", type="integer", description="Number of results to return", required=False, default=5),
                ToolParameter(name="filters", type="object", description="Metadata filters", required=False)
            ],
            function=query_rag,
            category="rag"
        ),
        
        Tool(
            name="get_rag_stats",
            description="Get statistics about the RAG system",
            parameters=[],
            function=get_rag_stats,
            category="rag"
        )
    ]