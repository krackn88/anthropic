"""
Command Line Interface for the RAG system
"""

import os
import sys
import json
import asyncio
import argparse
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGCLI:
    """Command Line Interface for the RAG system."""
    
    def __init__(self, rag_system):
        """Initialize with a RAG system."""
        self.rag_system = rag_system
    
    async def add_document(self, args: argparse.Namespace):
        """Add a document to the RAG system."""
        # Handle file input
        if args.file:
            metadata = {}
            if args.metadata:
                try:
                    metadata = json.loads(args.metadata)
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON metadata: {args.metadata}")
                    return
            
            print(f"Adding file to RAG system: {args.file}")
            document = await self.rag_system.add_document_from_file(args.file, metadata, args.chunk_strategy)
            
            print(f"File added successfully.")
            print(f"Document ID: {document.id}")
            print(f"Chunks: {len(document.chunks)}")
            print(f"Metadata: {json.dumps(document.metadata, indent=2)}")
        
        # Handle text input
        elif args.text:
            metadata = {}
            if args.metadata:
                try:
                    metadata = json.loads(args.metadata)
                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON metadata: {args.metadata}")
                    return
            
            print(f"Adding text to RAG system (first 100 chars): {args.text[:100]}...")
            document = await self.rag_system.add_document(args.text, metadata, args.chunk_strategy)
            
            print(f"Text added successfully.")
            print(f"Document ID: {document.id}")
            print(f"Chunks: {len(document.chunks)}")
            print(f"Metadata: {json.dumps(document.metadata, indent=2)}")
        
        else:
            print("Error: Either --file or --text must be specified.")
            return
    
    async def query(self, args: argparse.Namespace):
        """Query the RAG system."""
        filters = None
        if args.filters:
            try:
                filters = json.loads(args.filters)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON filters: {args.filters}")
                return
        
        print(f"Querying RAG system: {args.query}")
        results = await self.rag_system.query(args.query, args.top_k, filters)
        
        print(f"Found {len(results['results'])} results:")
        
        for i, result in enumerate(results["results"]):
            print(f"\nResult {i+1} (Score: {result['score']:.4f}):")
            print(f"Content: {result['content'][:200]}...")
            print(f"Metadata: {json.dumps(result['metadata'], indent=2)}")
    
    def list_documents(self, args: argparse.Namespace):
        """List all documents in the RAG system."""
        documents = self.rag_system.get_all_documents()
        
        print(f"Found {len(documents)} documents:")
        
        for i, doc in enumerate(documents):
            print(f"\nDocument {i+1}:")
            print(f"ID: {doc.id}")
            print(f"Content (first 100 chars): {doc.content[:100]}...")
            print(f"Chunks: {len(doc.chunks)}")
            print(f"Metadata: {json.dumps(doc.metadata, indent=2)}")
    
    def get_document(self, args: argparse.Namespace):
        """Get a document from the RAG system."""
        document = self.rag_system.get_document(args.id)
        
        if not document:
            print(f"Error: Document {args.id} not found.")
            return
        
        print(f"Document ID: {document.id}")
        print(f"Chunks: {len(document.chunks)}")
        print(f"Metadata: {json.dumps(document.metadata, indent=2)}")
        
        if args.full:
            print(f"\nContent:")
            print(document.content)
        else:
            print(f"\nContent (first 500 chars):")
            print(f"{document.content[:500]}...")
            print("\nUse --full to see the entire content.")
    
    def delete_document(self, args: argparse.Namespace):
        """Delete a document from the RAG system."""
        if not args.force:
            confirm = input(f"Are you sure you want to delete document {args.id}? (y/n): ")
            if confirm.lower() != 'y':
                print("Deletion cancelled.")
                return
        
        success = self.rag_system.delete_document(args.id)
        
        if success:
            print(f"Document {args.id} deleted successfully.")
        else:
            print(f"Error: Document {args.id} not found.")
    
    def get_stats(self, args: argparse.Namespace):
        """Get statistics about the RAG system."""
        stats = self.rag_system.get_stats()
        
        print("RAG System Statistics:")
        print(f"Documents: {stats['document_count']}")
        print(f"Chunks: {stats['chunk_count']}")
        
        if 'storage_size_bytes' in stats:
            size_mb = stats['storage_size_bytes'] / (1024 * 1024)
            print(f"Storage Size: {size_mb:.2f} MB")
        
        if 'last_updated' in stats:
            print(f"Last Updated: {stats['last_updated']}")

async def main():
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="RAG System CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add document command
    add_parser = subparsers.add_parser("add", help="Add a document to the RAG system")
    add_parser.add_argument("--file", type=str, help="Path to file to add")
    add_parser.add_argument("--text", type=str, help="Text content to add")
    add_parser.add_argument("--metadata", type=str, help="JSON metadata")
    add_parser.add_argument("--chunk-strategy", type=str, default="tokens", 
                            choices=["tokens", "paragraphs", "sentences"],
                            help="Chunking strategy")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("query", type=str, help="Query text")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    query_parser.add_argument("--filters", type=str, help="JSON metadata filters")
    
    # List documents command
    list_parser = subparsers.add_parser("list", help="List all documents in the RAG system")
    
    # Get document command
    get_parser = subparsers.add_parser("get", help="Get a document from the RAG system")
    get_parser.add_argument("id", type=str, help="Document ID")
    get_parser.add_argument("--full", action="store_true", help="Show full content")
    
    # Delete document command
    delete_parser = subparsers.add_parser("delete", help="Delete a document from the RAG system")
    delete_parser.add_argument("id", type=str, help="Document ID")
    delete_parser.add_argument("--force", action="store_true", help="Delete without confirmation")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get statistics about the RAG system")
    
    args = parser.parse_args()
    
    # Initialize RAG system
    from rag import RAGSystem
    rag_system = RAGSystem()
    
    # Initialize CLI
    cli = RAGCLI(rag_system)
    
    # Execute command
    if args.command == "add":
        await cli.add_document(args)
    elif args.command == "query":
        await cli.query(args)
    elif args.command == "list":
        cli.list_documents(args)
    elif args.command == "get":
        cli.get_document(args)
    elif args.command == "delete":
        cli.delete_document(args)
    elif args.command == "stats":
        cli.get_stats(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())