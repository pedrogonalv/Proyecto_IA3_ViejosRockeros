#!/usr/bin/env python3
"""
ChromaDB Connector Example for Agent Integration

This example demonstrates how to connect to the ChromaDB vector database
and perform basic operations like searching and retrieving documents.

The ChromaDB SQLite database structure:
- Location: /Users/santiagojorda/Downloads/clode_technical_rag_system/data/vectordb/chroma.sqlite3
- Collections:
  - "technical_manuals" - Main text chunks collection
  - "technical_manuals_images" - Image references collection
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import Config


class ChromaDBConnector:
    """Simple ChromaDB connector for agent integration"""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize ChromaDB connector
        
        Args:
            config: Configuration object (uses default if not provided)
        """
        self.config = config or Config()
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.config.CHROMA_PERSIST_DIRECTORY,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get collections
        self.text_collection = self.client.get_collection(
            name=self.config.CHROMA_COLLECTION_NAME
        )
        
        self.image_collection = self.client.get_collection(
            name=f"{self.config.CHROMA_COLLECTION_NAME}_images"
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        
        print(f"Connected to ChromaDB at: {self.config.CHROMA_PERSIST_DIRECTORY}")
        print(f"Text collection: {self.text_collection.name} (count: {self.text_collection.count()})")
        print(f"Image collection: {self.image_collection.name} (count: {self.image_collection.count()})")
    
    def search(self, query: str, n_results: int = 5, collection: str = "text") -> Dict:
        """
        Search for documents in ChromaDB
        
        Args:
            query: Search query
            n_results: Number of results to return
            collection: Which collection to search ("text" or "images")
            
        Returns:
            Dictionary with search results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Select collection
        target_collection = self.text_collection if collection == "text" else self.image_collection
        
        # Perform search
        results = target_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['metadatas', 'documents', 'distances']
        )
        
        return self._format_results(results)
    
    def search_with_filter(self, query: str, filters: Dict, n_results: int = 5) -> Dict:
        """
        Search with metadata filters
        
        Args:
            query: Search query
            filters: Metadata filters (e.g., {"manual_name": "AX5000_SystemManual_V2_5"})
            n_results: Number of results
            
        Returns:
            Filtered search results
        """
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        results = self.text_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filters,
            include=['metadatas', 'documents', 'distances']
        )
        
        return self._format_results(results)
    
    def get_by_id(self, doc_id: str) -> Dict:
        """
        Get a specific document by ID
        
        Args:
            doc_id: Document ID (e.g., "chunk_123")
            
        Returns:
            Document data
        """
        result = self.text_collection.get(
            ids=[doc_id],
            include=['metadatas', 'documents', 'embeddings']
        )
        
        if result['ids']:
            return {
                'id': result['ids'][0],
                'document': result['documents'][0],
                'metadata': result['metadatas'][0]
            }
        return None
    
    def list_manuals(self) -> List[str]:
        """
        Get list of all manuals in the database
        
        Returns:
            List of manual names
        """
        # Get a sample of documents to extract manual names
        # ChromaDB doesn't have a direct way to get unique metadata values
        sample = self.text_collection.get(limit=10000)
        
        manuals = set()
        for metadata in sample['metadatas']:
            if 'manual_name' in metadata:
                manuals.add(metadata['manual_name'])
        
        return sorted(list(manuals))
    
    def get_manual_stats(self, manual_name: str) -> Dict:
        """
        Get statistics for a specific manual
        
        Args:
            manual_name: Name of the manual
            
        Returns:
            Statistics dictionary
        """
        # Get all chunks for this manual
        results = self.text_collection.get(
            where={"manual_name": manual_name},
            limit=10000,
            include=['metadatas']
        )
        
        if not results['ids']:
            return None
        
        # Calculate statistics
        pages = set()
        chunks_per_page = {}
        
        for metadata in results['metadatas']:
            start_page = metadata.get('start_page')
            if start_page:
                pages.add(start_page)
                chunks_per_page[start_page] = chunks_per_page.get(start_page, 0) + 1
        
        return {
            'manual_name': manual_name,
            'total_chunks': len(results['ids']),
            'total_pages': len(pages),
            'pages': sorted(list(pages)),
            'avg_chunks_per_page': sum(chunks_per_page.values()) / len(chunks_per_page) if chunks_per_page else 0
        }
    
    def _format_results(self, results: Dict) -> List[Dict]:
        """Format ChromaDB results for easier use"""
        formatted = []
        
        if not results['ids']:
            return formatted
        
        # ChromaDB returns nested lists
        ids = results['ids'][0]
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        for i in range(len(ids)):
            formatted.append({
                'id': ids[i],
                'content': documents[i],
                'metadata': metadatas[i],
                'distance': distances[i],
                'score': 1.0 - distances[i]  # Convert distance to similarity score
            })
        
        return formatted


# Example usage
if __name__ == "__main__":
    # Initialize connector
    connector = ChromaDBConnector()
    
    print("\n" + "="*60)
    print("ChromaDB Connector Example")
    print("="*60)
    
    # 1. List available manuals
    print("\n1. Available Manuals:")
    manuals = connector.list_manuals()
    for manual in manuals:
        print(f"   - {manual}")
    
    # 2. Basic search
    print("\n2. Basic Search Example:")
    query = "motor configuration"
    results = connector.search(query, n_results=3)
    
    for i, result in enumerate(results):
        print(f"\n   Result {i+1}:")
        print(f"   - Manual: {result['metadata'].get('manual_name')}")
        print(f"   - Page: {result['metadata'].get('start_page')}")
        print(f"   - Score: {result['score']:.3f}")
        print(f"   - Content: {result['content'][:100]}...")
    
    # 3. Filtered search
    if manuals:
        print(f"\n3. Filtered Search (Manual: {manuals[0]}):")
        results = connector.search_with_filter(
            query="installation",
            filters={"manual_name": manuals[0]},
            n_results=2
        )
        
        for i, result in enumerate(results):
            print(f"\n   Result {i+1}:")
            print(f"   - Page: {result['metadata'].get('start_page')}")
            print(f"   - Content: {result['content'][:100]}...")
    
    # 4. Manual statistics
    if manuals:
        print(f"\n4. Statistics for {manuals[0]}:")
        stats = connector.get_manual_stats(manuals[0])
        if stats:
            print(f"   - Total chunks: {stats['total_chunks']}")
            print(f"   - Total pages: {stats['total_pages']}")
            print(f"   - Avg chunks/page: {stats['avg_chunks_per_page']:.1f}")
    
    print("\n" + "="*60)
    print("ChromaDB Structure Summary:")
    print("="*60)
    print("""
    Collections:
    - technical_manuals: Main text chunks from PDFs
    - technical_manuals_images: Image references and metadata
    
    Document Metadata Fields:
    - chunk_id: Unique chunk identifier
    - manual_id: Manual ID from SQLite
    - manual_name: Name of the source manual
    - manufacturer: Equipment manufacturer
    - model: Equipment model
    - document_type: Type of document (technical, etc.)
    - chunk_index: Index of chunk in document
    - start_page, end_page: Page range
    - keywords: Extracted keywords
    - entities: Named entities
    - importance_score: Relevance score
    
    Usage Tips:
    - Use search() for basic semantic search
    - Use search_with_filter() to limit to specific manuals
    - Results include both content and metadata
    - Distance is converted to similarity score (1 - distance)
    """)