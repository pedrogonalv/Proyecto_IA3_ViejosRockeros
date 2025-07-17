"""
ChromaDB Agent Connector for Technical Documentation RAG System

This connector provides a clean interface for agents to interact with the ChromaDB
vector database containing technical manual embeddings.
"""

import chromadb
from chromadb import PersistentClient
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Structured search result from ChromaDB"""
    chunk_id: str
    content: str
    manual_name: str
    manufacturer: str
    model: str
    page_start: int
    page_end: int
    score: float
    metadata: Dict[str, Any]


class ChromaDBAgentConnector:
    """
    Connector for agents to interact with the ChromaDB vector database.
    
    Database Structure:
    - Instance Name: chroma.sqlite3
    - Collections:
        1. technical_manuals - Text chunks with embeddings (dimension: 384)
        2. technical_manuals_images - Image references and metadata
    
    Metadata Fields:
    - chunk_id: Unique identifier
    - manual_id: Reference to SQLite manual table
    - manual_name: Source manual name
    - manufacturer: Equipment manufacturer
    - model: Equipment model
    - document_type: Type of document
    - chunk_index: Position in document
    - start_page, end_page: Page range
    - keywords: Extracted keywords
    - entities: Named entities
    - importance_score: Relevance score
    """
    
    def __init__(self, db_path: str = "/Users/santiagojorda/Downloads/clode_technical_rag_system/data/vectordb"):
        """
        Initialize connection to ChromaDB.
        
        Args:
            db_path: Path to ChromaDB storage directory
        """
        self.db_path = Path(db_path)
        self.client = None
        self.text_collection = None
        self.image_collection = None
        self._connect()
    
    def _connect(self):
        """Establish connection to ChromaDB"""
        try:
            self.client = PersistentClient(path=str(self.db_path))
            
            # Get collections
            self.text_collection = self.client.get_collection("technical_manuals")
            
            # Image collection might not exist in all setups
            try:
                self.image_collection = self.client.get_collection("technical_manuals_images")
            except Exception:
                logger.warning("Image collection not found")
                self.image_collection = None
                
            logger.info(f"Connected to ChromaDB at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    
    def search(self, 
               query: str, 
               n_results: int = 5,
               filter_dict: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Search for relevant chunks using semantic similarity.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            filter_dict: Optional metadata filters (e.g., {"manufacturer": "Siemens"})
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Build where clause for filtering
            where_clause = filter_dict if filter_dict else None
            
            # Perform search
            results = self.text_collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause,
                include=["metadatas", "documents", "distances"]
            )
            
            # Convert to SearchResult objects
            search_results = []
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                
                result = SearchResult(
                    chunk_id=results['ids'][0][i],
                    content=results['documents'][0][i],
                    manual_name=metadata.get('manual_name', 'Unknown'),
                    manufacturer=metadata.get('manufacturer', 'Unknown'),
                    model=metadata.get('model', 'Unknown'),
                    page_start=metadata.get('start_page', 0),
                    page_end=metadata.get('end_page', 0),
                    score=1 - results['distances'][0][i],  # Convert distance to similarity
                    metadata=metadata
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_manual_chunks(self, manual_name: str, limit: int = 100) -> List[SearchResult]:
        """
        Retrieve all chunks from a specific manual.
        
        Args:
            manual_name: Name of the manual
            limit: Maximum number of chunks to return
            
        Returns:
            List of SearchResult objects
        """
        return self.search(
            query="",  # Empty query
            n_results=limit,
            filter_dict={"manual_name": manual_name}
        )
    
    def get_context_window(self, 
                          chunk_id: str, 
                          window_size: int = 2) -> List[SearchResult]:
        """
        Get surrounding chunks for context.
        
        Args:
            chunk_id: ID of the central chunk
            window_size: Number of chunks before/after to retrieve
            
        Returns:
            List of SearchResult objects in order
        """
        try:
            # Get the target chunk
            target = self.text_collection.get(ids=[chunk_id], include=["metadatas"])
            if not target['ids']:
                return []
            
            target_metadata = target['metadatas'][0]
            manual_name = target_metadata.get('manual_name')
            chunk_index = target_metadata.get('chunk_index', 0)
            
            # Get surrounding chunks
            filter_dict = {
                "$and": [
                    {"manual_name": manual_name},
                    {"chunk_index": {"$gte": chunk_index - window_size}},
                    {"chunk_index": {"$lte": chunk_index + window_size}}
                ]
            }
            
            return self.search("", n_results=window_size * 2 + 1, filter_dict=filter_dict)
            
        except Exception as e:
            logger.error(f"Failed to get context window: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            stats = {
                "text_collection": {
                    "name": "technical_manuals",
                    "count": self.text_collection.count() if self.text_collection else 0
                }
            }
            
            if self.image_collection:
                stats["image_collection"] = {
                    "name": "technical_manuals_images",
                    "count": self.image_collection.count()
                }
            
            # Get unique manuals
            if self.text_collection:
                all_metadata = self.text_collection.get(include=["metadatas"])["metadatas"]
                unique_manuals = set(m.get("manual_name", "Unknown") for m in all_metadata)
                stats["unique_manuals"] = list(unique_manuals)
                stats["total_unique_manuals"] = len(unique_manuals)
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def search_by_page_range(self, 
                            manual_name: str, 
                            start_page: int, 
                            end_page: int) -> List[SearchResult]:
        """
        Search for chunks within a specific page range of a manual.
        
        Args:
            manual_name: Name of the manual
            start_page: Starting page number
            end_page: Ending page number
            
        Returns:
            List of SearchResult objects
        """
        filter_dict = {
            "$and": [
                {"manual_name": manual_name},
                {"start_page": {"$gte": start_page}},
                {"end_page": {"$lte": end_page}}
            ]
        }
        
        return self.search("", n_results=50, filter_dict=filter_dict)
    
    def find_similar_chunks(self, chunk_id: str, n_results: int = 5) -> List[SearchResult]:
        """
        Find chunks similar to a given chunk.
        
        Args:
            chunk_id: ID of the reference chunk
            n_results: Number of similar chunks to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Get the reference chunk
            ref_chunk = self.text_collection.get(
                ids=[chunk_id], 
                include=["documents", "embeddings"]
            )
            
            if not ref_chunk['ids']:
                return []
            
            # Use the embedding for similarity search
            if ref_chunk.get('embeddings'):
                results = self.text_collection.query(
                    query_embeddings=[ref_chunk['embeddings'][0]],
                    n_results=n_results + 1,  # +1 because it will include itself
                    include=["metadatas", "documents", "distances"]
                )
                
                # Convert and filter out the original chunk
                search_results = []
                for i in range(len(results['ids'][0])):
                    if results['ids'][0][i] != chunk_id:
                        metadata = results['metadatas'][0][i]
                        result = SearchResult(
                            chunk_id=results['ids'][0][i],
                            content=results['documents'][0][i],
                            manual_name=metadata.get('manual_name', 'Unknown'),
                            manufacturer=metadata.get('manufacturer', 'Unknown'),
                            model=metadata.get('model', 'Unknown'),
                            page_start=metadata.get('start_page', 0),
                            page_end=metadata.get('end_page', 0),
                            score=1 - results['distances'][0][i],
                            metadata=metadata
                        )
                        search_results.append(result)
                
                return search_results[:n_results]
            
            # Fallback to text-based search
            return self.search(ref_chunk['documents'][0], n_results=n_results)
            
        except Exception as e:
            logger.error(f"Failed to find similar chunks: {e}")
            return []
    
    def close(self):
        """Close the database connection"""
        # ChromaDB persistent client doesn't need explicit closing
        logger.info("ChromaDB connector closed")


# Example usage for agents
if __name__ == "__main__":
    # Initialize connector
    connector = ChromaDBAgentConnector()
    
    # Example: Search for information about sensors
    results = connector.search("temperature sensor calibration", n_results=3)
    for result in results:
        print(f"\nManual: {result.manual_name}")
        print(f"Pages: {result.page_start}-{result.page_end}")
        print(f"Score: {result.score:.3f}")
        print(f"Content: {result.content[:200]}...")
    
    # Example: Get statistics
    stats = connector.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"Total chunks: {stats['text_collection']['count']}")
    print(f"Unique manuals: {stats['total_unique_manuals']}")
    
    # Close connection
    connector.close()