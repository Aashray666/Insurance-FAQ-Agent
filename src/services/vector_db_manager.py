"""
Vector Database Manager for Insurance Policy Documents

This module provides a high-level interface for managing and querying
FAISS vector databases for different policy types.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.chunk import Chunk
from services.pdf_processor import PDFProcessor


class VectorDBManager:
    """
    Manages FAISS vector databases for different policy types.
    Provides unified interface for loading, querying, and managing vector databases.
    """
    
    def __init__(self, data_dir: str = "data", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize Vector Database Manager.
        
        Args:
            data_dir: Directory containing vector databases
            embedding_model: Name of the sentence transformer model
        """
        self.data_dir = Path(data_dir)
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.logger = logging.getLogger(__name__)
        
        # Cache for loaded databases
        self._loaded_databases = {}
        self._loaded_metadata = {}
        
        # Supported policy types
        self.supported_policy_types = ["private_car", "commercial_vehicle", "two_wheeler"]
    
    def _load_embedding_model(self) -> None:
        """Lazy load the embedding model"""
        if self.embedding_model is None:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                raise
    
    def is_database_available(self, policy_type: str) -> bool:
        """
        Check if vector database exists for given policy type.
        
        Args:
            policy_type: Policy type to check
            
        Returns:
            True if database exists, False otherwise
        """
        if policy_type not in self.supported_policy_types:
            return False
        
        vector_db_path = self.data_dir / "vector_dbs" / f"{policy_type}.faiss"
        metadata_path = self.data_dir / "embeddings" / f"{policy_type}_chunks.json"
        
        return vector_db_path.exists() and metadata_path.exists()
    
    def load_database(self, policy_type: str) -> bool:
        """
        Load vector database and metadata for given policy type.
        
        Args:
            policy_type: Policy type to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        if policy_type not in self.supported_policy_types:
            self.logger.error(f"Unsupported policy type: {policy_type}")
            return False
        
        if policy_type in self._loaded_databases:
            self.logger.debug(f"Database for {policy_type} already loaded")
            return True
        
        if not self.is_database_available(policy_type):
            self.logger.error(f"Vector database not available for {policy_type}")
            return False
        
        try:
            processor = PDFProcessor(self.embedding_model_name)
            index, metadata = processor.load_vector_database(policy_type, str(self.data_dir))
            
            self._loaded_databases[policy_type] = index
            self._loaded_metadata[policy_type] = metadata
            
            self.logger.info(f"Loaded vector database for {policy_type}: {len(metadata)} chunks")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load database for {policy_type}: {e}")
            return False
    
    def search_similar_chunks(
        self, 
        query: str, 
        policy_type: str, 
        top_k: int = 3,
        similarity_threshold: float = 0.5
    ) -> List[Chunk]:
        """
        Search for similar chunks in the vector database.
        
        Args:
            query: Search query
            policy_type: Policy type to search in
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar chunks
        """
        if not self.load_database(policy_type):
            return []
        
        self._load_embedding_model()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            self.logger.debug(f"Query embedding type: {type(query_embedding)}, shape: {query_embedding.shape}")
            
            # Ensure proper format for FAISS
            if len(query_embedding.shape) == 1:
                query_embedding = query_embedding.reshape(1, -1)
            query_embedding = query_embedding.astype(np.float32)
            self.logger.debug(f"Processed embedding type: {type(query_embedding)}, shape: {query_embedding.shape}")
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            self.logger.debug("Normalization successful")
            
            # Search in vector database
            index = self._loaded_databases[policy_type]
            metadata = self._loaded_metadata[policy_type]
            self.logger.debug(f"Index type: {type(index)}, metadata length: {len(metadata)}")
            
            scores, indices = index.search(query_embedding, top_k)
            self.logger.debug(f"Search successful: scores shape {scores.shape}, indices shape {indices.shape}")
            
            # Convert results to Chunk objects
            chunks = []
            
            # Map internal policy types to display names
            policy_type_mapping = {
                "private_car": "Private Car",
                "commercial_vehicle": "Commercial Vehicle", 
                "two_wheeler": "Two-wheeler"
            }
            
            display_policy_type = policy_type_mapping.get(policy_type, policy_type)
            
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx == -1 or score < similarity_threshold:
                    continue
                
                chunk_data = metadata[idx]
                chunk = Chunk(
                    content=chunk_data["content"],
                    embedding=None,  # Don't store embedding in search results
                    policy_type=display_policy_type,
                    section=chunk_data["section"],
                    similarity_score=float(score)
                )
                chunks.append(chunk)
            
            self.logger.debug(f"Found {len(chunks)} similar chunks for query in {policy_type}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Search failed for {policy_type}: {e}")
            return []
    
    def get_database_stats(self, policy_type: str) -> Optional[Dict]:
        """
        Get statistics about a vector database.
        
        Args:
            policy_type: Policy type to get stats for
            
        Returns:
            Dictionary with database statistics or None if not available
        """
        if not self.load_database(policy_type):
            return None
        
        index = self._loaded_databases[policy_type]
        metadata = self._loaded_metadata[policy_type]
        
        # Calculate section distribution
        section_counts = {}
        total_length = 0
        
        for chunk_data in metadata:
            section = chunk_data["section"]
            section_counts[section] = section_counts.get(section, 0) + 1
            total_length += chunk_data["length"]
        
        return {
            "policy_type": policy_type,
            "total_chunks": len(metadata),
            "embedding_dimension": index.d,
            "total_text_length": total_length,
            "average_chunk_length": total_length / len(metadata) if metadata else 0,
            "section_distribution": section_counts
        }
    
    def get_all_available_databases(self) -> List[str]:
        """
        Get list of all available policy types with vector databases.
        
        Returns:
            List of available policy types
        """
        available = []
        for policy_type in self.supported_policy_types:
            if self.is_database_available(policy_type):
                available.append(policy_type)
        return available
    
    def preload_all_databases(self) -> Dict[str, bool]:
        """
        Preload all available vector databases.
        
        Returns:
            Dictionary mapping policy types to load success status
        """
        results = {}
        available_types = self.get_all_available_databases()
        
        for policy_type in available_types:
            results[policy_type] = self.load_database(policy_type)
        
        return results
    
    def clear_cache(self) -> None:
        """Clear all loaded databases from memory"""
        self._loaded_databases.clear()
        self._loaded_metadata.clear()
        self.logger.info("Cleared vector database cache")
    
    def rebuild_database(self, policy_type: str, pdf_path: str) -> bool:
        """
        Rebuild vector database for a policy type from PDF.
        
        Args:
            policy_type: Policy type to rebuild
            pdf_path: Path to PDF file
            
        Returns:
            True if rebuild successful, False otherwise
        """
        if policy_type not in self.supported_policy_types:
            self.logger.error(f"Unsupported policy type: {policy_type}")
            return False
        
        try:
            # Clear existing cache for this policy type
            if policy_type in self._loaded_databases:
                del self._loaded_databases[policy_type]
                del self._loaded_metadata[policy_type]
            
            # Rebuild database
            processor = PDFProcessor(self.embedding_model_name)
            result = processor.build_vector_database(
                pdf_path=pdf_path,
                policy_type=policy_type,
                output_dir=str(self.data_dir)
            )
            
            if result["success"]:
                self.logger.info(f"Successfully rebuilt database for {policy_type}")
                return True
            else:
                self.logger.error(f"Failed to rebuild database for {policy_type}: {result['error']}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error rebuilding database for {policy_type}: {e}")
            return False