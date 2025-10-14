"""
RAG Retriever Implementation for Insurance Policy Documents

This module implements the RAGRetrieverInterface to provide semantic similarity
search capabilities using FAISS vector databases and sentence transformers.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.chunk import Chunk
from services.interfaces import RAGRetrieverInterface
from services.vector_db_manager import VectorDBManager


class RAGRetriever(RAGRetrieverInterface):
    """
    RAG retriever implementation using FAISS vector databases.
    
    Provides semantic similarity search with configurable top-k results,
    similarity threshold filtering, and fallback logic.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", data_dir: str = "data"):
        """
        Initialize the RAG retriever.
        
        Args:
            embedding_model: Name of the sentence transformer model to use
            data_dir: Directory containing vector databases
        """
        self.embedding_model_name = embedding_model
        self.data_dir = data_dir
        self.embedding_model = None
        self.vector_db_manager = VectorDBManager(data_dir, embedding_model)
        self.logger = logging.getLogger(__name__)
        
        # Policy type mapping for internal consistency
        self.policy_type_mapping = {
            "Private Car": "private_car",
            "Commercial Vehicle": "commercial_vehicle", 
            "Two-wheeler": "two_wheeler"
        }
        
        # Reverse mapping for display
        self.display_mapping = {v: k for k, v in self.policy_type_mapping.items()}
        
        # Default similarity threshold
        self.default_threshold = 0.5
        
        self.logger.info(f"Initialized RAG retriever with model: {embedding_model}")
    
    def _load_embedding_model(self) -> None:
        """Lazy load the embedding model"""
        if self.embedding_model is None:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                raise
    
    def _normalize_policy_type(self, policy_type: str) -> str:
        """
        Normalize policy type to internal format.
        
        Args:
            policy_type: Policy type in any format
            
        Returns:
            Normalized policy type for internal use
            
        Raises:
            ValueError: If policy type is invalid
        """
        if policy_type in self.policy_type_mapping:
            return self.policy_type_mapping[policy_type]
        elif policy_type in self.display_mapping:
            return policy_type
        else:
            valid_types = list(self.policy_type_mapping.keys())
            raise ValueError(f"Invalid policy type: {policy_type}. Valid types: {valid_types}")
    
    def load_vector_db(self, policy_type: str) -> None:
        """
        Load the vector database for a specific policy type.
        
        Args:
            policy_type: Type of policy ("Private Car", "Commercial Vehicle", "Two-wheeler")
            
        Raises:
            FileNotFoundError: If vector database file doesn't exist
            ValueError: If policy_type is invalid
        """
        try:
            normalized_type = self._normalize_policy_type(policy_type)
            
            if not self.vector_db_manager.is_database_available(normalized_type):
                raise FileNotFoundError(f"Vector database not found for policy type: {policy_type}")
            
            success = self.vector_db_manager.load_database(normalized_type)
            if not success:
                raise RuntimeError(f"Failed to load vector database for policy type: {policy_type}")
            
            self.logger.info(f"Successfully loaded vector database for {policy_type}")
            
        except ValueError as e:
            self.logger.error(f"Invalid policy type: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading vector database for {policy_type}: {e}")
            raise
    
    def retrieve_chunks(self, query: str, policy_type: str, top_k: int = 3) -> List[Chunk]:
        """
        Retrieve relevant text chunks for a query.
        
        Args:
            query: User query string
            policy_type: Type of policy to search in
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of relevant chunks with similarity scores
            
        Raises:
            ValueError: If policy_type is invalid or vector DB not loaded
        """
        if not query or not query.strip():
            self.logger.warning("Empty query provided")
            return []
        
        try:
            normalized_type = self._normalize_policy_type(policy_type)
            
            # Ensure vector database is loaded
            if not self.is_vector_db_loaded(policy_type):
                self.load_vector_db(policy_type)
            
            # Calculate dynamic similarity threshold
            threshold = self.calculate_similarity_threshold()
            
            # Search for similar chunks
            chunks = self.vector_db_manager.search_similar_chunks(
                query=query.strip(),
                policy_type=normalized_type,
                top_k=top_k,
                similarity_threshold=threshold
            )
            
            self.logger.info(f"Retrieved {len(chunks)} chunks for query in {policy_type}")
            
            # Log chunk details for debugging
            for i, chunk in enumerate(chunks):
                self.logger.debug(f"Chunk {i+1}: score={chunk.similarity_score:.3f}, "
                                f"section={chunk.section}, preview={chunk.get_preview(50)}")
            
            return chunks
            
        except ValueError as e:
            self.logger.error(f"Invalid input: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error retrieving chunks: {e}")
            return []
    
    def calculate_similarity_threshold(self) -> float:
        """
        Calculate dynamic similarity threshold based on query results.
        
        For now, returns a static threshold. Can be enhanced to be dynamic
        based on query complexity, result distribution, etc.
        
        Returns:
            Similarity threshold value between 0 and 1
        """
        # Static threshold for now - can be made dynamic later
        # based on factors like:
        # - Query length and complexity
        # - Historical performance
        # - Result score distribution
        return self.default_threshold
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query string.
        
        Args:
            query: Query string to embed
            
        Returns:
            Query embedding as numpy array
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        self._load_embedding_model()
        
        try:
            embedding = self.embedding_model.encode([query.strip()], convert_to_numpy=True)
            
            # Ensure proper format
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)
            
            self.logger.debug(f"Generated embedding for query: shape={embedding.shape}")
            return embedding[0]  # Return single embedding vector
            
        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            raise
    
    def is_vector_db_loaded(self, policy_type: str) -> bool:
        """
        Check if vector database is loaded for a policy type.
        
        Args:
            policy_type: Policy type to check
            
        Returns:
            True if vector database is loaded
        """
        try:
            normalized_type = self._normalize_policy_type(policy_type)
            return normalized_type in self.vector_db_manager._loaded_databases
        except ValueError:
            return False
    
    def get_available_policy_types(self) -> List[str]:
        """
        Get list of available policy types with vector databases.
        
        Returns:
            List of available policy types in display format
        """
        available_internal = self.vector_db_manager.get_all_available_databases()
        return [self.display_mapping[internal_type] for internal_type in available_internal]
    
    def preload_all_databases(self) -> Dict[str, bool]:
        """
        Preload all available vector databases.
        
        Returns:
            Dictionary mapping display policy types to load success status
        """
        internal_results = self.vector_db_manager.preload_all_databases()
        
        # Convert to display format
        display_results = {}
        for internal_type, success in internal_results.items():
            display_type = self.display_mapping[internal_type]
            display_results[display_type] = success
        
        return display_results
    
    def get_database_stats(self, policy_type: str) -> Optional[Dict]:
        """
        Get statistics about a vector database.
        
        Args:
            policy_type: Policy type to get stats for
            
        Returns:
            Dictionary with database statistics or None if not available
        """
        try:
            normalized_type = self._normalize_policy_type(policy_type)
            stats = self.vector_db_manager.get_database_stats(normalized_type)
            
            if stats:
                # Convert policy type back to display format
                stats["policy_type"] = policy_type
            
            return stats
            
        except ValueError:
            return None
    
    def clear_cache(self) -> None:
        """Clear all loaded databases from memory"""
        self.vector_db_manager.clear_cache()
        self.logger.info("Cleared RAG retriever cache")
    
    def set_similarity_threshold(self, threshold: float) -> None:
        """
        Set the default similarity threshold.
        
        Args:
            threshold: Similarity threshold between 0 and 1
            
        Raises:
            ValueError: If threshold is not between 0 and 1
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Similarity threshold must be between 0 and 1")
        
        self.default_threshold = threshold
        self.logger.info(f"Set similarity threshold to {threshold}")
    
    def search_with_fallback(self, query: str, policy_type: str, top_k: int = 3) -> Dict:
        """
        Search with fallback logic and detailed results.
        
        Args:
            query: User query string
            policy_type: Type of policy to search in
            top_k: Number of top chunks to retrieve
            
        Returns:
            Dictionary with chunks, metadata, and fallback information
        """
        result = {
            "chunks": [],
            "total_found": 0,
            "used_fallback": False,
            "threshold_used": self.default_threshold,
            "policy_type": policy_type,
            "query": query
        }
        
        try:
            # Primary search
            chunks = self.retrieve_chunks(query, policy_type, top_k)
            result["chunks"] = chunks
            result["total_found"] = len(chunks)
            
            # If no results found, try with lower threshold
            if not chunks:
                lower_threshold = max(0.3, self.default_threshold - 0.2)
                self.logger.info(f"No results found, trying with lower threshold: {lower_threshold}")
                
                # Temporarily lower threshold
                original_threshold = self.default_threshold
                self.set_similarity_threshold(lower_threshold)
                
                try:
                    chunks = self.retrieve_chunks(query, policy_type, top_k)
                    result["chunks"] = chunks
                    result["total_found"] = len(chunks)
                    result["used_fallback"] = True
                    result["threshold_used"] = lower_threshold
                finally:
                    # Restore original threshold
                    self.set_similarity_threshold(original_threshold)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in search with fallback: {e}")
            result["error"] = str(e)
            return result