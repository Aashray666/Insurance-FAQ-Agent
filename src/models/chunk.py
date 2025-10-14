"""
Chunk data model for vector database storage and retrieval.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Chunk:
    """
    Text chunk with embedding for vector database storage.
    
    Contains the text content, its vector embedding, and metadata
    for retrieval and similarity scoring.
    """
    
    content: str
    embedding: Optional[np.ndarray]
    policy_type: str
    section: str
    similarity_score: float = 0.0
    
    def __post_init__(self):
        """Validate the chunk data after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate the chunk data.
        
        Raises:
            ValueError: If validation fails
        """
        if not self.content or not isinstance(self.content, str):
            raise ValueError("Content must be a non-empty string")
        
        if self.embedding is not None:
            if not isinstance(self.embedding, np.ndarray):
                raise ValueError("Embedding must be a numpy array")
            
            if self.embedding.size == 0:
                raise ValueError("Embedding cannot be empty")
        
        valid_policy_types = ["Private Car", "Commercial Vehicle", "Two-wheeler"]
        if self.policy_type not in valid_policy_types:
            raise ValueError(f"Policy type must be one of: {valid_policy_types}")
        
        if not self.section or not isinstance(self.section, str):
            raise ValueError("Section must be a non-empty string")
        
        if not isinstance(self.similarity_score, (int, float)) or not 0 <= self.similarity_score <= 1:
            raise ValueError("Similarity score must be a number between 0 and 1")
    
    def calculate_similarity(self, query_embedding: np.ndarray) -> float:
        """
        Calculate cosine similarity with a query embedding.
        
        Args:
            query_embedding: Query embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if self.embedding is None:
            raise ValueError("Cannot calculate similarity: chunk embedding is None")
            
        if not isinstance(query_embedding, np.ndarray):
            raise ValueError("Query embedding must be a numpy array")
        
        if query_embedding.size == 0:
            raise ValueError("Query embedding cannot be empty")
        
        # Normalize vectors
        chunk_norm = np.linalg.norm(self.embedding)
        query_norm = np.linalg.norm(query_embedding)
        
        if chunk_norm == 0 or query_norm == 0:
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(self.embedding, query_embedding) / (chunk_norm * query_norm)
        
        # Ensure similarity is between 0 and 1
        similarity = max(0.0, min(1.0, (similarity + 1) / 2))
        
        self.similarity_score = similarity
        return similarity
    
    def is_relevant(self, threshold: float = 0.5) -> bool:
        """
        Check if the chunk is relevant based on similarity threshold.
        
        Args:
            threshold: Minimum similarity threshold (default: 0.5)
            
        Returns:
            True if similarity score is above threshold
        """
        return self.similarity_score >= threshold
    
    def get_preview(self, max_length: int = 150) -> str:
        """
        Get a preview of the chunk content.
        
        Args:
            max_length: Maximum length of preview (default: 150)
            
        Returns:
            Truncated content with ellipsis if needed
        """
        if len(self.content) <= max_length:
            return self.content
        
        return self.content[:max_length].rsplit(' ', 1)[0] + "..."
    
    def get_word_count(self) -> int:
        """
        Get the word count of the chunk content.
        
        Returns:
            Number of words in the content
        """
        return len(self.content.split())
    
    def to_dict(self, include_embedding: bool = False) -> dict:
        """
        Convert chunk to dictionary.
        
        Args:
            include_embedding: Whether to include the embedding array
            
        Returns:
            Dictionary representation of the chunk
        """
        result = {
            "content": self.content,
            "policy_type": self.policy_type,
            "section": self.section,
            "similarity_score": self.similarity_score,
            "word_count": self.get_word_count(),
            "preview": self.get_preview()
        }
        
        if include_embedding:
            result["embedding"] = self.embedding.tolist()
        
        return result