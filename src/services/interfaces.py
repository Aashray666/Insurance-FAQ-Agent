"""
Abstract base classes for all service components.

Defines consistent interfaces for RAG retrieval, FAQ matching,
claim lookup, and PDF processing services.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.chunk import Chunk
from models.faq_response import FAQResponse  
from models.claim_info import ClaimInfo


class RAGRetrieverInterface(ABC):
    """
    Abstract base class for RAG (Retrieval-Augmented Generation) retrieval services.
    
    Handles vector-based retrieval from policy documents using semantic similarity.
    """
    
    @abstractmethod
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG retriever.
        
        Args:
            embedding_model: Name of the sentence transformer model to use
        """
        pass
    
    @abstractmethod
    def load_vector_db(self, policy_type: str) -> None:
        """
        Load the vector database for a specific policy type.
        
        Args:
            policy_type: Type of policy ("Private Car", "Commercial Vehicle", "Two-wheeler")
            
        Raises:
            FileNotFoundError: If vector database file doesn't exist
            ValueError: If policy_type is invalid
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def calculate_similarity_threshold(self) -> float:
        """
        Calculate dynamic similarity threshold based on query results.
        
        Returns:
            Similarity threshold value between 0 and 1
        """
        pass
    
    @abstractmethod
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query string.
        
        Args:
            query: Query string to embed
            
        Returns:
            Query embedding as numpy array
        """
        pass
    
    @abstractmethod
    def is_vector_db_loaded(self, policy_type: str) -> bool:
        """
        Check if vector database is loaded for a policy type.
        
        Args:
            policy_type: Policy type to check
            
        Returns:
            True if vector database is loaded
        """
        pass


class FAQMatcherInterface(ABC):
    """
    Abstract base class for FAQ matching services.
    
    Handles structured FAQ lookup using keyword matching and scoring.
    """
    
    @abstractmethod
    def __init__(self, faq_data_path: str):
        """
        Initialize the FAQ matcher.
        
        Args:
            faq_data_path: Path to the FAQ JSON data file
        """
        pass
    
    @abstractmethod
    def match_faq(self, query: str, policy_type: str) -> Optional[FAQResponse]:
        """
        Find the best matching FAQ for a query.
        
        Args:
            query: User query string
            policy_type: Type of policy to search FAQs for
            
        Returns:
            Best matching FAQ response or None if no good match
        """
        pass
    
    @abstractmethod
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from a query string.
        
        Args:
            query: Query string to process
            
        Returns:
            List of extracted keywords
        """
        pass
    
    @abstractmethod
    def calculate_match_score(self, query_keywords: List[str], faq_keywords: List[str]) -> float:
        """
        Calculate match score between query and FAQ keywords.
        
        Args:
            query_keywords: Keywords from user query
            faq_keywords: Keywords from FAQ entry
            
        Returns:
            Match score between 0 and 1
        """
        pass
    
    @abstractmethod
    def get_faqs_by_category(self, policy_type: str, category: str) -> List[FAQResponse]:
        """
        Get all FAQs for a specific policy type and category.
        
        Args:
            policy_type: Type of policy
            category: FAQ category (e.g., "claims", "coverage")
            
        Returns:
            List of FAQs matching the criteria
        """
        pass
    
    @abstractmethod
    def load_faq_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load FAQ data from the JSON file.
        
        Returns:
            Dictionary of FAQ data organized by policy type
            
        Raises:
            FileNotFoundError: If FAQ data file doesn't exist
            ValueError: If FAQ data format is invalid
        """
        pass


class ClaimServiceInterface(ABC):
    """
    Abstract base class for claim lookup services.
    
    Handles claim status queries and validation against sample data.
    """
    
    @abstractmethod
    def __init__(self, claims_data_path: str):
        """
        Initialize the claim service.
        
        Args:
            claims_data_path: Path to the claims JSON data file
        """
        pass
    
    @abstractmethod
    def lookup_claim(self, claim_number: str) -> Optional[ClaimInfo]:
        """
        Look up claim information by claim number.
        
        Args:
            claim_number: Claim number to search for
            
        Returns:
            Claim information or None if not found
        """
        pass
    
    @abstractmethod
    def validate_claim_format(self, claim_number: str) -> bool:
        """
        Validate claim number format.
        
        Args:
            claim_number: Claim number to validate
            
        Returns:
            True if format is valid (PC/CV/TW + YYYY + NNN)
        """
        pass
    
    @abstractmethod
    def get_claims_by_policy_type(self, policy_type: str) -> List[ClaimInfo]:
        """
        Get all claims for a specific policy type.
        
        Args:
            policy_type: Type of policy
            
        Returns:
            List of claims for the policy type
        """
        pass
    
    @abstractmethod
    def get_claims_by_status(self, status: str) -> List[ClaimInfo]:
        """
        Get all claims with a specific status.
        
        Args:
            status: Claim status to filter by
            
        Returns:
            List of claims with the specified status
        """
        pass
    
    @abstractmethod
    def load_claims_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load claims data from the JSON file.
        
        Returns:
            Dictionary containing claims data
            
        Raises:
            FileNotFoundError: If claims data file doesn't exist
            ValueError: If claims data format is invalid
        """
        pass
    
    @abstractmethod
    def generate_sample_claims(self, count_per_type: int = 10) -> None:
        """
        Generate sample claim data for testing.
        
        Args:
            count_per_type: Number of claims to generate per policy type
        """
        pass


class PDFProcessorInterface(ABC):
    """
    Abstract base class for PDF processing services.
    
    Handles text extraction, chunking, and vector database creation from policy PDFs.
    """
    
    @abstractmethod
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the PDF processor.
        
        Args:
            embedding_model: Name of the sentence transformer model to use
        """
        pass
    
    @abstractmethod
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF cannot be processed
        """
        pass
    
    @abstractmethod
    def create_semantic_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Create semantic chunks from text content.
        
        Args:
            text: Text content to chunk
            chunk_size: Target size for each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        pass
    
    @abstractmethod
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for text chunks.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Array of embeddings for each chunk
        """
        pass
    
    @abstractmethod
    def build_vector_database(self, pdf_path: str, policy_type: str, output_dir: str = "data/vector_dbs") -> None:
        """
        Build a complete vector database from a PDF file.
        
        Args:
            pdf_path: Path to the policy PDF file
            policy_type: Type of policy ("Private Car", "Commercial Vehicle", "Two-wheeler")
            output_dir: Directory to save the vector database
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If policy_type is invalid
        """
        pass
    
    @abstractmethod
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text content
        """
        pass
    
    @abstractmethod
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract different sections from policy text.
        
        Args:
            text: Policy text content
            
        Returns:
            Dictionary mapping section names to content
        """
        pass
    
    @abstractmethod
    def save_vector_database(self, chunks: List[Chunk], policy_type: str, output_dir: str) -> None:
        """
        Save vector database and chunk metadata to files.
        
        Args:
            chunks: List of chunks with embeddings
            policy_type: Type of policy
            output_dir: Directory to save files
        """
        pass
    
    @abstractmethod
    def load_vector_database(self, policy_type: str, data_dir: str = "data/vector_dbs") -> List[Chunk]:
        """
        Load vector database and chunk metadata from files.
        
        Args:
            policy_type: Type of policy to load
            data_dir: Directory containing the vector database files
            
        Returns:
            List of chunks with embeddings
            
        Raises:
            FileNotFoundError: If vector database files don't exist
        """
        pass