"""
PDF Processing Service for Insurance Policy Documents

This module handles PDF text extraction, cleaning, and preprocessing
for the Insurance FAQ Agent system.
"""

import re
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

try:
    import PyPDF2
    PDF_LIBRARY = "PyPDF2"
except ImportError:
    try:
        import pdfplumber
        PDF_LIBRARY = "pdfplumber"
    except ImportError:
        raise ImportError("Neither PyPDF2 nor pdfplumber is installed. Please install one of them.")

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.chunk import Chunk


@dataclass
class ProcessingResult:
    """Result of PDF processing operation"""
    success: bool
    text: Optional[str] = None
    error_message: Optional[str] = None
    page_count: Optional[int] = None


class PDFProcessor:
    """
    Handles PDF text extraction, cleaning, and vector database creation
    for insurance policy documents.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize PDF processor with embedding model.
        
        Args:
            embedding_model: Name of the sentence transformer model to use
        """
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.logger = logging.getLogger(__name__)
        
        # Text cleaning patterns
        self.cleaning_patterns = [
            (r'\s+', ' '),  # Multiple whitespace to single space
            (r'\n\s*\n', '\n\n'),  # Multiple newlines to double newline
            (r'[^\w\s\.\,\;\:\!\?\-\(\)\[\]\"\'\/\%\$\&\@\#]', ''),  # Remove special chars except common punctuation
            (r'^\s+|\s+$', ''),  # Strip leading/trailing whitespace
        ]
    
    def _load_embedding_model(self) -> None:
        """Lazy load the embedding model to save memory"""
        if self.embedding_model is None:
            try:
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
                self.logger.info(f"Loaded embedding model: {self.embedding_model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load embedding model: {e}")
                raise
    
    def extract_text_from_pdf(self, pdf_path: str) -> ProcessingResult:
        """
        Extract text from PDF file with error handling.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ProcessingResult with extracted text or error information
        """
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            return ProcessingResult(
                success=False,
                error_message=f"PDF file not found: {pdf_path}"
            )
        
        if not pdf_file.suffix.lower() == '.pdf':
            return ProcessingResult(
                success=False,
                error_message=f"File is not a PDF: {pdf_path}"
            )
        
        try:
            if PDF_LIBRARY == "PyPDF2":
                return self._extract_with_pypdf2(pdf_file)
            else:
                return self._extract_with_pdfplumber(pdf_file)
                
        except Exception as e:
            self.logger.error(f"Unexpected error processing PDF {pdf_path}: {e}")
            return ProcessingResult(
                success=False,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def _extract_with_pypdf2(self, pdf_file: Path) -> ProcessingResult:
        """Extract text using PyPDF2 library"""
        try:
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    return ProcessingResult(
                        success=False,
                        error_message="PDF is encrypted and cannot be processed"
                    )
                
                text_content = []
                page_count = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text_content.append(page_text)
                    except Exception as e:
                        self.logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
                
                if not text_content:
                    return ProcessingResult(
                        success=False,
                        error_message="No text could be extracted from PDF"
                    )
                
                full_text = '\n'.join(text_content)
                cleaned_text = self._clean_text(full_text)
                
                return ProcessingResult(
                    success=True,
                    text=cleaned_text,
                    page_count=page_count
                )
                
        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=f"PyPDF2 extraction failed: {str(e)}"
            )
    
    def _extract_with_pdfplumber(self, pdf_file: Path) -> ProcessingResult:
        """Extract text using pdfplumber library"""
        try:
            import pdfplumber
            
            with pdfplumber.open(pdf_file) as pdf:
                text_content = []
                page_count = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_content.append(page_text)
                    except Exception as e:
                        self.logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
                
                if not text_content:
                    return ProcessingResult(
                        success=False,
                        error_message="No text could be extracted from PDF"
                    )
                
                full_text = '\n'.join(text_content)
                cleaned_text = self._clean_text(full_text)
                
                return ProcessingResult(
                    success=True,
                    text=cleaned_text,
                    page_count=page_count
                )
                
        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=f"pdfplumber extraction failed: {str(e)}"
            )
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        cleaned = text
        
        # Apply cleaning patterns
        for pattern, replacement in self.cleaning_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Remove excessive whitespace
        lines = cleaned.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                cleaned_lines.append(line)
        
        # Join with single newlines and ensure proper spacing
        result = '\n'.join(cleaned_lines)
        
        # Final cleanup
        result = re.sub(r'\n{3,}', '\n\n', result)  # Max 2 consecutive newlines
        result = result.strip()
        
        return result
    
    def validate_extracted_text(self, text: str, min_length: int = 100) -> bool:
        """
        Validate that extracted text meets minimum quality requirements.
        
        Args:
            text: Extracted text to validate
            min_length: Minimum text length required
            
        Returns:
            True if text is valid, False otherwise
        """
        if not text or len(text.strip()) < min_length:
            return False
        
        # Check for reasonable word count
        words = text.split()
        if len(words) < 20:
            return False
        
        # Check for reasonable character distribution
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars / len(text) < 0.3:  # At least 30% alphabetic characters
            return False
        
        return True
    
    def create_semantic_chunks(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Create semantic chunks from text with configurable size and overlap.
        
        Args:
            text: Input text to chunk
            chunk_size: Target size for each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        # Split text into sentences for better semantic boundaries
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_size + sentence_length > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap from previous chunk
                if overlap > 0 and len(current_chunk) > overlap:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + " " + sentence
                    current_size = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_size = sentence_length
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_size = len(current_chunk)
        
        # Add the last chunk if it has content
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very small chunks
        filtered_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 50]
        
        return filtered_chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using simple heuristics.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting - can be improved with NLTK if needed
        sentence_endings = r'[.!?]+\s+'
        sentences = re.split(sentence_endings, text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for text chunks using sentence transformers.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Numpy array of embeddings
        """
        if not chunks:
            return np.array([])
        
        self._load_embedding_model()
        
        try:
            # Generate embeddings in batches for memory efficiency
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                all_embeddings.append(batch_embeddings)
            
            # Concatenate all embeddings
            embeddings = np.vstack(all_embeddings)
            
            self.logger.info(f"Generated embeddings for {len(chunks)} chunks, shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def build_vector_database(self, pdf_path: str, policy_type: str, output_dir: str = "data") -> Dict[str, Any]:
        """
        Build FAISS vector database from PDF document.
        
        Args:
            pdf_path: Path to PDF file
            policy_type: Type of policy (private_car, commercial_vehicle, two_wheeler)
            output_dir: Directory to save vector database and metadata
            
        Returns:
            Dictionary with processing results and metadata
        """
        # Create output directories
        vector_db_dir = Path(output_dir) / "vector_dbs"
        embeddings_dir = Path(output_dir) / "embeddings"
        vector_db_dir.mkdir(parents=True, exist_ok=True)
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract text from PDF
        self.logger.info(f"Processing PDF: {pdf_path}")
        extraction_result = self.extract_text_from_pdf(pdf_path)
        
        if not extraction_result.success:
            return {
                "success": False,
                "error": extraction_result.error_message,
                "policy_type": policy_type
            }
        
        # Validate extracted text
        if not self.validate_extracted_text(extraction_result.text):
            return {
                "success": False,
                "error": "Extracted text failed validation (too short or poor quality)",
                "policy_type": policy_type
            }
        
        # Create semantic chunks
        self.logger.info("Creating semantic chunks...")
        chunks = self.create_semantic_chunks(extraction_result.text)
        
        if not chunks:
            return {
                "success": False,
                "error": "No valid chunks could be created from the text",
                "policy_type": policy_type
            }
        
        # Generate embeddings
        self.logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.generate_embeddings(chunks)
        
        if embeddings.size == 0:
            return {
                "success": False,
                "error": "Failed to generate embeddings",
                "policy_type": policy_type
            }
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        
        # Save FAISS index
        index_path = vector_db_dir / f"{policy_type}.faiss"
        faiss.write_index(index, str(index_path))
        
        # Create chunk metadata
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            chunk_data = {
                "id": i,
                "content": chunk,
                "policy_type": policy_type,
                "section": self._identify_section(chunk),
                "length": len(chunk)
            }
            chunk_metadata.append(chunk_data)
        
        # Save chunk metadata
        metadata_path = embeddings_dir / f"{policy_type}_chunks.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_metadata, f, indent=2, ensure_ascii=False)
        
        result = {
            "success": True,
            "policy_type": policy_type,
            "pdf_path": pdf_path,
            "chunks_count": len(chunks),
            "embedding_dimension": dimension,
            "index_path": str(index_path),
            "metadata_path": str(metadata_path),
            "page_count": extraction_result.page_count
        }
        
        self.logger.info(f"Successfully built vector database for {policy_type}: {len(chunks)} chunks")
        return result
    
    def _identify_section(self, chunk: str) -> str:
        """
        Identify the section type of a text chunk based on content.
        
        Args:
            chunk: Text chunk to analyze
            
        Returns:
            Section identifier
        """
        chunk_lower = chunk.lower()
        
        # Define section keywords
        section_keywords = {
            "coverage": ["coverage", "covered", "benefits", "protection", "insured"],
            "exclusions": ["exclusion", "excluded", "not covered", "except", "limitation"],
            "claims": ["claim", "settlement", "procedure", "process", "filing"],
            "definitions": ["definition", "means", "shall mean", "defined as"],
            "conditions": ["condition", "terms", "requirements", "obligations"],
            "premium": ["premium", "payment", "cost", "fee", "charge"]
        }
        
        # Count keyword matches for each section
        section_scores = {}
        for section, keywords in section_keywords.items():
            score = sum(1 for keyword in keywords if keyword in chunk_lower)
            if score > 0:
                section_scores[section] = score
        
        # Return section with highest score, or "general" if no matches
        if section_scores:
            return max(section_scores, key=section_scores.get)
        else:
            return "general"
    
    def load_vector_database(self, policy_type: str, data_dir: str = "data") -> tuple[faiss.Index, List[Dict]]:
        """
        Load existing vector database and metadata.
        
        Args:
            policy_type: Type of policy to load
            data_dir: Directory containing vector databases
            
        Returns:
            Tuple of (FAISS index, chunk metadata list)
        """
        vector_db_dir = Path(data_dir) / "vector_dbs"
        embeddings_dir = Path(data_dir) / "embeddings"
        
        index_path = vector_db_dir / f"{policy_type}.faiss"
        metadata_path = embeddings_dir / f"{policy_type}_chunks.json"
        
        if not index_path.exists():
            raise FileNotFoundError(f"Vector database not found: {index_path}")
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Chunk metadata not found: {metadata_path}")
        
        # Load FAISS index
        index = faiss.read_index(str(index_path))
        
        # Load chunk metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            chunk_metadata = json.load(f)
        
        self.logger.info(f"Loaded vector database for {policy_type}: {len(chunk_metadata)} chunks")
        return index, chunk_metadata