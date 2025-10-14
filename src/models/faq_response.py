"""
FAQ Response data model for structured FAQ answers.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class FAQResponse:
    """
    Structured FAQ response containing question, answer, and metadata.
    
    Used for storing and retrieving frequently asked questions
    with associated keywords and categorization.
    """
    
    question: str
    answer: str
    keywords: List[str]
    policy_type: str
    category: str  # "claims", "coverage", "procedures", "general"
    
    def __post_init__(self):
        """Validate the FAQ response data after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate the FAQ response data.
        
        Raises:
            ValueError: If validation fails
        """
        if not self.question or not isinstance(self.question, str):
            raise ValueError("Question must be a non-empty string")
        
        if not self.answer or not isinstance(self.answer, str):
            raise ValueError("Answer must be a non-empty string")
        
        if not isinstance(self.keywords, list) or not self.keywords:
            raise ValueError("Keywords must be a non-empty list")
        
        if not all(isinstance(keyword, str) for keyword in self.keywords):
            raise ValueError("All keywords must be strings")
        
        valid_policy_types = ["Private Car", "Commercial Vehicle", "Two-wheeler", "General"]
        if self.policy_type not in valid_policy_types:
            raise ValueError(f"Policy type must be one of: {valid_policy_types}")
        
        valid_categories = ["claims", "coverage", "procedures", "general", "documents", "payment", "quick facts", "policy", "cover", "premium"]
        if self.category not in valid_categories:
            raise ValueError(f"Category must be one of: {valid_categories}")
    
    def matches_keywords(self, query_keywords: List[str]) -> bool:
        """
        Check if the FAQ matches any of the query keywords.
        
        Args:
            query_keywords: List of keywords from user query
            
        Returns:
            True if any keywords match
        """
        if not query_keywords:
            return False
        
        # Convert to lowercase for case-insensitive matching
        faq_keywords_lower = [kw.lower() for kw in self.keywords]
        query_keywords_lower = [kw.lower() for kw in query_keywords]
        
        return any(qkw in faq_keywords_lower for qkw in query_keywords_lower)
    
    def calculate_match_score(self, query_keywords: List[str]) -> float:
        """
        Calculate a match score based on keyword overlap.
        
        Args:
            query_keywords: List of keywords from user query
            
        Returns:
            Match score between 0 and 1
        """
        if not query_keywords or not self.keywords:
            return 0.0
        
        # Convert to lowercase for case-insensitive matching
        faq_keywords_lower = set(kw.lower() for kw in self.keywords)
        query_keywords_lower = set(kw.lower() for kw in query_keywords)
        
        # Calculate Jaccard similarity
        intersection = len(faq_keywords_lower.intersection(query_keywords_lower))
        union = len(faq_keywords_lower.union(query_keywords_lower))
        
        return intersection / union if union > 0 else 0.0
    
    def is_relevant_for_policy(self, policy_type: str) -> bool:
        """
        Check if the FAQ is relevant for the given policy type.
        
        Args:
            policy_type: Policy type to check against
            
        Returns:
            True if FAQ is relevant for the policy type
        """
        return self.policy_type == policy_type or self.policy_type == "General"
    
    def get_formatted_response(self) -> str:
        """
        Get a formatted response string.
        
        Returns:
            Formatted FAQ response
        """
        return f"**Q:** {self.question}\n\n**A:** {self.answer}"
    
    def to_dict(self) -> dict:
        """
        Convert FAQ response to dictionary.
        
        Returns:
            Dictionary representation of the FAQ
        """
        return {
            "question": self.question,
            "answer": self.answer,
            "keywords": self.keywords,
            "policy_type": self.policy_type,
            "category": self.category
        }