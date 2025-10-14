"""
Response models for the Insurance FAQ Agent
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FormattedResponse:
    """Represents a formatted response from the system"""
    content: str
    source: str
    confidence: float
    
    def __str__(self) -> str:
        return self.content
    
    def __repr__(self) -> str:
        return f"FormattedResponse(content='{self.content[:50]}...', source='{self.source}', confidence={self.confidence})"