"""
Agent Response data model for structured responses.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from services.escalation_service import EscalationContext


@dataclass
class AgentResponse:
    """
    Structured response from the insurance agent.
    
    Contains the response content, metadata about the source,
    confidence scoring, and escalation information.
    """
    
    content: str
    source: str  # "faq", "policy_document", "claim_database", "escalation"
    confidence: float
    should_escalate: bool
    retrieved_chunks: Optional[List[str]] = None
    claim_info: Optional[Dict[str, Any]] = None
    escalation_context: Optional["EscalationContext"] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate the response data after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate the agent response data.
        
        Raises:
            ValueError: If validation fails
        """
        if not self.content or not isinstance(self.content, str):
            raise ValueError("Content must be a non-empty string")
        
        if not self.source or not isinstance(self.source, str):
            raise ValueError("Source must be a non-empty string")
        
        valid_sources = ["faq", "policy_document", "claim_database", "escalation", "general", "system", "error_handler", "fallback"]
        if self.source not in valid_sources:
            raise ValueError(f"Source must be one of: {valid_sources}")
        
        if not isinstance(self.confidence, (int, float)) or not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be a number between 0 and 1")
        
        if not isinstance(self.should_escalate, bool):
            raise ValueError("should_escalate must be a boolean")
        
        if self.retrieved_chunks is not None and not isinstance(self.retrieved_chunks, list):
            raise ValueError("retrieved_chunks must be a list or None")
        
        if self.claim_info is not None and not isinstance(self.claim_info, dict):
            raise ValueError("claim_info must be a dictionary or None")
    
    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """
        Check if the response has high confidence.
        
        Args:
            threshold: Confidence threshold (default: 0.7)
            
        Returns:
            True if confidence is above threshold
        """
        return self.confidence >= threshold
    
    def has_source_attribution(self) -> bool:
        """
        Check if the response has proper source attribution.
        
        Returns:
            True if response includes source information
        """
        return bool(self.retrieved_chunks or self.claim_info)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the response to a dictionary.
        
        Returns:
            Dictionary representation of the response
        """
        result = {
            "content": self.content,
            "source": self.source,
            "confidence": self.confidence,
            "should_escalate": self.should_escalate,
            "retrieved_chunks": self.retrieved_chunks,
            "claim_info": self.claim_info,
            "timestamp": self.timestamp.isoformat()
        }
        
        if self.escalation_context:
            result["escalation_context"] = {
                "trigger": str(self.escalation_context.trigger.value),
                "reason": self.escalation_context.reason,
                "confidence_score": self.escalation_context.confidence_score,
                "session_id": self.escalation_context.session_id,
                "timestamp": self.escalation_context.timestamp.isoformat()
            }
        
        return result