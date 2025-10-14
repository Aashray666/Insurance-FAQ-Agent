"""
Query Intent enumeration for classifying user queries.
"""

from enum import Enum


class QueryIntent(Enum):
    """
    Enumeration of possible query intents for the insurance agent.
    
    Used to classify user queries and route them to appropriate handlers.
    """
    
    CLAIM_STATUS = "claim_status"
    FAQ_QUESTION = "faq_question"
    POLICY_QUESTION = "policy_question"
    ESCALATION = "escalation"
    GREETING = "greeting"
    UNKNOWN = "unknown"
    
    def __str__(self) -> str:
        """Return the string representation of the intent."""
        return self.value
    
    @classmethod
    def from_string(cls, intent_str: str) -> "QueryIntent":
        """
        Create QueryIntent from string value.
        
        Args:
            intent_str: String representation of the intent
            
        Returns:
            QueryIntent enum value
            
        Raises:
            ValueError: If intent_str is not a valid intent
        """
        try:
            return cls(intent_str.lower())
        except ValueError:
            return cls.UNKNOWN
    
    def is_actionable(self) -> bool:
        """
        Check if the intent requires specific action.
        
        Returns:
            True if intent requires specific handling, False otherwise
        """
        return self in [
            self.CLAIM_STATUS,
            self.FAQ_QUESTION,
            self.POLICY_QUESTION,
            self.ESCALATION
        ]