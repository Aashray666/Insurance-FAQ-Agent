"""
Claim Information data model for insurance claims.
"""

from dataclasses import dataclass
from typing import Optional
from datetime import datetime
import re


@dataclass
class ClaimInfo:
    """
    Information about an insurance claim.
    
    Contains all relevant details about a claim including status,
    dates, amounts, and next steps.
    """
    
    claim_id: str
    policy_type: str
    customer_name: str
    claim_date: datetime
    status: str
    estimated_resolution: Optional[datetime]
    claim_amount: float
    description: str
    next_steps: Optional[str] = None
    
    def __post_init__(self):
        """Validate the claim data after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate the claim information.
        
        Raises:
            ValueError: If validation fails
        """
        if not self.claim_id or not isinstance(self.claim_id, str):
            raise ValueError("Claim ID must be a non-empty string")
        
        if not self.is_valid_claim_format(self.claim_id):
            raise ValueError("Claim ID must follow format: PC/CV/TW + YYYY + NNN")
        
        valid_policy_types = ["Private Car", "Commercial Vehicle", "Two-wheeler"]
        if self.policy_type not in valid_policy_types:
            raise ValueError(f"Policy type must be one of: {valid_policy_types}")
        
        if not self.customer_name or not isinstance(self.customer_name, str):
            raise ValueError("Customer name must be a non-empty string")
        
        if not isinstance(self.claim_date, datetime):
            raise ValueError("Claim date must be a datetime object")
        
        if self.estimated_resolution and not isinstance(self.estimated_resolution, datetime):
            raise ValueError("Estimated resolution must be a datetime object or None")
        
        if not isinstance(self.claim_amount, (int, float)) or self.claim_amount < 0:
            raise ValueError("Claim amount must be a non-negative number")
        
        if not self.description or not isinstance(self.description, str):
            raise ValueError("Description must be a non-empty string")
        
        valid_statuses = [
            "Submitted", "Under Review", "Approved", "Processing Payment", 
            "Settled", "Rejected", "Pending Documents", "Investigation"
        ]
        if self.status not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
    
    @staticmethod
    def is_valid_claim_format(claim_id: str) -> bool:
        """
        Validate claim ID format.
        
        Expected format: PC/CV/TW + YYYY + NNN (e.g., PC2024001, CV2024002, TW2024003)
        
        Args:
            claim_id: Claim ID to validate
            
        Returns:
            True if format is valid
        """
        pattern = r'^(PC|CV|TW)\d{4}\d{3}$'
        return bool(re.match(pattern, claim_id))
    
    def get_policy_prefix(self) -> str:
        """
        Get the policy type prefix from claim ID.
        
        Returns:
            Policy prefix (PC, CV, or TW)
        """
        return self.claim_id[:2]
    
    def is_resolved(self) -> bool:
        """
        Check if the claim is resolved.
        
        Returns:
            True if claim is settled or rejected
        """
        return self.status in ["Settled", "Rejected"]
    
    def is_pending_action(self) -> bool:
        """
        Check if the claim requires customer action.
        
        Returns:
            True if claim is pending documents or customer action
        """
        return self.status in ["Pending Documents", "Under Review"]
    
    def days_since_submission(self) -> int:
        """
        Calculate days since claim submission.
        
        Returns:
            Number of days since claim was submitted
        """
        return (datetime.now() - self.claim_date).days
    
    def estimated_days_remaining(self) -> Optional[int]:
        """
        Calculate estimated days until resolution.
        
        Returns:
            Number of days until estimated resolution, or None if no estimate
        """
        if not self.estimated_resolution:
            return None
        
        remaining = (self.estimated_resolution - datetime.now()).days
        return max(0, remaining)
    
    def to_dict(self) -> dict:
        """
        Convert claim info to dictionary.
        
        Returns:
            Dictionary representation of the claim
        """
        return {
            "claim_id": self.claim_id,
            "policy_type": self.policy_type,
            "customer_name": self.customer_name,
            "claim_date": self.claim_date.isoformat(),
            "status": self.status,
            "estimated_resolution": self.estimated_resolution.isoformat() if self.estimated_resolution else None,
            "claim_amount": self.claim_amount,
            "description": self.description,
            "next_steps": self.next_steps,
            "days_since_submission": self.days_since_submission(),
            "estimated_days_remaining": self.estimated_days_remaining()
        }