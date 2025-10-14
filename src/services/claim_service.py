"""
Claim Service for handling insurance claim data and operations.
"""

import json
import logging
import random
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.claim_info import ClaimInfo
from services.interfaces import ClaimServiceInterface


class ClaimService(ClaimServiceInterface):
    """
    Service for managing insurance claim data and operations.
    
    Handles claim lookup, validation, and sample data generation
    for demonstration purposes.
    """
    
    def __init__(self, claims_data_path: str = "data/sample_claims.json"):
        """
        Initialize the ClaimService.
        
        Args:
            claims_data_path: Path to the claims data JSON file
        """
        self.claims_data_path = Path(claims_data_path)
        self.claims: Dict[str, ClaimInfo] = {}
        self.logger = logging.getLogger(__name__)
        
        # Load existing claims or generate sample data
        if self.claims_data_path.exists():
            self.load_claims_data()
        else:
            self.generate_sample_claims()
            self.save_claims()
        
        self.logger.info(f"Initialized ClaimService with {len(self.claims)} claims")
    

    
    def _generate_policy_claims(
        self, 
        policy_type: str, 
        prefix: str, 
        count: int,
        base_amounts: tuple,
        descriptions: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Generate claims for a specific policy type.
        
        Args:
            policy_type: Type of insurance policy
            prefix: Claim ID prefix (PC, CV, TW)
            count: Number of claims to generate
            base_amounts: Tuple of (min_amount, max_amount)
            descriptions: List of possible claim descriptions
            
        Returns:
            List of claim dictionaries
        """
        claims = []
        statuses = [
            "Submitted", "Under Review", "Approved", "Processing Payment",
            "Settled", "Rejected", "Pending Documents", "Investigation"
        ]
        
        # Sample customer names
        customer_names = [
            "Rajesh Kumar", "Priya Sharma", "Amit Patel", "Sunita Singh",
            "Vikram Gupta", "Meera Reddy", "Arjun Nair", "Kavya Iyer",
            "Rohit Joshi", "Anita Verma", "Sanjay Agarwal", "Pooja Mehta",
            "Ravi Krishnan", "Deepika Rao", "Manoj Tiwari", "Shreya Bansal",
            "Kiran Malhotra", "Neha Chopra", "Suresh Yadav", "Ritika Saxena"
        ]
        
        for i in range(count):
            claim_number = f"{prefix}2024{str(i+1).zfill(3)}"
            
            # Random claim date within last 6 months
            days_ago = random.randint(1, 180)
            claim_date = datetime.now() - timedelta(days=days_ago)
            
            # Random status
            status = random.choice(statuses)
            
            # Estimated resolution based on status
            estimated_resolution = None
            next_steps = None
            
            if status in ["Submitted", "Under Review", "Pending Documents"]:
                # Future resolution date
                days_ahead = random.randint(5, 30)
                estimated_resolution = datetime.now() + timedelta(days=days_ahead)
                
                if status == "Submitted":
                    next_steps = "Initial review in progress"
                elif status == "Under Review":
                    next_steps = "Awaiting adjuster assessment"
                elif status == "Pending Documents":
                    next_steps = "Please submit required documents"
                    
            elif status == "Approved":
                days_ahead = random.randint(3, 10)
                estimated_resolution = datetime.now() + timedelta(days=days_ahead)
                next_steps = "Payment processing initiated"
                
            elif status == "Processing Payment":
                days_ahead = random.randint(1, 5)
                estimated_resolution = datetime.now() + timedelta(days=days_ahead)
                next_steps = "Payment will be credited within 3-5 business days"
                
            elif status == "Investigation":
                days_ahead = random.randint(15, 45)
                estimated_resolution = datetime.now() + timedelta(days=days_ahead)
                next_steps = "Investigation in progress, may require additional information"
                
            elif status == "Settled":
                next_steps = "Claim settled successfully"
                
            elif status == "Rejected":
                next_steps = "Claim rejected - please contact customer service for details"
            
            # Random claim amount within range
            min_amount, max_amount = base_amounts
            claim_amount = round(random.uniform(min_amount, max_amount), 2)
            
            # Random description
            description = random.choice(descriptions)
            
            # Random customer name
            customer_name = random.choice(customer_names)
            
            claim_data = {
                "claim_id": claim_number,
                "policy_type": policy_type,
                "customer_name": customer_name,
                "claim_date": claim_date,
                "status": status,
                "estimated_resolution": estimated_resolution,
                "claim_amount": claim_amount,
                "description": description,
                "next_steps": next_steps
            }
            
            claims.append(claim_data)
        
        return claims
    
    def lookup_claim(self, claim_number: str) -> Optional[ClaimInfo]:
        """
        Look up a claim by claim number.
        
        Args:
            claim_number: The claim number to search for
            
        Returns:
            ClaimInfo object if found, None otherwise
        """
        if not claim_number or not isinstance(claim_number, str):
            self.logger.warning("Invalid claim number provided for lookup")
            return None
        
        # Normalize claim number to uppercase
        normalized_claim_number = claim_number.strip().upper()
        
        # Validate format before lookup
        if not self.validate_claim_format(normalized_claim_number):
            self.logger.warning(f"Invalid claim number format: {claim_number}")
            return None
        
        claim = self.claims.get(normalized_claim_number)
        if claim:
            self.logger.info(f"Found claim: {normalized_claim_number}")
        else:
            self.logger.info(f"Claim not found: {normalized_claim_number}")
        
        return claim
    
    def validate_claim_format(self, claim_number: str) -> bool:
        """
        Validate claim number format.
        
        Args:
            claim_number: Claim number to validate
            
        Returns:
            True if format is valid (PC/CV/TW + YYYY + NNN)
        """
        if not claim_number or not isinstance(claim_number, str):
            return False
        
        return ClaimInfo.is_valid_claim_format(claim_number.strip().upper())
    
    def get_claims_by_policy_type(self, policy_type: str) -> List[ClaimInfo]:
        """
        Get all claims for a specific policy type.
        
        Args:
            policy_type: Policy type to filter by
            
        Returns:
            List of ClaimInfo objects for the policy type
        """
        if not policy_type:
            return []
        
        valid_policy_types = ["Private Car", "Commercial Vehicle", "Two-wheeler"]
        if policy_type not in valid_policy_types:
            self.logger.warning(f"Invalid policy type: {policy_type}")
            return []
        
        claims = [
            claim for claim in self.claims.values() 
            if claim.policy_type == policy_type
        ]
        
        self.logger.info(f"Found {len(claims)} claims for policy type: {policy_type}")
        return claims
    
    def get_claims_by_status(self, status: str) -> List[ClaimInfo]:
        """
        Get all claims with a specific status.
        
        Args:
            status: Status to filter by
            
        Returns:
            List of ClaimInfo objects with the status
        """
        if not status:
            return []
        
        valid_statuses = [
            "Submitted", "Under Review", "Approved", "Processing Payment", 
            "Settled", "Rejected", "Pending Documents", "Investigation"
        ]
        
        if status not in valid_statuses:
            self.logger.warning(f"Invalid status: {status}")
            return []
        
        claims = [
            claim for claim in self.claims.values()
            if claim.status == status
        ]
        
        self.logger.info(f"Found {len(claims)} claims with status: {status}")
        return claims
    
    def get_all_claims(self) -> List[ClaimInfo]:
        """
        Get all claims.
        
        Returns:
            List of all ClaimInfo objects
        """
        return list(self.claims.values())
    
    def save_claims(self) -> None:
        """
        Save claims data to JSON file.
        """
        # Ensure data directory exists
        self.claims_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert claims to serializable format
        claims_data = {
            "claims": [claim.to_dict() for claim in self.claims.values()]
        }
        
        with open(self.claims_data_path, 'w', encoding='utf-8') as f:
            json.dump(claims_data, f, indent=2, ensure_ascii=False)
    
    def load_claims_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load claims data from the JSON file.
        
        Returns:
            Dictionary containing claims data
            
        Raises:
            FileNotFoundError: If claims data file doesn't exist
            ValueError: If claims data format is invalid
        """
        try:
            if not self.claims_data_path.exists():
                raise FileNotFoundError(f"Claims data file not found: {self.claims_data_path}")
            
            with open(self.claims_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict) or "claims" not in data:
                raise ValueError("Invalid claims data format: missing 'claims' key")
            
            self.claims = {}
            claims_loaded = 0
            
            for claim_dict in data.get("claims", []):
                try:
                    # Convert date strings back to datetime objects
                    claim_dict["claim_date"] = datetime.fromisoformat(claim_dict["claim_date"])
                    if claim_dict.get("estimated_resolution"):
                        claim_dict["estimated_resolution"] = datetime.fromisoformat(
                            claim_dict["estimated_resolution"]
                        )
                    
                    # Remove computed fields that shouldn't be in constructor
                    claim_dict.pop("days_since_submission", None)
                    claim_dict.pop("estimated_days_remaining", None)
                    
                    claim = ClaimInfo(**claim_dict)
                    self.claims[claim.claim_id] = claim
                    claims_loaded += 1
                    
                except Exception as e:
                    self.logger.warning(f"Skipping invalid claim data: {e}")
                    continue
            
            self.logger.info(f"Loaded {claims_loaded} claims from {self.claims_data_path}")
            return data
                
        except FileNotFoundError as e:
            self.logger.error(f"Claims data file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in claims data file: {e}")
            raise ValueError(f"Invalid JSON format in claims data file: {e}")
        except Exception as e:
            self.logger.error(f"Error loading claims data: {e}")
            raise ValueError(f"Error loading claims data: {e}")
    
    def add_claim(self, claim: ClaimInfo) -> None:
        """
        Add a new claim to the service.
        
        Args:
            claim: ClaimInfo object to add
        """
        self.claims[claim.claim_id] = claim
        self.save_claims()
    
    def update_claim_status(self, claim_id: str, new_status: str, next_steps: Optional[str] = None) -> bool:
        """
        Update the status of an existing claim.
        
        Args:
            claim_id: ID of the claim to update
            new_status: New status for the claim
            next_steps: Optional next steps information
            
        Returns:
            True if update was successful, False if claim not found
        """
        if claim_id in self.claims:
            claim = self.claims[claim_id]
            claim.status = new_status
            if next_steps:
                claim.next_steps = next_steps
            self.save_claims()
            return True
        return False
    
    def get_claim_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all claims.
        
        Returns:
            Dictionary with claim statistics
        """
        if not self.claims:
            return {}
        
        total_claims = len(self.claims)
        
        # Count by status
        status_counts = {}
        for claim in self.claims.values():
            status_counts[claim.status] = status_counts.get(claim.status, 0) + 1
        
        # Count by policy type
        policy_counts = {}
        for claim in self.claims.values():
            policy_counts[claim.policy_type] = policy_counts.get(claim.policy_type, 0) + 1
        
        # Calculate total claim amounts
        total_amount = sum(claim.claim_amount for claim in self.claims.values())
        avg_amount = total_amount / total_claims if total_claims > 0 else 0
        
        return {
            "total_claims": total_claims,
            "status_distribution": status_counts,
            "policy_type_distribution": policy_counts,
            "total_claim_amount": total_amount,
            "average_claim_amount": round(avg_amount, 2)
        }
    
    def generate_sample_claims(self, count_per_type: int = 10) -> None:
        """
        Generate sample claim data for testing.
        
        Args:
            count_per_type: Number of claims to generate per policy type
        """
        self.logger.info(f"Generating {count_per_type} sample claims per policy type")
        
        sample_claims = []
        
        # Private Car claims (PC2024XXX)
        pc_claims = self._generate_policy_claims(
            policy_type="Private Car",
            prefix="PC",
            count=15 if count_per_type == 10 else count_per_type,
            base_amounts=(5000, 150000),
            descriptions=[
                "Accident damage to front bumper and headlight",
                "Rear-end collision damage to trunk and bumper", 
                "Side mirror damage from parking incident",
                "Windshield crack from road debris",
                "Theft of vehicle from parking lot",
                "Fire damage to engine compartment",
                "Flood damage to interior and electronics",
                "Vandalism damage to paint and windows",
                "Hit and run damage to driver side door",
                "Collision with animal on highway"
            ]
        )
        sample_claims.extend(pc_claims)
        
        # Commercial Vehicle claims (CV2024XXX)
        cv_claims = self._generate_policy_claims(
            policy_type="Commercial Vehicle",
            prefix="CV", 
            count=12 if count_per_type == 10 else count_per_type,
            base_amounts=(10000, 500000),
            descriptions=[
                "Cargo damage during transportation",
                "Truck collision with bridge overpass",
                "Delivery van theft from depot",
                "Loading dock accident damage",
                "Multi-vehicle highway accident",
                "Mechanical breakdown on delivery route",
                "Fire damage to commercial fleet vehicle",
                "Vandalism to company vehicle overnight",
                "Collision during goods delivery",
                "Weather-related damage to transport vehicle"
            ]
        )
        sample_claims.extend(cv_claims)
        
        # Two-wheeler claims (TW2024XXX)
        tw_claims = self._generate_policy_claims(
            policy_type="Two-wheeler",
            prefix="TW",
            count=18 if count_per_type == 10 else count_per_type,
            base_amounts=(2000, 80000),
            descriptions=[
                "Motorcycle accident at intersection",
                "Scooter theft from residential area",
                "Collision with car while lane changing",
                "Fall due to wet road conditions",
                "Vandalism damage to parked bike",
                "Fire damage to motorcycle engine",
                "Hit by vehicle while parked",
                "Accident during overtaking maneuver",
                "Collision with stray animal",
                "Damage from falling tree branch"
            ]
        )
        sample_claims.extend(tw_claims)
        
        # Convert to ClaimInfo objects and store
        self.claims = {}
        for claim_data in sample_claims:
            claim = ClaimInfo(**claim_data)
            self.claims[claim.claim_id] = claim
        
        self.logger.info(f"Generated {len(self.claims)} sample claims")
    
    def format_claim_status(self, claim: ClaimInfo) -> Dict[str, Any]:
        """
        Format claim status information for display.
        
        Args:
            claim: ClaimInfo object to format
            
        Returns:
            Dictionary with formatted claim status information
        """
        formatted = {
            "claim_number": claim.claim_id,
            "policy_type": claim.policy_type,
            "customer_name": claim.customer_name,
            "status": claim.status,
            "claim_date": claim.claim_date.strftime("%B %d, %Y"),
            "claim_amount": f"₹{claim.claim_amount:,.2f}",
            "description": claim.description,
            "days_since_submission": claim.days_since_submission(),
            "next_steps": claim.next_steps or "No additional steps required"
        }
        
        # Add estimated resolution if available
        if claim.estimated_resolution:
            formatted["estimated_resolution"] = claim.estimated_resolution.strftime("%B %d, %Y")
            formatted["estimated_days_remaining"] = claim.estimated_days_remaining()
        else:
            formatted["estimated_resolution"] = "Not available"
            formatted["estimated_days_remaining"] = None
        
        # Add status-specific information
        if claim.status == "Settled":
            formatted["status_message"] = "Your claim has been successfully settled."
        elif claim.status == "Rejected":
            formatted["status_message"] = "Your claim has been rejected. Please contact customer service for details."
        elif claim.status == "Processing Payment":
            formatted["status_message"] = "Payment is being processed and will be credited soon."
        elif claim.status == "Pending Documents":
            formatted["status_message"] = "Please submit the required documents to proceed."
        elif claim.status == "Under Review":
            formatted["status_message"] = "Your claim is currently under review by our team."
        elif claim.status == "Investigation":
            formatted["status_message"] = "Your claim is under investigation. We may contact you for additional information."
        else:
            formatted["status_message"] = "Your claim is being processed."
        
        return formatted
    
    def search_claims(self, query: str) -> List[ClaimInfo]:
        """
        Search claims by various criteria.
        
        Args:
            query: Search query (can be claim number, customer name, or description)
            
        Returns:
            List of matching claims
        """
        if not query or not query.strip():
            return []
        
        query = query.strip().lower()
        matching_claims = []
        
        for claim in self.claims.values():
            # Check claim number (case insensitive)
            if query.upper() in claim.claim_id.upper():
                matching_claims.append(claim)
                continue
            
            # Check customer name (case insensitive)
            if query in claim.customer_name.lower():
                matching_claims.append(claim)
                continue
            
            # Check description (case insensitive)
            if query in claim.description.lower():
                matching_claims.append(claim)
                continue
        
        self.logger.info(f"Found {len(matching_claims)} claims matching query: {query}")
        return matching_claims
    
    def get_recent_claims(self, days: int = 30) -> List[ClaimInfo]:
        """
        Get claims submitted in the last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of recent claims
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_claims = [
            claim for claim in self.claims.values()
            if claim.claim_date >= cutoff_date
        ]
        
        # Sort by claim date (most recent first)
        recent_claims.sort(key=lambda x: x.claim_date, reverse=True)
        
        self.logger.info(f"Found {len(recent_claims)} claims from last {days} days")
        return recent_claims