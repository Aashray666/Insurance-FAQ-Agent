#!/usr/bin/env python3
"""
Script to generate sample claim data for the Insurance FAQ Agent.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services.claim_service import ClaimService


def main():
    """Generate sample claims and display statistics."""
    print("Generating sample insurance claims...")
    
    # Initialize ClaimService (will generate sample data)
    claim_service = ClaimService()
    
    # Display statistics
    stats = claim_service.get_claim_statistics()
    
    print(f"\n✅ Generated {stats['total_claims']} sample claims")
    print(f"💰 Total claim amount: ₹{stats['total_claim_amount']:,.2f}")
    print(f"📊 Average claim amount: ₹{stats['average_claim_amount']:,.2f}")
    
    print("\n📋 Claims by Policy Type:")
    for policy_type, count in stats['policy_type_distribution'].items():
        print(f"  • {policy_type}: {count} claims")
    
    print("\n📈 Claims by Status:")
    for status, count in stats['status_distribution'].items():
        print(f"  • {status}: {count} claims")
    
    # Show some example claims
    print("\n🔍 Sample Claims:")
    all_claims = claim_service.get_all_claims()
    
    # Show one example from each policy type
    policy_types = ["Private Car", "Commercial Vehicle", "Two-wheeler"]
    for policy_type in policy_types:
        policy_claims = claim_service.get_claims_by_policy_type(policy_type)
        if policy_claims:
            claim = policy_claims[0]
            print(f"\n  {policy_type} Example:")
            print(f"    Claim ID: {claim.claim_id}")
            print(f"    Customer: {claim.customer_name}")
            print(f"    Status: {claim.status}")
            print(f"    Amount: ₹{claim.claim_amount:,.2f}")
            print(f"    Description: {claim.description}")
            if claim.next_steps:
                print(f"    Next Steps: {claim.next_steps}")
    
    # Test claim lookup functionality
    print("\n🔎 Testing Claim Lookup:")
    test_claims = ["PC2024001", "CV2024001", "TW2024001", "INVALID123"]
    
    for claim_id in test_claims:
        claim = claim_service.lookup_claim(claim_id)
        if claim:
            print(f"  ✅ {claim_id}: Found - {claim.status}")
        else:
            print(f"  ❌ {claim_id}: Not found")
    
    # Test claim format validation
    print("\n✅ Testing Claim Format Validation:")
    test_formats = ["PC2024001", "CV2024999", "TW2024123", "PC24001", "XY2024001", "PC2024"]
    
    for claim_format in test_formats:
        is_valid = claim_service.validate_claim_format(claim_format)
        status = "✅ Valid" if is_valid else "❌ Invalid"
        print(f"  {claim_format}: {status}")
    
    print(f"\n💾 Sample claims saved to: {claim_service.claims_data_path}")
    print("🎉 Sample claim generation completed successfully!")


if __name__ == "__main__":
    main()