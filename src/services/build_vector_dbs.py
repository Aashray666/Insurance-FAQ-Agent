"""
Script to build vector databases for all policy types from PDF documents.

This script processes the three Reliance policy PDF documents and creates
separate FAISS vector databases for each policy type.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from services.pdf_processor import PDFProcessor


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('vector_db_build.log')
        ]
    )


def main():
    """Main function to build all vector databases"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Define policy documents mapping
    policy_documents = {
        "private_car": "Reliance_Private_Car_Package_Policy_wording.pdf",
        "commercial_vehicle": "Reliance_Commercial_Vehicles_Package_Policy_wording.pdf", 
        "two_wheeler": "Reliance_Two_wheeler_Package_Policy_wording.pdf"
    }
    
    # Initialize PDF processor
    processor = PDFProcessor(embedding_model="all-MiniLM-L6-v2")
    
    results = {}
    
    logger.info("Starting vector database creation for all policy types...")
    
    for policy_type, pdf_filename in policy_documents.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {policy_type.upper()} policy")
        logger.info(f"{'='*50}")
        
        # Check if PDF file exists
        pdf_path = Path(pdf_filename)
        if not pdf_path.exists():
            logger.error(f"PDF file not found: {pdf_filename}")
            results[policy_type] = {
                "success": False,
                "error": f"File not found: {pdf_filename}"
            }
            continue
        
        try:
            # Build vector database
            result = processor.build_vector_database(
                pdf_path=str(pdf_path),
                policy_type=policy_type,
                output_dir="data"
            )
            
            results[policy_type] = result
            
            if result["success"]:
                logger.info(f"✅ Successfully processed {policy_type}")
                logger.info(f"   - Chunks created: {result['chunks_count']}")
                logger.info(f"   - Embedding dimension: {result['embedding_dimension']}")
                logger.info(f"   - Pages processed: {result['page_count']}")
                logger.info(f"   - Index saved to: {result['index_path']}")
                logger.info(f"   - Metadata saved to: {result['metadata_path']}")
            else:
                logger.error(f"❌ Failed to process {policy_type}: {result['error']}")
                
        except Exception as e:
            logger.error(f"❌ Unexpected error processing {policy_type}: {e}")
            results[policy_type] = {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    # Print summary
    logger.info(f"\n{'='*50}")
    logger.info("PROCESSING SUMMARY")
    logger.info(f"{'='*50}")
    
    successful = 0
    failed = 0
    
    for policy_type, result in results.items():
        if result["success"]:
            successful += 1
            logger.info(f"✅ {policy_type}: SUCCESS ({result['chunks_count']} chunks)")
        else:
            failed += 1
            logger.error(f"❌ {policy_type}: FAILED - {result['error']}")
    
    logger.info(f"\nTotal: {successful} successful, {failed} failed")
    
    if successful == len(policy_documents):
        logger.info("🎉 All vector databases created successfully!")
        return True
    else:
        logger.warning("⚠️  Some vector databases failed to create")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)