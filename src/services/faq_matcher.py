"""
FAQ Matcher Implementation for Insurance FAQ System

This module implements the FAQMatcherInterface to provide keyword-based
FAQ matching and retrieval capabilities.
"""

import json
import logging
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.faq_response import FAQResponse
from services.interfaces import FAQMatcherInterface


class FAQMatcher(FAQMatcherInterface):
    """
    FAQ matcher implementation using keyword-based matching.
    
    Provides fast lookup for common questions with scoring and
    policy-type specific FAQ databases.
    """
    
    def __init__(self, faq_data_path: str = "data/faq_database.json"):
        """
        Initialize the FAQ matcher.
        
        Args:
            faq_data_path: Path to the FAQ JSON data file
        """
        self.faq_data_path = Path(faq_data_path)
        self.faqs: Dict[str, List[FAQResponse]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Load existing FAQ data or create sample data
        if self.faq_data_path.exists():
            self.load_faq_data()
        else:
            self._generate_sample_faqs()
            self._save_faq_data()
        
        self.logger.info(f"Initialized FAQ matcher with {sum(len(faqs) for faqs in self.faqs.values())} FAQs")
    
    def match_faq(self, query: str, policy_type: str) -> Optional[FAQResponse]:
        """
        Find the best matching FAQ for a query.
        
        Args:
            query: User query string
            policy_type: Type of policy to search FAQs for
            
        Returns:
            Best matching FAQ response or None if no good match
        """
        if not query or not query.strip():
            return None
        
        query_keywords = self.extract_keywords(query)
        if not query_keywords:
            return None
        
        best_match = None
        best_score = 0.0
        min_threshold = 0.5  # Balanced threshold for good quality matches
        
        # Map policy types to database keys
        policy_mapping = {
            "Private Car": "private_car",
            "Commercial Vehicle": "commercial_vehicle", 
            "Two-wheeler": "two_wheeler"
        }
        
        # Get the database key for the policy type
        db_policy_key = policy_mapping.get(policy_type, policy_type.lower().replace(" ", "_"))
        
        # Get FAQs for the specific policy type and general FAQs
        policy_faqs = self.faqs.get(db_policy_key, [])
        general_faqs = self.faqs.get("general", [])
        
        all_faqs = policy_faqs + general_faqs
        
        for faq in all_faqs:
            score = self.calculate_match_score(query_keywords, faq.keywords)
            
            # More strict matching - require multiple keyword matches for high scores
            query_words = set(word.lower() for word in query.split() if len(word) > 3)
            faq_question_words = set(word.lower() for word in faq.question.split() if len(word) > 3)
            faq_answer_words = set(word.lower() for word in faq.answer.split() if len(word) > 3)
            
            # Calculate word overlap
            question_overlap = len(query_words.intersection(faq_question_words)) / len(query_words) if query_words else 0
            answer_overlap = len(query_words.intersection(faq_answer_words)) / len(query_words) if query_words else 0
            
            # Boost score based on word overlap (more strict)
            if question_overlap >= 0.5:  # At least 50% of query words in question
                score += 0.4
            elif question_overlap >= 0.3:  # At least 30% of query words in question
                score += 0.2
            
            if answer_overlap >= 0.3:  # At least 30% of query words in answer
                score += 0.1
            
            # Penalty for very different contexts
            if len(query_words) > 3 and question_overlap < 0.3:
                score *= 0.3  # Heavy penalty for poor context match
            
            # Special handling for specific terms that need exact context
            if "unnamed passengers" in query.lower() and "unnamed" not in faq.question.lower():
                score *= 0.1  # Heavy penalty for missing specific context
            
            if score > best_score and score >= min_threshold:
                best_score = score
                best_match = faq
        
        if best_match:
            self.logger.info(f"Found FAQ match with score {best_score:.2f}: {best_match.question[:50]}...")
        else:
            self.logger.info(f"No FAQ match found for query: {query[:50]}...")
        
        return best_match
    
    def get_top_matches(self, query: str, policy_type: str, top_k: int = 3) -> List[tuple[FAQResponse, float]]:
        """
        Get top K matching FAQs for a query with their scores.
        
        Args:
            query: User query string
            policy_type: Type of policy to search FAQs for
            top_k: Number of top matches to return
            
        Returns:
            List of tuples (FAQ, score) sorted by score descending
        """
        if not query or not query.strip():
            return []
        
        query_keywords = self.extract_keywords(query)
        if not query_keywords:
            return []
        
        matches = []
        min_threshold = 0.1  # Lower threshold for multiple results
        
        # Map policy types to database keys
        policy_mapping = {
            "Private Car": "private_car",
            "Commercial Vehicle": "commercial_vehicle", 
            "Two-wheeler": "two_wheeler"
        }
        
        # Get the database key for the policy type
        db_policy_key = policy_mapping.get(policy_type, policy_type.lower().replace(" ", "_"))
        
        # Get FAQs for the specific policy type and general FAQs
        policy_faqs = self.faqs.get(db_policy_key, [])
        general_faqs = self.faqs.get("general", [])
        
        all_faqs = policy_faqs + general_faqs
        
        for faq in all_faqs:
            score = self.calculate_match_score(query_keywords, faq.keywords)
            
            # Boost score for exact question matches
            if any(keyword.lower() in faq.question.lower() for keyword in query_keywords):
                score += 0.3
            
            # Boost score for partial question matches
            query_lower = query.lower()
            question_lower = faq.question.lower()
            if any(word in question_lower for word in query_lower.split() if len(word) > 3):
                score += 0.2
            
            # Boost score for answer content matches
            if any(keyword.lower() in faq.answer.lower() for keyword in query_keywords):
                score += 0.1
            
            if score >= min_threshold:
                matches.append((faq, score))
        
        # Sort by score descending and return top K
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:top_k]
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        Extract keywords from a query string.
        
        Args:
            query: Query string to process
            
        Returns:
            List of extracted keywords
        """
        if not query:
            return []
        
        # Convert to lowercase and remove punctuation
        cleaned_query = re.sub(r'[^\w\s]', ' ', query.lower())
        
        # Split into words
        words = cleaned_query.split()
        
        # Remove common stop words
        stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
            'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after',
            'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'can', 'will', 'should', 'would', 'could'
        }
        
        # Filter out stop words and short words
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords
    
    def calculate_match_score(self, query_keywords: List[str], faq_keywords: List[str]) -> float:
        """
        Calculate match score between query and FAQ keywords.
        
        Args:
            query_keywords: Keywords from user query
            faq_keywords: Keywords from FAQ entry
            
        Returns:
            Match score between 0 and 1
        """
        if not query_keywords or not faq_keywords:
            return 0.0
        
        # Convert to lowercase for case-insensitive matching
        query_set = set(kw.lower() for kw in query_keywords)
        faq_set = set(kw.lower() for kw in faq_keywords)
        
        # Calculate exact matches
        exact_matches = len(query_set.intersection(faq_set))
        
        # Calculate partial matches (substring matching)
        partial_matches = 0
        for query_kw in query_set:
            for faq_kw in faq_set:
                if query_kw in faq_kw or faq_kw in query_kw:
                    partial_matches += 0.5
                    break
        
        # Calculate total possible matches
        total_query_keywords = len(query_set)
        
        if total_query_keywords == 0:
            return 0.0
        
        # Calculate score based on coverage of query keywords
        exact_score = exact_matches / total_query_keywords
        partial_score = partial_matches / total_query_keywords
        
        # Combine scores with weights
        final_score = (exact_score * 0.8) + (partial_score * 0.2)
        
        # Boost score for multiple matches
        if exact_matches > 1:
            final_score += min(exact_matches * 0.1, 0.3)
        
        return min(final_score, 1.0)
    
    def get_faqs_by_category(self, policy_type: str, category: str) -> List[FAQResponse]:
        """
        Get all FAQs for a specific policy type and category.
        
        Args:
            policy_type: Type of policy
            category: FAQ category (e.g., "claims", "coverage")
            
        Returns:
            List of FAQs matching the criteria
        """
        # Map policy types to database keys
        policy_mapping = {
            "Private Car": "private_car",
            "Commercial Vehicle": "commercial_vehicle", 
            "Two-wheeler": "two_wheeler"
        }
        
        # Get the database key for the policy type
        db_policy_key = policy_mapping.get(policy_type, policy_type.lower().replace(" ", "_"))
        
        policy_faqs = self.faqs.get(db_policy_key, [])
        general_faqs = self.faqs.get("general", [])
        
        all_faqs = policy_faqs + general_faqs
        
        return [faq for faq in all_faqs if faq.category == category]
    
    def load_faq_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load FAQ data from the JSON file.
        
        Returns:
            Dictionary of FAQ data organized by policy type
            
        Raises:
            FileNotFoundError: If FAQ data file doesn't exist
            ValueError: If FAQ data format is invalid
        """
        try:
            if not self.faq_data_path.exists():
                raise FileNotFoundError(f"FAQ data file not found: {self.faq_data_path}")
            
            with open(self.faq_data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                raise ValueError("Invalid FAQ data format: must be a dictionary")
            
            self.faqs = {}
            total_loaded = 0
            
            for policy_type, faq_list in data.items():
                if not isinstance(faq_list, list):
                    self.logger.warning(f"Skipping invalid FAQ data for {policy_type}: not a list")
                    continue
                
                self.faqs[policy_type] = []
                
                for faq_dict in faq_list:
                    try:
                        # Map database keys to expected policy types
                        policy_type_mapping = {
                            "private_car": "Private Car",
                            "commercial_vehicle": "Commercial Vehicle",
                            "two_wheeler": "Two-wheeler",
                            "general": "General"
                        }
                        
                        # Ensure the FAQ has the policy_type field set with correct value
                        mapped_policy_type = policy_type_mapping.get(policy_type, "General")
                        faq_dict['policy_type'] = mapped_policy_type
                        
                        # Also update valid categories
                        valid_categories = ["claims", "coverage", "procedures", "general", "documents", "payment", "quick facts", "policy", "cover", "premium"]
                        if faq_dict.get('category') not in valid_categories:
                            # Map common categories
                            category_mapping = {
                                "quick facts": "general",
                                "cover": "coverage",
                                "premium": "payment"
                            }
                            faq_dict['category'] = category_mapping.get(faq_dict.get('category'), "general")
                        
                        faq = FAQResponse(**faq_dict)
                        self.faqs[policy_type].append(faq)
                        total_loaded += 1
                    except Exception as e:
                        self.logger.warning(f"Skipping invalid FAQ: {e}")
                        continue
            
            self.logger.info(f"Loaded {total_loaded} FAQs from {self.faq_data_path}")
            return data
                
        except FileNotFoundError as e:
            self.logger.error(f"FAQ data file not found: {e}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in FAQ data file: {e}")
            raise ValueError(f"Invalid JSON format in FAQ data file: {e}")
        except Exception as e:
            self.logger.error(f"Error loading FAQ data: {e}")
            raise ValueError(f"Error loading FAQ data: {e}")
    
    def _save_faq_data(self) -> None:
        """Save FAQ data to JSON file."""
        # Ensure data directory exists
        self.faq_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert FAQs to serializable format
        data = {}
        for policy_type, faq_list in self.faqs.items():
            data[policy_type] = [faq.to_dict() for faq in faq_list]
        
        with open(self.faq_data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _generate_sample_faqs(self) -> None:
        """Generate sample FAQ data for testing."""
        self.logger.info("Generating sample FAQ data")
        
        # General FAQs
        general_faqs = [
            FAQResponse(
                question="How do I file an insurance claim?",
                answer="To file a claim, call our 24/7 helpline at 1-800-RELIANCE or visit our website. You'll need your policy number, details of the incident, and any relevant documents.",
                keywords=["file", "claim", "submit", "how", "process"],
                policy_type="General",
                category="claims"
            ),
            FAQResponse(
                question="What documents are required for claim filing?",
                answer="Required documents include: policy copy, FIR (if applicable), driving license, RC book, repair estimates, photos of damage, and medical reports (for injury claims).",
                keywords=["documents", "required", "claim", "filing", "papers"],
                policy_type="General",
                category="documents"
            ),
            FAQResponse(
                question="How long does claim processing take?",
                answer="Claim processing typically takes 7-15 working days for simple claims. Complex claims involving investigation may take 30-45 days.",
                keywords=["time", "processing", "claim", "how", "long", "duration"],
                policy_type="General",
                category="claims"
            )
        ]
        
        # Private Car FAQs
        private_car_faqs = [
            FAQResponse(
                question="What is covered under Private Car insurance?",
                answer="Private Car insurance covers damage to your vehicle, third-party liability, theft, fire, natural disasters, and personal accident cover for the owner-driver.",
                keywords=["covered", "coverage", "private", "car", "what"],
                policy_type="Private Car",
                category="coverage"
            ),
            FAQResponse(
                question="What is the deductible for Private Car insurance?",
                answer="The deductible varies based on your policy. Typically ranges from ₹1,000 to ₹5,000 for own damage claims. Third-party claims have no deductible.",
                keywords=["deductible", "private", "car", "amount", "own", "damage"],
                policy_type="Private Car",
                category="coverage"
            )
        ]
        
        # Commercial Vehicle FAQs
        commercial_vehicle_faqs = [
            FAQResponse(
                question="What is covered under Commercial Vehicle insurance?",
                answer="Commercial Vehicle insurance covers vehicle damage, third-party liability, goods in transit, driver and conductor coverage, and legal liability to paid driver.",
                keywords=["covered", "coverage", "commercial", "vehicle", "what"],
                policy_type="Commercial Vehicle",
                category="coverage"
            ),
            FAQResponse(
                question="Is goods in transit covered?",
                answer="Yes, goods in transit are covered up to the sum insured mentioned in your policy. Coverage includes damage due to accident, fire, theft, and natural calamities.",
                keywords=["goods", "transit", "covered", "commercial", "cargo"],
                policy_type="Commercial Vehicle",
                category="coverage"
            )
        ]
        
        # Two-wheeler FAQs
        two_wheeler_faqs = [
            FAQResponse(
                question="What is covered under Two-wheeler insurance?",
                answer="Two-wheeler insurance covers damage to your bike/scooter, third-party liability, theft, fire, natural disasters, and personal accident cover.",
                keywords=["covered", "coverage", "two", "wheeler", "bike", "scooter", "what"],
                policy_type="Two-wheeler",
                category="coverage"
            ),
            FAQResponse(
                question="Is helmet and accessories covered?",
                answer="Yes, helmet and accessories are covered up to ₹5,000 under comprehensive two-wheeler insurance, subject to policy terms and conditions.",
                keywords=["helmet", "accessories", "covered", "two", "wheeler"],
                policy_type="Two-wheeler",
                category="coverage"
            )
        ]
        
        # Store all FAQs
        self.faqs = {
            "General": general_faqs,
            "Private Car": private_car_faqs,
            "Commercial Vehicle": commercial_vehicle_faqs,
            "Two-wheeler": two_wheeler_faqs
        }
        
        self.logger.info(f"Generated {sum(len(faqs) for faqs in self.faqs.values())} sample FAQs")