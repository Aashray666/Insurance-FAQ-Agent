"""
LangGraph node definitions for the Insurance Agent workflow.

Contains all the individual nodes that make up the agent's decision-making
and response generation process.
"""

import re
from typing import Dict, Any, Optional, List
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.query_intent import QueryIntent
from models.agent_response import AgentResponse
from agent.conversation_state import ConversationState, ConversationContext


class GraphNodes:
    """
    Collection of LangGraph nodes for the Insurance Agent workflow.
    
    Each method represents a node in the graph that processes the conversation
    state and makes decisions about routing and response generation.
    """
    
    def __init__(self):
        """Initialize the graph nodes."""
        pass
    
    def policy_selector_node(self, state: ConversationState) -> ConversationState:
        """
        Node for handling policy type selection.
        
        Determines if user needs to select a policy type or if one can be inferred
        from their query. Routes to policy selection if needed.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated conversation state
        """
        context = ConversationContext(state=state)
        context.increment_step()
        
        current_query = state.get("current_query", "").lower()
        
        # Check if policy type is already selected
        if context.is_policy_selected():
            # Policy already selected, continue to intent classification
            context.set_route_decision("intent_classification", "Policy type already selected")
            return context.state
        
        # Try to infer policy type from query
        policy_keywords = {
            "Private Car": ["car", "auto", "vehicle", "private car", "personal vehicle", "automobile"],
            "Commercial Vehicle": ["commercial", "business", "truck", "lorry", "commercial vehicle", "fleet"],
            "Two-wheeler": ["bike", "motorcycle", "scooter", "two wheeler", "two-wheeler", "motorbike"]
        }
        
        detected_policy = None
        for policy_type, keywords in policy_keywords.items():
            if any(keyword in current_query for keyword in keywords):
                detected_policy = policy_type
                break
        
        if detected_policy:
            # Policy type inferred from query
            context.set_policy_type(detected_policy)
            context.set_route_decision("intent_classification", f"Inferred policy type: {detected_policy}")
            
            # Add system message about policy selection
            context.state["response_content"] = f"I understand you're asking about {detected_policy} insurance. Let me help you with that."
            context.state["response_source"] = "system"
            context.state["response_confidence"] = 0.8
        else:
            # Need explicit policy selection
            context.set_route_decision("policy_selection", "Policy type needs to be selected")
            context.state["response_content"] = """
            Welcome! I'm here to help you with your motor insurance questions. 
            
            Please select which type of insurance policy you'd like to ask about:
            1. Private Car Insurance
            2. Commercial Vehicle Insurance  
            3. Two-wheeler Insurance
            
            You can simply type the number or the policy type name.
            """
            context.state["response_source"] = "system"
            context.state["response_confidence"] = 1.0
        
        return context.state
    
    def intent_classifier_node(self, state: ConversationState) -> ConversationState:
        """
        Node for classifying user query intent.
        
        Analyzes the user's query to determine what type of assistance they need
        and sets the appropriate intent with confidence scoring.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated conversation state with detected intent
        """
        context = ConversationContext(state=state)
        context.increment_step()
        
        current_query = state.get("current_query", "").lower()
        
        # Intent classification patterns
        intent_patterns = {
            QueryIntent.CLAIM_STATUS: [
                r"claim\s*(status|number|id)",
                r"(check|track|status)\s*.*claim",
                r"claim\s*[a-z]{2}\d{7}",  # Claim number pattern
                r"where\s*is\s*my\s*claim",
                r"claim\s*(progress|update)"
            ],
            QueryIntent.FAQ_QUESTION: [
                r"how\s*to\s*(file|submit|make)\s*claim",
                r"what\s*(documents|papers)\s*.*need",
                r"claim\s*(process|procedure|steps)",
                r"how\s*long\s*.*claim",
                r"settlement\s*(process|time)",
                r"required\s*documents",
                r"how\s*do\s*i\s*(file|submit)",
                r"what\s*are\s*the\s*steps"
            ],
            QueryIntent.POLICY_QUESTION: [
                r"what\s*is\s*covered",
                r"coverage\s*(details|information)",
                r"deductible\s*(amount|information)",
                r"exclusions?",
                r"premium\s*(amount|cost)",
                r"policy\s*(terms|conditions|details)",
                r"what\s*does\s*.*policy\s*cover"
            ],
            QueryIntent.ESCALATION: [
                r"speak\s*to\s*(manager|supervisor|human)",
                r"not\s*(satisfied|happy)",
                r"complaint",
                r"escalate",
                r"human\s*(agent|representative)",
                r"customer\s*service",
                r"manager",
                r"supervisor"
            ],
            QueryIntent.GREETING: [
                r"^(hi|hello|hey|good\s*(morning|afternoon|evening))",
                r"^(thanks?|thank\s*you)",
                r"^(bye|goodbye|see\s*you)"
            ]
        }
        
        # Calculate intent scores
        intent_scores = {}
        for intent, patterns in intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, current_query))
                score += matches * 0.3  # Each match adds 0.3 to confidence
            
            # Boost score for exact keyword matches
            if intent == QueryIntent.CLAIM_STATUS and any(word in current_query for word in ["claim", "status", "track"]):
                score += 0.4
            elif intent == QueryIntent.FAQ_QUESTION and any(word in current_query for word in ["how", "what", "process", "documents"]):
                score += 0.3
            elif intent == QueryIntent.POLICY_QUESTION and any(word in current_query for word in ["coverage", "covered", "policy", "deductible"]):
                score += 0.3
            elif intent == QueryIntent.ESCALATION and any(word in current_query for word in ["manager", "complaint", "dissatisfied", "supervisor", "human"]):
                score += 0.5
            
            intent_scores[intent] = min(score, 1.0)  # Cap at 1.0
        
        # Special handling for "how to" questions - prioritize FAQ over claim status
        if re.search(r"how\s*to", current_query.lower()):
            if QueryIntent.FAQ_QUESTION in intent_scores:
                intent_scores[QueryIntent.FAQ_QUESTION] += 0.3  # Boost FAQ score for "how to" questions
        
        # Determine best intent
        if not intent_scores or max(intent_scores.values()) < 0.2:
            detected_intent = QueryIntent.UNKNOWN
            confidence = 0.1
        else:
            detected_intent = max(intent_scores, key=intent_scores.get)
            confidence = intent_scores[detected_intent]
        
        # Special handling for claim number detection
        claim_number_pattern = r"[A-Z]{2}\d{7}"
        if re.search(claim_number_pattern, current_query.upper()):
            detected_intent = QueryIntent.CLAIM_STATUS
            confidence = max(confidence, 0.9)
        
        context.set_intent(detected_intent, confidence)
        
        # Set routing decision based on intent
        if detected_intent == QueryIntent.CLAIM_STATUS:
            context.set_route_decision("claim_lookup", f"Intent: {detected_intent}, Confidence: {confidence:.2f}")
        elif detected_intent == QueryIntent.FAQ_QUESTION:
            context.set_route_decision("faq_search", f"Intent: {detected_intent}, Confidence: {confidence:.2f}")
        elif detected_intent == QueryIntent.POLICY_QUESTION:
            context.set_route_decision("faq_search", f"Intent: {detected_intent}, Confidence: {confidence:.2f}")  # Try FAQ first
        elif detected_intent == QueryIntent.ESCALATION:
            context.set_route_decision("escalation", f"Intent: {detected_intent}, Confidence: {confidence:.2f}")
        elif detected_intent == QueryIntent.GREETING:
            context.set_route_decision("greeting_response", f"Intent: {detected_intent}, Confidence: {confidence:.2f}")
        else:
            # Unknown or low confidence - try FAQ first, then RAG
            context.set_route_decision("faq_search", f"Intent: {detected_intent}, Confidence: {confidence:.2f}")
        
        return context.state
    
    def greeting_response_node(self, state: ConversationState) -> ConversationState:
        """
        Node for handling greeting and general conversation.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated conversation state with greeting response
        """
        context = ConversationContext(state=state)
        context.increment_step()
        
        current_query = state.get("current_query", "").lower()
        
        if any(word in current_query for word in ["hi", "hello", "hey", "good"]):
            response_content = """
            Hello! I'm your Insurance Assistant. I'm here to help you with:
            
            • Checking claim status
            • Answering questions about your motor insurance policy
            • Providing information about claims procedures
            • General insurance inquiries
            
            What can I help you with today?
            """
        elif any(word in current_query for word in ["thanks", "thank"]):
            response_content = "You're welcome! Is there anything else I can help you with regarding your insurance?"
        elif any(word in current_query for word in ["bye", "goodbye"]):
            response_content = "Goodbye! Feel free to return anytime if you have more insurance questions. Have a great day!"
        else:
            response_content = "I'm here to help with your insurance questions. What would you like to know?"
        
        # Create response
        response = AgentResponse(
            content=response_content,
            source="system",
            confidence=1.0,
            should_escalate=False
        )
        
        context.add_agent_response(response)
        context.set_route_decision("complete", "Greeting handled")
        
        return context.state
    
    def escalation_node(self, state: ConversationState) -> ConversationState:
        """
        Node for handling escalation to human agents.
        
        Uses the EscalationService to detect triggers and generate appropriate
        escalation responses with context collection.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated conversation state with escalation response
        """
        context = ConversationContext(state=state)
        context.increment_step()
        
        # Import escalation service (avoid circular imports)
        from services.escalation_service import EscalationService
        
        escalation_service = EscalationService()
        
        # Get current query and conversation data
        current_query = state.get("current_query", "")
        conversation_history = state.get("conversation_history", [])
        response_confidence = state.get("response_confidence", 0.0)
        intent_confidence = state.get("intent_confidence", 0.0)
        
        # Detect escalation triggers
        should_escalate, triggers, reason = escalation_service.detect_escalation_triggers(
            query=current_query,
            confidence=intent_confidence,
            conversation_history=conversation_history,
            response_confidence=response_confidence
        )
        
        # Create escalation context
        escalation_context = escalation_service.create_escalation_context(
            triggers=triggers,
            reason=reason,
            confidence_score=response_confidence,
            conversation_history=conversation_history,
            policy_type=state.get("selected_policy_type"),
            session_id=state.get("session_id", "unknown"),
            error_details=state.get("escalation_reason")
        )
        
        # Generate escalation response
        response = escalation_service.generate_escalation_response(escalation_context)
        
        # Add response to context
        context.add_agent_response(response)
        context.set_route_decision("complete", f"Escalated: {reason}")
        
        return context.state
    
    def response_generator_node(self, state: ConversationState) -> ConversationState:
        """
        Node for generating final formatted responses.
        
        Takes retrieved information and formats it into a user-friendly response
        with proper source attribution.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated conversation state with formatted response
        """
        context = ConversationContext(state=state)
        context.increment_step()
        
        # Check if response is already generated
        if state.get("response_content") and state.get("response_source"):
            context.set_route_decision("complete", "Response already generated")
            return context.state
        
        # Generate response based on available information
        retrieved_chunks = state.get("retrieved_chunks", [])
        faq_matches = state.get("faq_matches", [])
        claim_info = state.get("claim_info")
        
        if claim_info:
            # Format claim information response
            response_content = self._format_claim_response(claim_info)
            source = "claim_database"
            confidence = 0.95
        elif faq_matches:
            # Format FAQ response
            response_content = self._format_faq_response(faq_matches[0])
            source = "faq"
            confidence = 0.9
        elif retrieved_chunks:
            # Format RAG response
            response_content = self._format_rag_response(retrieved_chunks, state.get("current_query", ""))
            source = "policy_document"
            confidence = 0.8
        else:
            # No information found - provide helpful fallback
            response_content = self._format_fallback_response(state.get("current_query", ""))
            source = "general"
            confidence = 0.3
        
        # Create formatted response
        response = AgentResponse(
            content=response_content,
            source=source,
            confidence=confidence,
            should_escalate=confidence < 0.5,
            retrieved_chunks=retrieved_chunks if retrieved_chunks else None,
            claim_info=claim_info
        )
        
        context.add_agent_response(response)
        context.set_route_decision("complete", f"Response generated from {source}")
        
        return context.state
    
    def _format_claim_response(self, claim_info: Dict[str, Any]) -> str:
        """Format claim information into user-friendly response."""
        # Handle date formatting
        claim_date = claim_info.get('claim_date', 'N/A')
        if isinstance(claim_date, str) and claim_date != 'N/A':
            try:
                from datetime import datetime
                parsed_date = datetime.fromisoformat(claim_date.replace('Z', '+00:00'))
                claim_date = parsed_date.strftime("%B %d, %Y")
            except:
                pass  # Keep original string if parsing fails
        
        estimated_resolution = claim_info.get('estimated_resolution')
        if estimated_resolution and isinstance(estimated_resolution, str):
            try:
                from datetime import datetime
                parsed_date = datetime.fromisoformat(estimated_resolution.replace('Z', '+00:00'))
                estimated_resolution = parsed_date.strftime("%B %d, %Y")
            except:
                pass  # Keep original string if parsing fails
        
        response = f"""**Claim Status Information**

**Claim Number:** {claim_info.get('claim_id', 'N/A')}
**Status:** {claim_info.get('status', 'Unknown')}
**Policy Type:** {claim_info.get('policy_type', 'N/A')}
**Claim Date:** {claim_date}
**Claim Amount:** ₹{claim_info.get('claim_amount', 0):,.2f}

**Description:** {claim_info.get('description', 'No description available')}

**Next Steps:** {claim_info.get('next_steps', 'Please contact customer service for more information')}"""

        if estimated_resolution:
            response += f"\n\n**Estimated Resolution:** {estimated_resolution}"
        
        # Add days information if available
        days_since = claim_info.get('days_since_submission')
        days_remaining = claim_info.get('estimated_days_remaining')
        
        if days_since is not None:
            response += f"\n\n*Submitted {days_since} days ago*"
        
        if days_remaining is not None and days_remaining > 0:
            response += f" | *Estimated {days_remaining} days remaining*"
        
        response += "\n\nIf you have any questions about your claim, please don't hesitate to ask!"
        
        return response
    
    def _format_faq_response(self, faq_match: Dict[str, Any]) -> str:
        """Format FAQ response with source attribution."""
        return f"""
        **{faq_match.get('question', 'Frequently Asked Question')}**
        
        {faq_match.get('answer', 'Answer not available')}
        
        *Source: FAQ Database - {faq_match.get('category', 'General').title()} Category*
        
        Is there anything else you'd like to know about this topic?
        """
    
    def _format_rag_response(self, chunks: List[Dict[str, Any]], query: str) -> str:
        """Format RAG response with retrieved information."""
        if not chunks:
            return "I couldn't find specific information about your query in the policy documents."
        
        # Extract content from chunk dictionaries
        chunk_contents = []
        for chunk in chunks[:2]:  # Use top 2 chunks
            content = chunk.get('content', '')
            similarity = chunk.get('similarity_score', 0)
            section = chunk.get('section', 'Unknown Section')
            
            if content:
                chunk_contents.append(f"**From {section}:**\n{content}")
        
        if not chunk_contents:
            return "I found some relevant information but couldn't extract the content properly."
        
        combined_info = "\n\n".join(chunk_contents)
        
        return f"""Based on your policy documents, here's the information I found:

{combined_info}

*Source: Policy Document*

Would you like me to clarify any specific aspect of this information?"""
    
    def _format_fallback_response(self, query: str) -> str:
        """Format fallback response when no specific information is found."""
        return f"""
        I understand you're asking about: "{query}"
        
        I wasn't able to find specific information about this in our knowledge base. However, I can help you with:
        
        • Checking claim status (provide your claim number)
        • General policy coverage questions
        • Claims filing procedures
        • Contact information for human support
        
        Could you please rephrase your question or let me know if you'd like to speak with a human representative?
        """


def create_conditional_routing_logic() -> Dict[str, Any]:
    """
    Create conditional routing logic for LangGraph edges.
    
    Returns:
        Dictionary containing routing conditions and next node mappings
    """
    return {
        "policy_selection_routes": {
            "policy_selection": "policy_selector_node",
            "intent_classification": "intent_classifier_node"
        },
        "intent_classification_routes": {
            "claim_lookup": "claim_lookup_tool",
            "faq_search": "faq_search_tool", 
            "rag_search": "rag_search_tool",
            "escalation": "escalation_node",
            "greeting_response": "greeting_response_node"
        },
        "service_routes": {
            "faq_fallback_to_rag": "rag_search_tool",
            "response_generation": "response_generator_node",
            "complete": "END"
        }
    }