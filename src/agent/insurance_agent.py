"""
Main Insurance Agent using LangGraph for workflow orchestration.

Coordinates the entire conversation flow from policy selection through
response generation using a state-based graph approach.
"""

from typing import Dict, Any, Optional, List
import uuid
import logging
from datetime import datetime

from langgraph.graph import StateGraph, END

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.agent_response import AgentResponse
from agent.conversation_state import ConversationState, ConversationContext, create_initial_state
from agent.graph_nodes import GraphNodes, create_conditional_routing_logic


class InsuranceAgent:
    """
    Main Insurance Agent orchestrator using LangGraph.
    
    Manages the complete conversation workflow from policy selection
    through response generation using a state-based graph approach.
    """
    
    def __init__(self):
        """Initialize the Insurance Agent with LangGraph workflow."""
        self.nodes = GraphNodes()
        self.graph = None
        self.current_session: Optional[ConversationContext] = None
        self.routing_logic = create_conditional_routing_logic()
        self.logger = logging.getLogger(__name__)
        
        # Initialize services (will be injected later)
        self.claim_service = None
        self.faq_matcher = None
        self.rag_retriever = None
        
        self._build_graph()
    
    def _build_graph(self) -> None:
        """Build the LangGraph workflow."""
        # Create the state graph
        workflow = StateGraph(ConversationState)
        
        # Add nodes
        workflow.add_node("policy_selector", self.nodes.policy_selector_node)
        workflow.add_node("intent_classifier", self.nodes.intent_classifier_node)
        workflow.add_node("greeting_handler", self.nodes.greeting_response_node)
        workflow.add_node("escalation_handler", self.nodes.escalation_node)
        workflow.add_node("response_generator", self.nodes.response_generator_node)
        
        # Add tool nodes (will be implemented in next subtask)
        workflow.add_node("claim_lookup_tool", self._claim_lookup_tool)
        workflow.add_node("faq_search_tool", self._faq_search_tool)
        workflow.add_node("rag_search_tool", self._rag_search_tool)
        
        # Set entry point
        workflow.set_entry_point("policy_selector")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "policy_selector",
            self._route_from_policy_selector,
            {
                "policy_selection": "policy_selector",
                "intent_classification": "intent_classifier"
            }
        )
        
        workflow.add_conditional_edges(
            "intent_classifier", 
            self._route_from_intent_classifier,
            {
                "claim_lookup": "claim_lookup_tool",
                "faq_search": "faq_search_tool",
                "rag_search": "rag_search_tool", 
                "escalation": "escalation_handler",
                "greeting_response": "greeting_handler"
            }
        )
        
        # Add edges from tool nodes to response generator or escalation
        workflow.add_conditional_edges(
            "claim_lookup_tool",
            self._route_from_tools,
            {
                "response_generation": "response_generator",
                "escalation": "escalation_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "faq_search_tool",
            self._route_from_tools,
            {
                "response_generation": "response_generator",
                "rag_fallback": "rag_search_tool",
                "escalation": "escalation_handler"
            }
        )
        
        workflow.add_conditional_edges(
            "rag_search_tool",
            self._route_from_tools,
            {
                "response_generation": "response_generator",
                "escalation": "escalation_handler"
            }
        )
        
        # Add edges to END
        workflow.add_edge("response_generator", END)
        workflow.add_edge("escalation_handler", END)
        workflow.add_edge("greeting_handler", END)
        
        # Compile the graph
        self.graph = workflow.compile()
    
    def _route_from_policy_selector(self, state: ConversationState) -> str:
        """
        Route from policy selector based on state.
        
        Args:
            state: Current conversation state
            
        Returns:
            Next node name
        """
        route_decision = state.get("route_decision")
        
        if route_decision == "policy_selection":
            return "policy_selection"
        elif route_decision == "intent_classification":
            return "intent_classification"
        else:
            # Default to intent classification if policy is selected
            return "intent_classification"
    
    def _route_from_intent_classifier(self, state: ConversationState) -> str:
        """
        Route from intent classifier based on detected intent.
        
        Args:
            state: Current conversation state
            
        Returns:
            Next node name
        """
        route_decision = state.get("route_decision")
        
        if route_decision == "claim_lookup":
            return "claim_lookup"
        elif route_decision == "faq_search":
            return "faq_search"
        elif route_decision == "rag_search":
            return "rag_search"
        elif route_decision == "escalation":
            return "escalation"
        elif route_decision == "greeting_response":
            return "greeting_response"
        else:
            # Default fallback to FAQ search
            return "faq_search"
    
    def _route_from_tools(self, state: ConversationState) -> str:
        """
        Route from tool nodes based on results and confidence.
        
        Args:
            state: Current conversation state
            
        Returns:
            Next node name
        """
        # Check if escalation is needed
        context = ConversationContext(state=state)
        if context.should_escalate_conversation():
            return "escalation"
        
        # Check if we have results to generate response
        has_claim_info = bool(state.get("claim_info"))
        has_faq_matches = bool(state.get("faq_matches")) and len(state.get("faq_matches", [])) > 0
        has_retrieved_chunks = bool(state.get("retrieved_chunks")) and len(state.get("retrieved_chunks", [])) > 0
        
        # Get current route decision to understand which tool we came from
        current_route = state.get("route_decision")
        
        # Conditional routing logic:
        # 1. If FAQ search didn't find results, try RAG fallback
        if (current_route == "faq_search" and not has_faq_matches and not has_retrieved_chunks):
            return "rag_fallback"
        
        # 2. If we have any results, generate response
        if has_claim_info or has_faq_matches or has_retrieved_chunks:
            return "response_generation"
        
        # 3. Check confidence level for escalation
        confidence = state.get("response_confidence", 0.0)
        if confidence < 0.3:
            return "escalation"
        
        # 4. No results found, escalate
        return "escalation"
    
    # Tool nodes - integrated retrieval services
    def _claim_lookup_tool(self, state: ConversationState) -> ConversationState:
        """
        Tool node for claim lookup service with comprehensive error handling.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated conversation state
        """
        context = ConversationContext(state=state)
        context.increment_step()
        
        current_query = state.get("current_query", "")
        
        try:
            # Extract claim number from query
            import re
            claim_pattern = r"[A-Z]{2}\d{7}"
            claim_match = re.search(claim_pattern, current_query.upper())
            
            if claim_match and self.claim_service:
                claim_number = claim_match.group()
                
                # Look up claim using the claim service
                claim_info = self.claim_service.lookup_claim(claim_number)
                
                if claim_info:
                    # Convert ClaimInfo to dictionary for state storage
                    context.state["claim_info"] = claim_info.to_dict()
                    context.state["response_confidence"] = 0.95
                    self.logger.info(f"Successfully found claim: {claim_number}")
                else:
                    # Claim not found - use graceful degradation
                    from services.error_handler import graceful_degradation
                    
                    fallback_response = graceful_degradation.execute_fallback(
                        service_name="claim_service",
                        original_query=current_query,
                        claim_number=claim_number
                    )
                    
                    if fallback_response:
                        context.state["response_content"] = fallback_response.content
                        context.state["response_source"] = fallback_response.source
                        context.state["response_confidence"] = fallback_response.confidence
                        context.state["should_escalate"] = fallback_response.should_escalate
                    else:
                        context.state["escalation_reason"] = f"Claim number {claim_number} not found in our records"
                        context.state["response_confidence"] = 0.1
                    
                    self.logger.info(f"Claim not found: {claim_number}")
            else:
                # No valid claim number found or service not available
                if not claim_match:
                    context.state["escalation_reason"] = "No valid claim number found in query"
                else:
                    context.state["escalation_reason"] = "Claim service not available"
                context.state["response_confidence"] = 0.1
                
        except Exception as e:
            # Use comprehensive error handling
            from services.error_handler import error_handler
            
            error_response, should_escalate = error_handler.handle_error(
                exception=e,
                component="claim_lookup_tool",
                user_query=current_query,
                context={"claim_service_available": self.claim_service is not None}
            )
            
            context.state["response_content"] = error_response.content
            context.state["response_source"] = error_response.source
            context.state["response_confidence"] = error_response.confidence
            context.state["should_escalate"] = should_escalate
            context.state["escalation_reason"] = f"Claim lookup error: {str(e)}"
            
            self.logger.error(f"Error in claim lookup: {e}")
        
        return context.state
    
    def _faq_search_tool(self, state: ConversationState) -> ConversationState:
        """
        Tool node for FAQ search service with comprehensive error handling.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated conversation state
        """
        context = ConversationContext(state=state)
        context.increment_step()
        
        current_query = state.get("current_query", "")
        policy_type = state.get("selected_policy_type", "General")
        
        try:
            if self.faq_matcher and current_query:
                # Search for matching FAQ
                faq_match = self.faq_matcher.match_faq(current_query, policy_type)
                
                if faq_match:
                    # Store FAQ match
                    context.state["faq_matches"] = [faq_match.to_dict()]
                    context.state["response_confidence"] = 0.9
                    self.logger.info(f"Found FAQ match: {faq_match.question[:50]}...")
                else:
                    # No FAQ match found - this is normal, not an error
                    context.state["faq_matches"] = []
                    context.state["response_confidence"] = 0.2
                    self.logger.info("No FAQ match found")
            else:
                # FAQ service not available or no query
                context.state["faq_matches"] = []
                context.state["response_confidence"] = 0.1
                if not self.faq_matcher:
                    context.state["escalation_reason"] = "FAQ service not available"
                    
        except Exception as e:
            # Use comprehensive error handling
            from services.error_handler import error_handler, graceful_degradation
            
            error_response, should_escalate = error_handler.handle_error(
                exception=e,
                component="faq_search_tool",
                user_query=current_query,
                context={
                    "policy_type": policy_type,
                    "faq_service_available": self.faq_matcher is not None
                }
            )
            
            # Try graceful degradation if not escalating
            if not should_escalate:
                fallback_response = graceful_degradation.execute_fallback(
                    service_name="faq_matcher",
                    original_query=current_query,
                    policy_type=policy_type
                )
                
                if fallback_response:
                    context.state["response_content"] = fallback_response.content
                    context.state["response_source"] = fallback_response.source
                    context.state["response_confidence"] = fallback_response.confidence
                    context.state["should_escalate"] = fallback_response.should_escalate
                else:
                    context.state["response_content"] = error_response.content
                    context.state["response_source"] = error_response.source
                    context.state["response_confidence"] = error_response.confidence
                    context.state["should_escalate"] = should_escalate
            else:
                context.state["response_content"] = error_response.content
                context.state["response_source"] = error_response.source
                context.state["response_confidence"] = error_response.confidence
                context.state["should_escalate"] = should_escalate
            
            context.state["faq_matches"] = []
            context.state["escalation_reason"] = f"FAQ search error: {str(e)}"
            self.logger.error(f"Error in FAQ search: {e}")
        
        return context.state
    
    def _rag_search_tool(self, state: ConversationState) -> ConversationState:
        """
        Tool node for RAG search service with comprehensive error handling.
        
        Args:
            state: Current conversation state
            
        Returns:
            Updated conversation state
        """
        context = ConversationContext(state=state)
        context.increment_step()
        
        current_query = state.get("current_query", "")
        policy_type = state.get("selected_policy_type")
        
        try:
            if self.rag_retriever and current_query and policy_type:
                # Retrieve relevant chunks using RAG
                chunks = self.rag_retriever.retrieve_chunks(
                    query=current_query,
                    policy_type=policy_type,
                    top_k=3
                )
                
                if chunks:
                    # Store retrieved chunks as dictionaries
                    context.state["retrieved_chunks"] = [chunk.to_dict() for chunk in chunks]
                    
                    # Calculate confidence based on similarity scores
                    avg_similarity = sum(chunk.similarity_score for chunk in chunks) / len(chunks)
                    context.state["response_confidence"] = min(avg_similarity + 0.1, 0.95)
                    
                    self.logger.info(f"Retrieved {len(chunks)} chunks with avg similarity {avg_similarity:.3f}")
                else:
                    # No relevant chunks found - this is normal, not an error
                    context.state["retrieved_chunks"] = []
                    context.state["response_confidence"] = 0.2
                    self.logger.info("No relevant chunks found")
            else:
                # RAG service not available, no query, or no policy type
                context.state["retrieved_chunks"] = []
                context.state["response_confidence"] = 0.1
                
                if not self.rag_retriever:
                    context.state["escalation_reason"] = "Document search service not available"
                elif not policy_type:
                    context.state["escalation_reason"] = "Policy type not selected"
                    
        except Exception as e:
            # Use comprehensive error handling
            from services.error_handler import error_handler, graceful_degradation
            
            error_response, should_escalate = error_handler.handle_error(
                exception=e,
                component="rag_search_tool",
                user_query=current_query,
                context={
                    "policy_type": policy_type,
                    "rag_service_available": self.rag_retriever is not None
                }
            )
            
            # Try graceful degradation if not escalating
            if not should_escalate:
                fallback_response = graceful_degradation.execute_fallback(
                    service_name="rag_retriever",
                    original_query=current_query,
                    policy_type=policy_type
                )
                
                if fallback_response:
                    context.state["response_content"] = fallback_response.content
                    context.state["response_source"] = fallback_response.source
                    context.state["response_confidence"] = fallback_response.confidence
                    context.state["should_escalate"] = fallback_response.should_escalate
                else:
                    context.state["response_content"] = error_response.content
                    context.state["response_source"] = error_response.source
                    context.state["response_confidence"] = error_response.confidence
                    context.state["should_escalate"] = should_escalate
            else:
                context.state["response_content"] = error_response.content
                context.state["response_source"] = error_response.source
                context.state["response_confidence"] = error_response.confidence
                context.state["should_escalate"] = should_escalate
            
            context.state["retrieved_chunks"] = []
            context.state["escalation_reason"] = f"RAG search error: {str(e)}"
            self.logger.error(f"Error in RAG search: {e}")
        
        return context.state
    
    def process_query(self, query: str, policy_type: Optional[str] = None) -> AgentResponse:
        """
        Process a user query through the LangGraph workflow.
        
        Args:
            query: User's query
            policy_type: Optional policy type if already selected
            
        Returns:
            Agent response
        """
        # Create or update session state
        if not self.current_session:
            session_id = str(uuid.uuid4())
            initial_state = create_initial_state(session_id, query)
            self.current_session = ConversationContext(state=initial_state)
        else:
            self.current_session.add_user_message(query)
            self.current_session.reset_query_context()
        
        # Set policy type if provided
        if policy_type:
            self.current_session.set_policy_type(policy_type)
        
        # Run the graph with comprehensive error handling
        try:
            result = self.graph.invoke(self.current_session.state)
            
            # Extract response from final state
            response = AgentResponse(
                content=result.get("response_content", "I apologize, but I couldn't process your request."),
                source=result.get("response_source", "system"),
                confidence=result.get("response_confidence", 0.0),
                should_escalate=result.get("should_escalate", False),
                retrieved_chunks=result.get("retrieved_chunks"),
                claim_info=result.get("claim_info")
            )
            
            # Update session with response
            self.current_session.add_agent_response(response)
            
            return response
            
        except Exception as e:
            # Use comprehensive error handling
            from services.error_handler import error_handler
            
            error_response, should_escalate = error_handler.handle_error(
                exception=e,
                component="insurance_agent",
                user_query=query,
                context={
                    "policy_type": policy_type,
                    "session_id": self.current_session.state.get("session_id") if self.current_session else None
                }
            )
            
            # Update session with error response
            if self.current_session:
                self.current_session.add_agent_response(error_response)
            
            return error_response
    
    def get_conversation_state(self) -> Dict[str, Any]:
        """
        Get current conversation state summary.
        
        Returns:
            Dictionary containing conversation state information
        """
        if not self.current_session:
            return {"status": "no_active_session"}
        
        return self.current_session.get_conversation_summary()
    
    def reset_conversation(self) -> None:
        """Reset the current conversation session."""
        self.current_session = None
    
    def set_services(self, claim_service=None, faq_matcher=None, rag_retriever=None):
        """
        Inject service dependencies.
        
        Args:
            claim_service: Claim lookup service
            faq_matcher: FAQ matching service  
            rag_retriever: RAG retrieval service
        """
        self.claim_service = claim_service
        self.faq_matcher = faq_matcher
        self.rag_retriever = rag_retriever
        
        self.logger.info(f"Services injected - Claim: {claim_service is not None}, "
                        f"FAQ: {faq_matcher is not None}, RAG: {rag_retriever is not None}")