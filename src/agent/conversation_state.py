"""
LangGraph conversation state management for the Insurance Agent.

Defines the state schema that maintains conversation context,
policy type selection, and routing information throughout the workflow.
"""

from typing import Dict, List, Optional, Any, TypedDict
from dataclasses import dataclass, field
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.query_intent import QueryIntent
from models.agent_response import AgentResponse


class ConversationState(TypedDict):
    """
    LangGraph state schema for conversation management.
    
    This state is passed between all nodes in the LangGraph workflow
    and maintains the complete conversation context.
    """
    
    # Current query and conversation
    current_query: str
    conversation_history: List[Dict[str, str]]
    
    # Policy context
    selected_policy_type: Optional[str]  # "Private Car", "Commercial Vehicle", "Two-wheeler"
    policy_context: Dict[str, Any]
    
    # Query processing
    detected_intent: Optional[QueryIntent]
    intent_confidence: float
    
    # Routing decisions
    route_decision: Optional[str]  # "claim_lookup", "faq_search", "rag_search", "escalation"
    should_escalate: bool
    escalation_reason: Optional[str]
    
    # Retrieved information
    retrieved_chunks: List[str]
    faq_matches: List[Dict[str, Any]]
    claim_info: Optional[Dict[str, Any]]
    
    # Response generation
    response_content: str
    response_source: str
    response_confidence: float
    
    # Session management
    session_id: str
    timestamp: str
    step_count: int


@dataclass
class ConversationContext:
    """
    Helper class for managing conversation context and state transitions.
    
    Provides utility methods for state validation, updates, and history management.
    """
    
    state: ConversationState
    
    def __post_init__(self):
        """Initialize default values if not provided."""
        if not self.state.get("conversation_history"):
            self.state["conversation_history"] = []
        
        if not self.state.get("policy_context"):
            self.state["policy_context"] = {}
        
        if not self.state.get("retrieved_chunks"):
            self.state["retrieved_chunks"] = []
        
        if not self.state.get("faq_matches"):
            self.state["faq_matches"] = []
        
        if self.state.get("intent_confidence") is None:
            self.state["intent_confidence"] = 0.0
        
        if self.state.get("response_confidence") is None:
            self.state["response_confidence"] = 0.0
        
        if not self.state.get("should_escalate"):
            self.state["should_escalate"] = False
        
        if not self.state.get("step_count"):
            self.state["step_count"] = 0
        
        if not self.state.get("timestamp"):
            self.state["timestamp"] = datetime.now().isoformat()
    
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to conversation history.
        
        Args:
            message: User's message content
        """
        self.state["conversation_history"].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        self.state["current_query"] = message
    
    def add_agent_response(self, response: AgentResponse) -> None:
        """
        Add an agent response to conversation history.
        
        Args:
            response: Agent's response object
        """
        self.state["conversation_history"].append({
            "role": "agent",
            "content": response.content,
            "source": response.source,
            "confidence": response.confidence,
            "timestamp": response.timestamp.isoformat()
        })
        
        # Update state with response information
        self.state["response_content"] = response.content
        self.state["response_source"] = response.source
        self.state["response_confidence"] = response.confidence
        self.state["should_escalate"] = response.should_escalate
        
        if response.retrieved_chunks:
            self.state["retrieved_chunks"] = response.retrieved_chunks
        
        if response.claim_info:
            self.state["claim_info"] = response.claim_info
    
    def set_policy_type(self, policy_type: str) -> None:
        """
        Set the selected policy type and update context.
        
        Args:
            policy_type: Selected policy type
            
        Raises:
            ValueError: If policy type is invalid
        """
        valid_types = ["Private Car", "Commercial Vehicle", "Two-wheeler"]
        if policy_type not in valid_types:
            raise ValueError(f"Invalid policy type. Must be one of: {valid_types}")
        
        self.state["selected_policy_type"] = policy_type
        self.state["policy_context"]["type"] = policy_type
        self.state["policy_context"]["selected_at"] = datetime.now().isoformat()
    
    def set_intent(self, intent: QueryIntent, confidence: float) -> None:
        """
        Set the detected query intent and confidence.
        
        Args:
            intent: Detected query intent
            confidence: Intent classification confidence (0-1)
        """
        self.state["detected_intent"] = intent
        self.state["intent_confidence"] = confidence
    
    def set_route_decision(self, route: str, reason: Optional[str] = None) -> None:
        """
        Set the routing decision for query processing.
        
        Args:
            route: Route decision ("claim_lookup", "faq_search", "rag_search", "escalation", "policy_selection", "complete")
            reason: Optional reason for the routing decision
        """
        valid_routes = [
            "claim_lookup", "faq_search", "rag_search", "escalation", 
            "policy_selection", "complete", "intent_classification", 
            "greeting_response", "response_generation", "rag_fallback"
        ]
        if route not in valid_routes:
            raise ValueError(f"Invalid route. Must be one of: {valid_routes}")
        
        self.state["route_decision"] = route
        if reason:
            self.state["policy_context"]["route_reason"] = reason
    
    def increment_step(self) -> None:
        """Increment the step counter for the current conversation."""
        self.state["step_count"] += 1
    
    def should_escalate_conversation(self) -> bool:
        """
        Determine if conversation should be escalated based on various factors.
        
        Uses the EscalationService for comprehensive escalation detection.
        
        Returns:
            True if conversation should be escalated
        """
        # Check explicit escalation flag
        if self.state.get("should_escalate"):
            return True
        
        # Import escalation service (avoid circular imports)
        try:
            from services.escalation_service import EscalationService
            
            escalation_service = EscalationService()
            
            # Get current query and conversation data
            current_query = self.state.get("current_query", "")
            conversation_history = self.state.get("conversation_history", [])
            response_confidence = self.state.get("response_confidence", 1.0)
            intent_confidence = self.state.get("intent_confidence", 1.0)
            
            # Use escalation service to detect triggers
            should_escalate, triggers, reason = escalation_service.detect_escalation_triggers(
                query=current_query,
                confidence=intent_confidence,
                conversation_history=conversation_history,
                response_confidence=response_confidence
            )
            
            if should_escalate:
                self.state["escalation_reason"] = reason
                return True
                
        except ImportError:
            # Fallback to basic escalation detection if service not available
            pass
        
        # Fallback escalation detection
        # Check for low confidence responses
        if self.state.get("response_confidence", 1.0) < 0.6:
            self.state["escalation_reason"] = "Low confidence response"
            return True
        
        # Check for repeated failed queries (low intent confidence)
        recent_intents = []
        for msg in self.state["conversation_history"][-6:]:  # Last 3 exchanges
            if msg.get("role") == "user":
                recent_intents.append(self.state.get("intent_confidence", 1.0))
        
        if len(recent_intents) >= 3 and all(conf < 0.5 for conf in recent_intents):
            self.state["escalation_reason"] = "Multiple low-confidence queries"
            return True
        
        # Check for explicit escalation keywords in recent messages
        escalation_keywords = ["manager", "supervisor", "complaint", "dissatisfied", "not satisfied", "human"]
        recent_messages = [msg.get("content", "").lower() for msg in self.state["conversation_history"][-2:]]
        
        for message in recent_messages:
            if any(keyword in message for keyword in escalation_keywords):
                self.state["escalation_reason"] = "Explicit escalation request"
                return True
        
        return False
    
    def is_policy_selected(self) -> bool:
        """
        Check if a policy type has been selected.
        
        Returns:
            True if policy type is selected
        """
        return bool(self.state.get("selected_policy_type"))
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current conversation state.
        
        Returns:
            Dictionary containing conversation summary
        """
        return {
            "session_id": self.state.get("session_id"),
            "policy_type": self.state.get("selected_policy_type"),
            "message_count": len(self.state.get("conversation_history", [])),
            "current_intent": str(self.state.get("detected_intent", "unknown")),
            "last_route": self.state.get("route_decision"),
            "should_escalate": self.should_escalate_conversation(),
            "step_count": self.state.get("step_count", 0)
        }
    
    def reset_query_context(self) -> None:
        """Reset query-specific context while preserving conversation history."""
        self.state["current_query"] = ""
        self.state["detected_intent"] = None
        self.state["intent_confidence"] = 0.0
        self.state["route_decision"] = None
        self.state["retrieved_chunks"] = []
        self.state["faq_matches"] = []
        self.state["claim_info"] = None
        self.state["response_content"] = ""
        self.state["response_source"] = ""
        self.state["response_confidence"] = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to dictionary for serialization.
        
        Returns:
            Dictionary representation of the state
        """
        # Convert QueryIntent enum to string for serialization
        state_dict = dict(self.state)
        if state_dict.get("detected_intent"):
            state_dict["detected_intent"] = str(state_dict["detected_intent"])
        
        return state_dict
    
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> "ConversationContext":
        """
        Create ConversationContext from dictionary.
        
        Args:
            state_dict: Dictionary representation of state
            
        Returns:
            ConversationContext instance
        """
        # Convert string back to QueryIntent enum
        if state_dict.get("detected_intent"):
            state_dict["detected_intent"] = QueryIntent.from_string(state_dict["detected_intent"])
        
        return cls(state=ConversationState(state_dict))


def create_initial_state(session_id: str, user_query: str = "") -> ConversationState:
    """
    Create initial conversation state for a new session.
    
    Args:
        session_id: Unique session identifier
        user_query: Initial user query (optional)
        
    Returns:
        Initial conversation state
    """
    state = ConversationState(
        current_query=user_query,
        conversation_history=[],
        selected_policy_type=None,
        policy_context={},
        detected_intent=None,
        intent_confidence=0.0,
        route_decision=None,
        should_escalate=False,
        escalation_reason=None,
        retrieved_chunks=[],
        faq_matches=[],
        claim_info=None,
        response_content="",
        response_source="",
        response_confidence=0.0,
        session_id=session_id,
        timestamp=datetime.now().isoformat(),
        step_count=0
    )
    
    context = ConversationContext(state=state)
    if user_query:
        context.add_user_message(user_query)
    
    return state