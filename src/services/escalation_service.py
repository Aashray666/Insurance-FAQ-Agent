"""
Escalation Service for Insurance FAQ Agent

Handles query escalation when the agent cannot provide satisfactory answers
"""

import json
import os
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass, asdict


@dataclass
class EscalationTicket:
    """Represents an escalated customer query"""
    ticket_id: str
    timestamp: str
    customer_query: str
    policy_type: str
    attempted_sources: List[str]  # ["FAQ", "RAG", "Claims"]
    agent_response: str
    escalation_reason: str
    status: str = "PENDING"  # PENDING, ASSIGNED, RESOLVED
    priority: str = "MEDIUM"  # LOW, MEDIUM, HIGH
    customer_feedback: Optional[str] = None


class EscalationService:
    """Service to handle query escalations for human intervention"""
    
    def __init__(self, escalation_file: str = "data/escalations.json"):
        """
        Initialize escalation service
        
        Args:
            escalation_file: Path to store escalation tickets
        """
        self.escalation_file = escalation_file
        self.tickets = self._load_tickets()
    
    def _load_tickets(self) -> List[EscalationTicket]:
        """Load existing escalation tickets from file"""
        if not os.path.exists(self.escalation_file):
            return []
        
        try:
            with open(self.escalation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return [EscalationTicket(**ticket) for ticket in data]
        except Exception as e:
            print(f"Error loading escalation tickets: {e}")
            return []
    
    def _save_tickets(self):
        """Save escalation tickets to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.escalation_file), exist_ok=True)
            
            with open(self.escalation_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(ticket) for ticket in self.tickets], f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving escalation tickets: {e}")
    
    def _generate_ticket_id(self) -> str:
        """Generate unique ticket ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"ESC-{timestamp}"
    
    def escalate_query(
        self, 
        customer_query: str, 
        policy_type: str,
        attempted_sources: List[str],
        agent_response: str,
        escalation_reason: str,
        priority: str = "MEDIUM"
    ) -> EscalationTicket:
        """
        Create an escalation ticket for human intervention
        
        Args:
            customer_query: The original customer question
            policy_type: Type of insurance policy
            attempted_sources: Sources that were tried (FAQ, RAG, Claims)
            agent_response: What the agent responded with
            escalation_reason: Why it's being escalated
            priority: Ticket priority level
            
        Returns:
            Created escalation ticket
        """
        ticket = EscalationTicket(
            ticket_id=self._generate_ticket_id(),
            timestamp=datetime.now().isoformat(),
            customer_query=customer_query,
            policy_type=policy_type,
            attempted_sources=attempted_sources,
            agent_response=agent_response,
            escalation_reason=escalation_reason,
            priority=priority
        )
        
        self.tickets.append(ticket)
        self._save_tickets()
        
        return ticket
    
    def get_pending_tickets(self) -> List[EscalationTicket]:
        """Get all pending escalation tickets"""
        return [ticket for ticket in self.tickets if ticket.status == "PENDING"]
    
    def get_ticket_by_id(self, ticket_id: str) -> Optional[EscalationTicket]:
        """Get specific ticket by ID"""
        for ticket in self.tickets:
            if ticket.ticket_id == ticket_id:
                return ticket
        return None
    
    def update_ticket_status(self, ticket_id: str, status: str, customer_feedback: str = None):
        """Update ticket status and add customer feedback"""
        ticket = self.get_ticket_by_id(ticket_id)
        if ticket:
            ticket.status = status
            if customer_feedback:
                ticket.customer_feedback = customer_feedback
            self._save_tickets()
    
    def get_escalation_stats(self) -> Dict:
        """Get escalation statistics"""
        total_tickets = len(self.tickets)
        pending = len([t for t in self.tickets if t.status == "PENDING"])
        resolved = len([t for t in self.tickets if t.status == "RESOLVED"])
        
        # Common escalation reasons
        reasons = {}
        for ticket in self.tickets:
            reason = ticket.escalation_reason
            reasons[reason] = reasons.get(reason, 0) + 1
        
        return {
            "total_tickets": total_tickets,
            "pending": pending,
            "resolved": resolved,
            "common_reasons": reasons
        }
    
    def should_escalate(self, query: str, sources_tried: List[str], found_results: bool) -> tuple[bool, str]:
        """
        Determine if a query should be escalated based on various factors
        
        Args:
            query: Customer query
            sources_tried: List of sources that were attempted
            found_results: Whether any results were found
            
        Returns:
            Tuple of (should_escalate, reason)
        """
        # Escalation triggers
        escalation_keywords = [
            "complaint", "unsatisfied", "not helpful", "wrong answer", 
            "speak to human", "manager", "supervisor", "escalate",
            "disappointed", "frustrated", "angry", "terrible service"
        ]
        
        complex_queries = [
            "legal", "lawsuit", "court", "attorney", "lawyer",
            "regulatory", "ombudsman", "grievance", "dispute"
        ]
        
        query_lower = query.lower()
        
        # Check for explicit escalation requests
        if any(keyword in query_lower for keyword in escalation_keywords):
            return True, "Customer requested human assistance"
        
        # Check for complex legal/regulatory queries
        if any(keyword in query_lower for keyword in complex_queries):
            return True, "Complex legal/regulatory query requiring human expertise"
        
        # Check if no results found after trying multiple sources
        if not found_results and len(sources_tried) >= 2:
            return True, "No relevant information found in available sources"
        
        # Check for very long queries (might be complex)
        if len(query.split()) > 20:
            return True, "Complex multi-part query requiring human review"
        
        return False, ""
    
    def format_escalation_response(self, ticket: EscalationTicket) -> str:
        """Format a user-friendly escalation response"""
        return f"""🔄 **Query Escalated for Human Assistance**

**Ticket ID:** {ticket.ticket_id}
**Status:** Your query has been forwarded to our customer service team

**What happens next:**
• A human agent will review your question within 24 hours
• You'll receive a detailed response via email or phone
• Complex queries typically take 1-2 business days to resolve

**Your Query:** "{ticket.customer_query}"

**Why escalated:** {ticket.escalation_reason}

**Reference Number:** {ticket.ticket_id}

💬 **Need immediate assistance?** 
Call our 24/7 helpline: 1800-XXX-XXXX
Email: support@insurance.com

Thank you for your patience! 🙏"""