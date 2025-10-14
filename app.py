"""
Conversational Insurance FAQ Agent - Streamlit Interface
"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

# from agent.insurance_agent import InsuranceAgent  # Commented out to avoid LangGraph dependency
from services.claim_service import ClaimService
from services.faq_matcher import FAQMatcher
from services.rag_retriever import RAGRetriever
from services.response_formatter import ResponseFormatter
from services.escalation_service import EscalationService

# Page config
st.set_page_config(
    page_title="Insurance FAQ Agent",
    page_icon="🛡️",
    layout="wide"
)

# Temporarily disable caching to ensure fresh FAQ data
# @st.cache_resource  
def get_services():
    """Initialize and return the core services."""
    try:
        # Initialize services
        claim_service = ClaimService("data/sample_claims.json")
        faq_matcher = FAQMatcher("data/faq_database.json")
        # Force reload FAQ data to get latest updates
        faq_matcher.load_faq_data()
        escalation_service = EscalationService("data/escalations.json")
        
        # Initialize RAG retriever
        rag_retriever = None
        try:
            rag_retriever = RAGRetriever()
            rag_retriever.preload_all_databases()
            st.success("✅ All services loaded: FAQ, Claims, and Policy Document Search")
        except Exception as e:
            st.warning(f"RAG retriever not available: {e}")
        
        return {
            "claim_service": claim_service, 
            "faq_matcher": faq_matcher,
            "rag_retriever": rag_retriever,
            "escalation_service": escalation_service
        }
        
    except Exception as e:
        st.error(f"Error initializing services: {e}")
        return None

def main():
    st.title("🛡️ Conversational Insurance FAQ Agent")
    st.markdown("Ask me anything about insurance policies, claims, or coverage details!")
    
    # Get core services
    services = get_services()
    if not services:
        st.error("Failed to initialize system")
        return
    
    # Gemini formatting option (outside cached function)
    use_gemini = st.sidebar.checkbox("🤖 Use Gemini AI Formatting", value=True, help="Enable Gemini 2.5 Flash for better responses")
    
    if use_gemini:
        response_formatter = ResponseFormatter(use_gemini=True)
        st.sidebar.success("✅ Gemini 2.5 Flash enabled (hardcoded API key)")
    else:
        response_formatter = ResponseFormatter(use_gemini=False)
        st.sidebar.info("📝 Using template formatting")
    
    # Using direct services (no complex agent needed)
    claim_service = services["claim_service"]
    faq_matcher = services["faq_matcher"]
    rag_retriever = services["rag_retriever"]
    escalation_service = services["escalation_service"]
    
    # Display status
    st.info("🔧 Direct Services Mode - Claims ✅ FAQ ✅ RAG ✅ Gemini AI ✅")
    

    
    # Policy selection
    st.subheader("Select Your Policy Type")
    policy_type = st.selectbox(
        "Choose your policy:",
        ["Private Car", "Commercial Vehicle", "Two-wheeler"],
        index=0
    )
    
    st.divider()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Add welcome message
        welcome_msg = f"""
        👋 Hello! I'm your Insurance FAQ Agent. I can help you with:
        
        🔍 **Policy Questions**: Coverage details, exclusions, benefits
        📋 **Claims**: Status updates, filing process, required documents  
        📄 **Policy Documents**: Search through your {policy_type} policy
        
        Just ask me anything in natural language!
        """
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about insurance, claims, or policy details..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Direct services routing
                    response_text = ""
                    
                    # Check if it's a specific claim number lookup first
                    import re
                    claim_pattern = r"[A-Z]{2}\d{7}"
                    claim_match = re.search(claim_pattern, prompt.upper())
                    
                    # More specific claim lookup detection - only for actual claim numbers or very specific claim status queries
                    is_claim_lookup = claim_match or (
                        "claim" in prompt.lower() and 
                        any(word in prompt.lower() for word in ["check", "status", "track", "lookup", "find", "pc2024", "cv2024", "tw2024"])
                    )
                    
                    if is_claim_lookup:
                        # Handle claim queries
                        if claim_match:
                            claim_number = claim_match.group()
                            claim_info = claim_service.lookup_claim(claim_number)
                            
                            if claim_info:
                                # Convert claim_info to dict for response_formatter
                                claim_data = {
                                    'claim_id': claim_info.claim_id,
                                    'status': claim_info.status,
                                    'customer_name': claim_info.customer_name,
                                    'claim_amount': claim_info.claim_amount,
                                    'policy_type': claim_info.policy_type,
                                    'claim_date': claim_info.claim_date,
                                    'description': claim_info.description,
                                    'estimated_resolution': claim_info.estimated_resolution
                                }
                                formatted_response = response_formatter.format_claim_response(claim_data)
                                response_text = formatted_response.content
                            else:
                                response_text = f"❌ **Claim {claim_number} not found**\n\nPlease check the claim number and try again. Valid format: PC2024001, CV2024001, TW2024001"
                        else:
                            response_text = "🔍 **Claim Lookup Help**\n\nTo check a claim, please provide the claim number (e.g., PC2024001, CV2024001, TW2024001).\n\nSample claims to try:\n• PC2024001 (Private Car)\n• CV2024001 (Commercial Vehicle)\n• TW2024001 (Two-wheeler)"
                    
                    else:
                        # Handle FAQ and policy questions
                        # Try FAQ first
                        faq_match = faq_matcher.match_faq(prompt, policy_type)
                        if faq_match:

                            
                            # Convert faq_match to dict for response_formatter
                            faq_data = {
                                'question': faq_match.question,
                                'answer': faq_match.answer
                            }
                            formatted_response = response_formatter.format_faq_response(faq_data)
                            response_text = formatted_response.content
                        else:
                            # Try RAG if available
                            if rag_retriever:
                                chunks = rag_retriever.retrieve_chunks(prompt, policy_type, top_k=3)
                                if chunks:
                                    # Use response_formatter for RAG responses
                                    formatted_response = response_formatter.format_rag_response(chunks, prompt, policy_type)
                                    response_text = formatted_response.content
                                else:
                                    # No RAG results found - check for escalation
                                    sources_tried = ["FAQ", "RAG"]
                                    should_escalate, escalation_reason = escalation_service.should_escalate(
                                        prompt, sources_tried, False
                                    )
                                    
                                    if should_escalate:
                                        # Create escalation ticket
                                        ticket = escalation_service.escalate_query(
                                            customer_query=prompt,
                                            policy_type=policy_type,
                                            attempted_sources=sources_tried,
                                            agent_response="No relevant information found",
                                            escalation_reason=escalation_reason
                                        )
                                        response_text = escalation_service.format_escalation_response(ticket)
                                    else:
                                        response_text = "❓ I couldn't find specific information about that in the policy documents. Could you try rephrasing your question or contact our support team?"
                            else:
                                response_text = "❓ I couldn't find specific information about that. Could you try rephrasing your question?"
                    
                    # Check for escalation keywords in any response
                    if not is_claim_lookup and "escalate" not in response_text.lower():
                        escalation_keywords = ["complaint", "unsatisfied", "not helpful", "speak to human", "manager"]
                        if any(keyword in prompt.lower() for keyword in escalation_keywords):
                            sources_tried = ["FAQ", "RAG"] if rag_retriever else ["FAQ"]
                            ticket = escalation_service.escalate_query(
                                customer_query=prompt,
                                policy_type=policy_type,
                                attempted_sources=sources_tried,
                                agent_response=response_text,
                                escalation_reason="Customer requested human assistance"
                            )
                            response_text = escalation_service.format_escalation_response(ticket)
                    
                    # Display response
                    st.markdown(response_text)
                    
                    # Add escalation button for unsatisfactory responses
                    if "❓" in response_text and "escalate" not in response_text.lower():
                        st.markdown("---")
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            if st.button("🔄 Not Helpful? Escalate to Human", key=f"escalate_{len(st.session_state.messages)}"):
                                sources_tried = []
                                if faq_matcher.match_faq(prompt, policy_type):
                                    sources_tried.append("FAQ")
                                if rag_retriever:
                                    sources_tried.append("RAG")
                                if is_claim_lookup:
                                    sources_tried.append("Claims")
                                
                                ticket = escalation_service.escalate_query(
                                    customer_query=prompt,
                                    policy_type=policy_type,
                                    attempted_sources=sources_tried,
                                    agent_response=response_text,
                                    escalation_reason="Customer found response unsatisfactory"
                                )
                                
                                escalation_msg = escalation_service.format_escalation_response(ticket)
                                st.success("✅ Query escalated successfully!")
                                st.markdown(escalation_msg)
                                st.session_state.messages.append({"role": "assistant", "content": escalation_msg})
                                st.rerun()
                        
                        with col2:
                            if st.button("💡 Try Different Question", key=f"retry_{len(st.session_state.messages)}"):
                                st.info("💡 Try rephrasing your question or ask about:\n• Policy coverage\n• Claim process\n• Specific benefits")
                    

                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                except Exception as e:
                    error_msg = f"❌ Sorry, I encountered an error: {str(e)}\n\nPlease try again or contact support."
                    st.markdown(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Sidebar with examples and info
    with st.sidebar:
        st.header("💡 Example Questions")
        
        st.subheader("📋 Policy Questions")
        st.markdown("""
        - What is covered under my policy?
        - What are the exclusions?
        - How do I file a claim?
        - What documents are required?
        - What is NCB (No Claim Bonus)?
        """)
        
        st.subheader("🔍 Claim Queries")
        st.markdown("""
        - Check claim PC2024001
        - Check claim CV2024001  
        - Check claim TW2024001
        - What documents are needed for claims?
        - How long does claim processing take?
        """)
        
        st.subheader("📄 Policy Document Search")
        st.markdown("""
        - What are the exclusions for natural disasters?
        - Coverage for theft and fire
        - Deductible amounts and limits
        - Premium calculation factors
        """)
        
        st.subheader("📞 Contact Information")
        st.markdown("""
        - **Customer Service:** 1-800-RELIANCE
        - **Claims Support:** 1-800-CLAIMS  
        - **Emergency:** 1-800-EMERGENCY
        """)
        
        st.subheader("🔄 Reset Chat")
        if st.button("Clear Conversation"):
            st.session_state.messages = []

if __name__ == "__main__":
    main()