"""
Response Formatter Service for Insurance FAQ Agent

Simple service that formats RAG responses using Gemini 2.5 Flash API
"""

import os
import json
import requests
from typing import List, Optional
from models.chunk import Chunk
from models.response import FormattedResponse


class ResponseFormatter:
    """Simple formatter using Gemini 2.5 Flash or template fallback"""
    
    def __init__(self, use_gemini: bool = False):
        """
        Initialize the response formatter
        
        Args:
            use_gemini: Whether to use Gemini API for formatting
        """
        self.use_gemini = use_gemini
        # Hardcoded API key for college project
        self.api_key = "AIzaSyCbCM4jwE8BrDuTAfOlIOUoXjMw2CCzYP0"
    
    def format_rag_response(self, chunks: List[Chunk], query: str, policy_type: str) -> FormattedResponse:
        """
        Format RAG response using Gemini or template fallback
        
        Args:
            chunks: Retrieved chunks from RAG
            query: Original user query
            policy_type: Type of insurance policy
            
        Returns:
            FormattedResponse with formatted content
        """
        if not chunks:
            return FormattedResponse(
                content="❓ I couldn't find specific information about that in the policy documents.",
                source="No relevant chunks found",
                confidence=0.0
            )
        
        if self.use_gemini and self.api_key:
            return self._format_with_gemini(chunks, query, policy_type)
        else:
            return self._format_with_template(chunks, query, policy_type)
    
    def _format_with_gemini(self, chunks: List[Chunk], query: str, policy_type: str) -> FormattedResponse:
        """Format response using Gemini 2.5 Flash"""
        try:
            # Prepare context from chunks
            context_text = self._prepare_context(chunks)
            
            # Create prompt
            prompt = self._create_prompt(query, policy_type, context_text)
            
            # Call Gemini API
            response = self._call_gemini_api(prompt)
            
            # Calculate confidence
            avg_confidence = sum(chunk.similarity_score for chunk in chunks) / len(chunks)
            
            return FormattedResponse(
                content=response,
                source=f"Gemini-formatted from {len(chunks)} policy sections",
                confidence=avg_confidence
            )
            
        except Exception as e:
            print(f"⚠️ Gemini formatting failed: {e}. Using template formatting.")
            return self._format_with_template(chunks, query, policy_type)
    
    def _prepare_context(self, chunks: List[Chunk]) -> str:
        """Prepare context text from chunks"""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"Section {i} ({chunk.section}):\n{chunk.content}\n")
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, policy_type: str, context: str) -> str:
        """Create prompt for Gemini"""
        return f"""You are an insurance customer service assistant. Answer the customer's question based on the policy information provided.

Customer Question: {query}
Policy Type: {policy_type}

Policy Information:
{context}

Instructions:
- Provide a clear, helpful answer
- Use the policy information as your source
- Keep it professional but friendly
- Use bullet points if helpful
- Don't make up information

Response:"""
    
    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini 2.5 Flash API"""
        # Hardcoded URL for Gemini 2.5 Flash
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
        
        headers = {"Content-Type": "application/json"}
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 500
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    
    def _format_with_template(self, chunks: List[Chunk], query: str, policy_type: str) -> FormattedResponse:
        """Format response using template-based approach (fallback)"""
        # Clean and organize content
        content_by_section = {}
        for chunk in chunks:
            section = chunk.section.lower()
            if section not in content_by_section:
                content_by_section[section] = []
            
            # Clean the content
            clean_content = self._clean_content(chunk.content)
            if clean_content and clean_content not in content_by_section[section]:
                content_by_section[section].append(clean_content)
        
        # Build formatted response
        response_parts = []
        
        # Determine response type based on query
        if any(word in query.lower() for word in ['cover', 'coverage', 'include', 'protect']):
            response_parts.append("**Coverage Information:**")
            response_parts.append(f"Your {policy_type} policy covers the following:")
        elif any(word in query.lower() for word in ['exclude', 'exclusion', 'not cover', "doesn't cover"]):
            response_parts.append("**Exclusions:**")
            response_parts.append(f"Your {policy_type} policy does NOT cover:")
        elif any(word in query.lower() for word in ['claim', 'process', 'procedure', 'how to']):
            response_parts.append("**Claim Information:**")
            response_parts.append("Here's what you need to know:")
        else:
            response_parts.append("**Policy Information:**")
            response_parts.append(f"Based on your {policy_type} policy:")
        
        # Add content from each section
        for section, contents in content_by_section.items():
            for content in contents[:2]:  # Limit to 2 items per section
                response_parts.append(f"• {content}")
        
        # Add source information
        sections = list(content_by_section.keys())
        response_parts.append(f"\n📋 **Source Sections:** {', '.join(sections)}")
        
        # Calculate confidence
        avg_confidence = sum(chunk.similarity_score for chunk in chunks) / len(chunks)
        
        return FormattedResponse(
            content="\n".join(response_parts),
            source=f"Template-formatted from {len(chunks)} policy sections",
            confidence=avg_confidence
        )
    
    def _clean_content(self, content: str) -> str:
        """Clean and format content for better readability"""
        if not content:
            return ""
        
        # Remove extra whitespace and normalize
        content = " ".join(content.split())
        
        # Remove common policy document artifacts
        artifacts = [
            "nditions limitations and exceptions of this Policy",
            "GENERAL CONDITIONS",
            "CONDITIONS PRECEDENT TO LIABILITY",
            "Page |",
            "www.reliancegeneral.co.in"
        ]
        
        for artifact in artifacts:
            content = content.replace(artifact, "")
        
        # Clean up the content
        content = content.strip()
        
        # Ensure it starts with a capital letter
        if content and content[0].islower():
            content = content[0].upper() + content[1:]
        
        # Remove if too short or just numbers/symbols
        if len(content) < 10 or content.isdigit():
            return ""
        
        return content
    
    def format_claim_response(self, claim_data: dict) -> FormattedResponse:
        """Format claim status response"""
        if not claim_data:
            return FormattedResponse(
                content="❓ Claim not found. Please check your claim number and try again.",
                source="Claim database",
                confidence=0.0
            )
        
        status_emoji = {
            "Submitted": "📝",
            "Under Review": "🔍", 
            "Approved": "✅",
            "Processing Payment": "💳",
            "Settled": "✅",
            "Rejected": "❌"
        }
        
        emoji = status_emoji.get(claim_data.get("status", ""), "📋")
        
        response = f"""**Claim Status Update**

{emoji} **Claim Number:** {claim_data.get('claim_id', 'N/A')}
📅 **Submitted:** {claim_data.get('claim_date', 'N/A')}
🔄 **Current Status:** {claim_data.get('status', 'Unknown')}
💰 **Claim Amount:** ₹{claim_data.get('claim_amount', 'N/A'):,}
⏱️ **Estimated Resolution:** {claim_data.get('estimated_resolution', 'N/A')}

📝 **Details:** {claim_data.get('description', 'No additional details available')}"""
        
        return FormattedResponse(
            content=response,
            source="Claim database",
            confidence=1.0
        )
    
    def format_faq_response(self, faq_data: dict) -> FormattedResponse:
        """Format FAQ response"""
        if not faq_data:
            return FormattedResponse(
                content="❓ I couldn't find a specific answer to that question.",
                source="FAQ database",
                confidence=0.0
            )
        
        response = f"""**{faq_data.get('question', 'FAQ Response')}**

{faq_data.get('answer', 'No answer available')}"""
        
        return FormattedResponse(
            content=response,
            source="FAQ database", 
            confidence=0.9
        )