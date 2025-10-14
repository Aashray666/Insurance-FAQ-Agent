"""
Comprehensive error handling service for the Insurance Agent.

Provides error detection, classification, recovery strategies,
and user-friendly error messages with graceful degradation.
"""

import logging
import traceback
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from models.agent_response import AgentResponse


class ErrorType(Enum):
    """Types of errors that can occur in the system."""
    PDF_PROCESSING_ERROR = "pdf_processing_error"
    VECTOR_SEARCH_ERROR = "vector_search_error"
    FAQ_SEARCH_ERROR = "faq_search_error"
    CLAIM_LOOKUP_ERROR = "claim_lookup_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    NETWORK_ERROR = "network_error"
    DATA_CORRUPTION = "data_corruption"
    CONFIGURATION_ERROR = "configuration_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT_ERROR = "timeout_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"          # Minor issues, system can continue normally
    MEDIUM = "medium"    # Moderate issues, some functionality affected
    HIGH = "high"        # Major issues, significant functionality lost
    CRITICAL = "critical"  # System-breaking issues, requires immediate attention


@dataclass
class ErrorContext:
    """Context information for error handling."""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    original_exception: Optional[Exception]
    component: str
    timestamp: datetime
    user_query: Optional[str] = None
    recovery_suggestions: List[str] = None
    fallback_available: bool = False
    error_code: Optional[str] = None


class ErrorHandler:
    """
    Comprehensive error handling service.
    
    Provides error classification, recovery strategies, and user-friendly
    error messages with graceful degradation capabilities.
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.logger = logging.getLogger(__name__)
        self.error_patterns = self._initialize_error_patterns()
        self.recovery_strategies = self._initialize_recovery_strategies()
        self.fallback_responses = self._initialize_fallback_responses()
        
        # Error tracking
        self.error_counts = {}
        self.recent_errors = []
        self.max_recent_errors = 50
    
    def _initialize_error_patterns(self) -> Dict[str, ErrorType]:
        """Initialize patterns for error classification."""
        return {
            # PDF Processing Errors
            "pdf": ErrorType.PDF_PROCESSING_ERROR,
            "pdfplumber": ErrorType.PDF_PROCESSING_ERROR,
            "pypdf": ErrorType.PDF_PROCESSING_ERROR,
            "text extraction": ErrorType.PDF_PROCESSING_ERROR,
            "corrupt": ErrorType.DATA_CORRUPTION,
            
            # Vector Search Errors
            "faiss": ErrorType.VECTOR_SEARCH_ERROR,
            "embedding": ErrorType.VECTOR_SEARCH_ERROR,
            "vector": ErrorType.VECTOR_SEARCH_ERROR,
            "similarity": ErrorType.VECTOR_SEARCH_ERROR,
            "index": ErrorType.VECTOR_SEARCH_ERROR,
            
            # FAQ Search Errors
            "faq": ErrorType.FAQ_SEARCH_ERROR,
            "keyword": ErrorType.FAQ_SEARCH_ERROR,
            "matching": ErrorType.FAQ_SEARCH_ERROR,
            
            # Claim Lookup Errors
            "claim": ErrorType.CLAIM_LOOKUP_ERROR,
            "lookup": ErrorType.CLAIM_LOOKUP_ERROR,
            "database": ErrorType.CLAIM_LOOKUP_ERROR,
            
            # Service Errors
            "service unavailable": ErrorType.SERVICE_UNAVAILABLE,
            "connection": ErrorType.NETWORK_ERROR,
            "timeout": ErrorType.TIMEOUT_ERROR,
            "memory": ErrorType.MEMORY_ERROR,
            
            # Configuration Errors
            "config": ErrorType.CONFIGURATION_ERROR,
            "setting": ErrorType.CONFIGURATION_ERROR,
            "path": ErrorType.CONFIGURATION_ERROR,
            
            # Validation Errors
            "validation": ErrorType.VALIDATION_ERROR,
            "invalid": ErrorType.VALIDATION_ERROR,
            "format": ErrorType.VALIDATION_ERROR
        }
    
    def _initialize_recovery_strategies(self) -> Dict[ErrorType, List[str]]:
        """Initialize recovery strategies for different error types."""
        return {
            ErrorType.PDF_PROCESSING_ERROR: [
                "Try alternative PDF processing method",
                "Skip corrupted sections and continue",
                "Use cached processed content if available",
                "Fall back to manual text input"
            ],
            ErrorType.VECTOR_SEARCH_ERROR: [
                "Fall back to keyword-based search",
                "Use alternative embedding model",
                "Search in cached results",
                "Reduce search complexity"
            ],
            ErrorType.FAQ_SEARCH_ERROR: [
                "Use simple keyword matching",
                "Search in backup FAQ database",
                "Fall back to general responses",
                "Use cached FAQ results"
            ],
            ErrorType.CLAIM_LOOKUP_ERROR: [
                "Retry with different claim format",
                "Search in backup claim database",
                "Provide manual lookup instructions",
                "Escalate to human agent"
            ],
            ErrorType.SERVICE_UNAVAILABLE: [
                "Use cached responses",
                "Fall back to alternative service",
                "Provide offline information",
                "Schedule retry"
            ],
            ErrorType.NETWORK_ERROR: [
                "Retry with exponential backoff",
                "Use cached data",
                "Switch to offline mode",
                "Provide contact information"
            ],
            ErrorType.TIMEOUT_ERROR: [
                "Reduce query complexity",
                "Use faster search method",
                "Provide partial results",
                "Suggest trying again later"
            ],
            ErrorType.MEMORY_ERROR: [
                "Clear cache and retry",
                "Process in smaller chunks",
                "Use lightweight alternatives",
                "Restart service components"
            ]
        }
    
    def _initialize_fallback_responses(self) -> Dict[ErrorType, str]:
        """Initialize fallback responses for different error types."""
        return {
            ErrorType.PDF_PROCESSING_ERROR: """
I'm having trouble accessing the policy document right now. However, I can still help you with:

• General insurance information from our FAQ database
• Claim status lookup (if you have your claim number)
• Contact information for human support

What would you like to know about your insurance?
            """.strip(),
            
            ErrorType.VECTOR_SEARCH_ERROR: """
I'm experiencing some technical difficulties with document search, but I can still assist you using our FAQ database.

Let me try to answer your question using our frequently asked questions, or I can connect you with a human representative for detailed policy information.

Would you like me to search our FAQ database or connect you with customer service?
            """.strip(),
            
            ErrorType.FAQ_SEARCH_ERROR: """
I'm having trouble accessing our FAQ database at the moment. However, I can still help you with:

• Claim status lookup (provide your claim number)
• General insurance guidance
• Connect you with a human representative

How would you like to proceed?
            """.strip(),
            
            ErrorType.CLAIM_LOOKUP_ERROR: """
I'm having difficulty accessing the claim database right now. 

For immediate claim status information, please:
• Call our 24/7 claim hotline: 1-800-RELIANCE
• Visit our website's claim portal
• Use our mobile app

I can also connect you with a human representative who can look up your claim status directly.

Would you like me to connect you with customer service?
            """.strip(),
            
            ErrorType.SERVICE_UNAVAILABLE: """
Some of our services are temporarily unavailable, but I'm still here to help!

I can assist you with:
• General insurance questions
• Provide contact information
• Connect you with human support
• Schedule a callback

What would you like to do?
            """.strip(),
            
            ErrorType.NETWORK_ERROR: """
I'm experiencing connectivity issues, but I can still provide basic assistance.

For immediate help:
• Call customer service: 1-800-RELIANCE
• Visit our website: www.relianceinsurance.com
• Use our mobile app

I'll keep trying to restore full functionality. Is there anything urgent I can help you with using the information I have available?
            """.strip()
        }
    
    def handle_error(
        self, 
        exception: Exception, 
        component: str,
        user_query: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[AgentResponse, bool]:
        """
        Handle an error and provide appropriate response and recovery.
        
        Args:
            exception: The exception that occurred
            component: Component where error occurred
            user_query: User's query that caused the error
            context: Additional context information
            
        Returns:
            Tuple of (AgentResponse, should_escalate)
        """
        # Classify the error
        error_context = self._classify_error(exception, component, user_query, context)
        
        # Log the error
        self._log_error(error_context)
        
        # Track the error
        self._track_error(error_context)
        
        # Determine if escalation is needed
        should_escalate = self._should_escalate_error(error_context)
        
        # Generate user-friendly response
        response = self._generate_error_response(error_context, should_escalate)
        
        # Attempt recovery if possible
        recovery_attempted = self._attempt_recovery(error_context)
        
        return response, should_escalate
    
    def _classify_error(
        self, 
        exception: Exception, 
        component: str,
        user_query: Optional[str],
        context: Optional[Dict[str, Any]]
    ) -> ErrorContext:
        """Classify the error and determine its type and severity."""
        error_message = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        
        # Determine error type
        error_type = ErrorType.UNKNOWN_ERROR
        for pattern, etype in self.error_patterns.items():
            if pattern in error_message or pattern in exception_type or pattern in component.lower():
                error_type = etype
                break
        
        # Determine severity
        severity = self._determine_severity(exception, error_type, component)
        
        # Generate recovery suggestions
        recovery_suggestions = self.recovery_strategies.get(error_type, [])
        
        # Check if fallback is available
        fallback_available = error_type in self.fallback_responses
        
        return ErrorContext(
            error_type=error_type,
            severity=severity,
            message=str(exception),
            original_exception=exception,
            component=component,
            timestamp=datetime.now(),
            user_query=user_query,
            recovery_suggestions=recovery_suggestions,
            fallback_available=fallback_available,
            error_code=self._generate_error_code(error_type, component)
        )
    
    def _determine_severity(self, exception: Exception, error_type: ErrorType, component: str) -> ErrorSeverity:
        """Determine the severity of an error."""
        # Critical errors
        if isinstance(exception, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        
        if error_type in [ErrorType.DATA_CORRUPTION, ErrorType.CONFIGURATION_ERROR]:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in [ErrorType.SERVICE_UNAVAILABLE, ErrorType.MEMORY_ERROR]:
            return ErrorSeverity.HIGH
        
        if "core" in component.lower() or "main" in component.lower():
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if error_type in [ErrorType.VECTOR_SEARCH_ERROR, ErrorType.PDF_PROCESSING_ERROR]:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors (default)
        return ErrorSeverity.LOW
    
    def _generate_error_code(self, error_type: ErrorType, component: str) -> str:
        """Generate a unique error code for tracking."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        type_code = error_type.value[:3].upper()
        component_code = component[:3].upper()
        return f"ERR-{type_code}-{component_code}-{timestamp}"
    
    def _log_error(self, error_context: ErrorContext) -> None:
        """Log the error with appropriate level."""
        log_message = (
            f"Error in {error_context.component}: {error_context.message} "
            f"[Type: {error_context.error_type.value}, Severity: {error_context.severity.value}, "
            f"Code: {error_context.error_code}]"
        )
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, exc_info=error_context.original_exception)
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, exc_info=error_context.original_exception)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _track_error(self, error_context: ErrorContext) -> None:
        """Track error for pattern analysis."""
        error_key = f"{error_context.error_type.value}_{error_context.component}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Add to recent errors
        self.recent_errors.append(error_context)
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors.pop(0)
    
    def _should_escalate_error(self, error_context: ErrorContext) -> bool:
        """Determine if error should trigger escalation."""
        # Always escalate critical errors
        if error_context.severity == ErrorSeverity.CRITICAL:
            return True
        
        # Escalate if same error occurs frequently
        error_key = f"{error_context.error_type.value}_{error_context.component}"
        if self.error_counts.get(error_key, 0) >= 3:
            return True
        
        # Escalate specific error types
        escalation_types = [
            ErrorType.DATA_CORRUPTION,
            ErrorType.CONFIGURATION_ERROR,
            ErrorType.SERVICE_UNAVAILABLE
        ]
        
        if error_context.error_type in escalation_types:
            return True
        
        return False
    
    def _generate_error_response(self, error_context: ErrorContext, should_escalate: bool) -> AgentResponse:
        """Generate user-friendly error response."""
        if error_context.fallback_available and not should_escalate:
            # Use fallback response
            content = self.fallback_responses[error_context.error_type]
            confidence = 0.7
        else:
            # Generate escalation response
            content = self._generate_escalation_error_response(error_context)
            confidence = 0.3
        
        return AgentResponse(
            content=content,
            source="error_handler",
            confidence=confidence,
            should_escalate=should_escalate
        )
    
    def _generate_escalation_error_response(self, error_context: ErrorContext) -> str:
        """Generate error response that includes escalation."""
        return f"""I apologize, but I'm experiencing technical difficulties that prevent me from fully assisting you right now.

**Error Details:**
- Error Code: {error_context.error_code}
- Issue: Technical service interruption

**Immediate Assistance:**
- Customer Service: 1-800-RELIANCE (1-800-735-4262)
- Email: support@relianceinsurance.com
- Live Chat: Available 24/7 on our website

**What You Can Do:**
- Try your question again in a few minutes
- Contact our human representatives for immediate help
- Visit our website for self-service options

A human representative can provide the assistance you need right away. I've logged this issue for our technical team to resolve.

Would you like me to connect you with customer service now?"""
    
    def _attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from the error."""
        recovery_strategies = error_context.recovery_suggestions
        
        if not recovery_strategies:
            return False
        
        # For now, just log the recovery attempt
        # In a full implementation, this would execute actual recovery strategies
        self.logger.info(f"Recovery attempted for {error_context.error_code}: {recovery_strategies[0]}")
        
        return True
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "error_counts": dict(self.error_counts),
            "recent_error_count": len(self.recent_errors),
            "most_common_errors": sorted(
                self.error_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
        }
    
    def clear_error_history(self) -> None:
        """Clear error tracking history."""
        self.error_counts.clear()
        self.recent_errors.clear()
        self.logger.info("Error history cleared")


class GracefulDegradation:
    """
    Provides graceful degradation strategies when services fail.
    
    Implements fallback mechanisms to maintain basic functionality
    even when primary services are unavailable.
    """
    
    def __init__(self):
        """Initialize graceful degradation handler."""
        self.fallback_strategies = {
            "rag_retriever": self._rag_fallback,
            "faq_matcher": self._faq_fallback,
            "claim_service": self._claim_fallback,
            "pdf_processor": self._pdf_fallback
        }
        self.cached_responses = {}
        self.offline_data = {}
    
    def execute_fallback(self, service_name: str, original_query: str, **kwargs) -> Optional[AgentResponse]:
        """
        Execute fallback strategy for a failed service.
        
        Args:
            service_name: Name of the failed service
            original_query: Original user query
            **kwargs: Additional parameters for fallback
            
        Returns:
            Fallback response or None if no fallback available
        """
        if service_name in self.fallback_strategies:
            return self.fallback_strategies[service_name](original_query, **kwargs)
        
        return None
    
    def _rag_fallback(self, query: str, **kwargs) -> AgentResponse:
        """Fallback for RAG retriever failure."""
        return AgentResponse(
            content="""I'm having trouble accessing the detailed policy documents right now, but I can still help you with general insurance questions.

For specific policy details, I recommend:
• Checking your physical policy documents
• Calling customer service: 1-800-RELIANCE
• Visiting our website's policy section

What general insurance question can I help you with?""",
            source="fallback",
            confidence=0.6,
            should_escalate=False
        )
    
    def _faq_fallback(self, query: str, **kwargs) -> AgentResponse:
        """Fallback for FAQ matcher failure."""
        return AgentResponse(
            content="""I'm having trouble accessing our FAQ database, but I can still provide general guidance.

For common questions, you can:
• Visit our website's FAQ section
• Call customer service: 1-800-RELIANCE
• Check your policy documents

Is there a specific insurance topic I can help you with using general information?""",
            source="fallback",
            confidence=0.5,
            should_escalate=False
        )
    
    def _claim_fallback(self, query: str, **kwargs) -> AgentResponse:
        """Fallback for claim service failure."""
        return AgentResponse(
            content="""I'm unable to access the claim database right now. For immediate claim status:

**Direct Claim Support:**
• Claim Hotline: 1-800-RELIANCE (Press 1 for claims)
• Online Portal: www.relianceinsurance.com/claims
• Mobile App: Reliance Insurance Claims

**What You'll Need:**
• Your claim number
• Policy number
• Contact information

Would you like me to connect you with our claims department?""",
            source="fallback",
            confidence=0.8,
            should_escalate=True
        )
    
    def _pdf_fallback(self, query: str, **kwargs) -> AgentResponse:
        """Fallback for PDF processor failure."""
        return AgentResponse(
            content="""I'm having trouble processing policy documents right now. For policy information:

**Alternative Options:**
• Your physical policy documents
• Customer service: 1-800-RELIANCE
• Online policy portal
• Mobile app

I can still help with general insurance questions and claim status lookup.

What would you like to know about your insurance?""",
            source="fallback",
            confidence=0.6,
            should_escalate=False
        )


# Global error handler instance
error_handler = ErrorHandler()
graceful_degradation = GracefulDegradation()


def handle_service_error(func: Callable) -> Callable:
    """
    Decorator for handling service errors with graceful degradation.
    
    Args:
        func: Function to wrap with error handling
        
    Returns:
        Wrapped function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Extract component name from function
            component = func.__name__ if hasattr(func, '__name__') else 'unknown'
            
            # Handle the error
            response, should_escalate = error_handler.handle_error(
                exception=e,
                component=component,
                user_query=kwargs.get('query', ''),
                context=kwargs
            )
            
            # Try graceful degradation
            if not should_escalate:
                fallback_response = graceful_degradation.execute_fallback(
                    service_name=component,
                    original_query=kwargs.get('query', ''),
                    **kwargs
                )
                if fallback_response:
                    return fallback_response
            
            return response
    
    return wrapper