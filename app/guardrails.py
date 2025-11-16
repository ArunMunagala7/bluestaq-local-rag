import re
from typing import Dict, List, Tuple

class Guardrails:
    """Basic content safety guardrails for RAG system."""
    
    def __init__(self, config: Dict):
        self.blocked_topics = config.get("guardrails", {}).get("blocked_topics", [])
        self.pii_patterns = config.get("guardrails", {}).get("pii_patterns", {})
        
    def check_query_safety(self, query: str) -> Tuple[bool, str]:
        """
        Check if query violates topic restrictions.
        
        Returns:
            (is_safe, reason) - is_safe=False if blocked, reason explains why
        """
        query_lower = query.lower()
        
        # Check blocked topics
        for topic in self.blocked_topics:
            if topic.lower() in query_lower:
                return False, f"This topic is restricted: {topic}"
        
        return True, ""
    
    def redact_pii(self, text: str) -> str:
        """
        Redact PII from text using regex patterns.
        
        Returns:
            Text with PII replaced by [REDACTED]
        """
        redacted = text
        
        # Email
        if self.pii_patterns.get("email"):
            redacted = re.sub(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                '[REDACTED_EMAIL]',
                redacted
            )
        
        # Phone (basic US format)
        if self.pii_patterns.get("phone"):
            redacted = re.sub(
                r'\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
                '[REDACTED_PHONE]',
                redacted
            )
        
        # SSN (XXX-XX-XXXX format)
        if self.pii_patterns.get("ssn"):
            redacted = re.sub(
                r'\b\d{3}-\d{2}-\d{4}\b',
                '[REDACTED_SSN]',
                redacted
            )
        
        # Credit card (basic pattern)
        if self.pii_patterns.get("credit_card"):
            redacted = re.sub(
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
                '[REDACTED_CC]',
                redacted
            )
        
        return redacted
    
    def validate_answer(self, answer: str, sources: List[Dict]) -> Tuple[bool, str]:
        """
        Basic validation that answer isn't empty and sources were used.
        
        Returns:
            (is_valid, warning) - warning message if validation fails
        """
        if not answer or len(answer.strip()) < 10:
            return False, "Answer too short or empty"
        
        if not sources:
            return False, "No sources retrieved - cannot answer"
        
        return True, ""
