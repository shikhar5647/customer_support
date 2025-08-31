"""
Response Formatting Module
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class FormattedResponse(BaseModel):
    """Formatted response output"""
    formatted_text: str
    formatting_type: str  # plain, markdown, html, structured
    metadata: Dict[str, Any] = Field(default_factory=dict)
    formatting_time: float = 0.0

class ResponseFormatter:
    """Formats customer service responses for different output types"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Formatting templates
        self.templates = {
            "greeting": "Hello! Thank you for contacting us. {response}",
            "apology": "I apologize for the inconvenience. {response}",
            "escalation": "I understand your concern. Let me escalate this to our specialist team. {response}",
            "confirmation": "I've confirmed the following: {response}",
            "next_steps": "Here are the next steps: {response}",
            "contact_info": "For further assistance, please contact us at {contact_details}."
        }
        
        # Formatting rules
        self.formatting_rules = {
            "max_line_length": 80,
            "paragraph_spacing": 2,
            "bullet_style": "-",
            "emphasis_markers": ["**", "*", "_"]
        }
    
    def format_response(
        self, 
        response_data: Dict[str, Any], 
        format_type: str = "plain",
        template: Optional[str] = None
    ) -> FormattedResponse:
        """Format response based on type and template"""
        try:
            start_time = datetime.now().timestamp()
            
            # Extract response components
            response_text = response_data.get("response_text", "")
            intent_info = response_data.get("intent_classification", {})
            sop_info = response_data.get("sop_compliance", True)
            tools_used = response_data.get("tools_used", [])
            confidence_score = response_data.get("confidence_score", 0.0)
            
            # Apply template if specified
            if template and template in self.templates:
                response_text = self.templates[template].format(response=response_text)
            
            # Format based on type
            if format_type == "markdown":
                formatted_text = self._format_markdown(response_data)
            elif format_type == "html":
                formatted_text = self._format_html(response_data)
            elif format_type == "structured":
                formatted_text = self._format_structured(response_data)
            else:
                formatted_text = self._format_plain(response_data)
            
            # Calculate formatting time
            formatting_time = datetime.now().timestamp() - start_time
            
            return FormattedResponse(
                formatted_text=formatted_text,
                formatting_type=format_type,
                metadata={
                    "original_length": len(response_text),
                    "formatted_length": len(formatted_text),
                    "template_used": template,
                    "confidence_score": confidence_score
                },
                formatting_time=formatting_time
            )
            
        except Exception as e:
            self.logger.error(f"Response formatting failed: {str(e)}")
            return FormattedResponse(
                formatted_text=response_data.get("response_text", "Formatting error occurred"),
                formatting_type="plain",
                metadata={"error": str(e)}
            )
    
    def _format_plain(self, response_data: Dict[str, Any]) -> str:
        """Format response as plain text"""
        response_text = response_data.get("response_text", "")
        
        # Clean up text
        formatted = self._clean_text(response_text)
        
        # Add confidence indicator if low
        confidence_score = response_data.get("confidence_score", 0.0)
        if confidence_score < 0.7:
            formatted += "\n\nNote: This response was generated with lower confidence. Please verify the information."
        
        # Add SOP compliance note
        if not response_data.get("sop_compliance", True):
            formatted += "\n\nNote: This response may not fully comply with standard operating procedures."
        
        return formatted
    
    def _format_markdown(self, response_data: Dict[str, Any]) -> str:
        """Format response as markdown"""
        response_text = response_data.get("response_text", "")
        intent_info = response_data.get("intent_classification", {})
        tools_used = response_data.get("tools_used", [])
        
        formatted_parts = []
        
        # Add intent classification
        if intent_info:
            intent = intent_info.get("primary_intent", "")
            if intent:
                formatted_parts.append(f"**Intent:** {intent}")
        
        # Add main response
        formatted_parts.append(response_text)
        
        # Add tools used
        if tools_used:
            formatted_parts.append(f"\n**Tools Used:** {', '.join(tools_used)}")
        
        # Add confidence score
        confidence_score = response_data.get("confidence_score", 0.0)
        formatted_parts.append(f"\n**Confidence:** {confidence_score:.2f}")
        
        # Add SOP compliance
        sop_compliance = response_data.get("sop_compliance", True)
        formatted_parts.append(f"**SOP Compliant:** {'Yes' if sop_compliance else 'No'}")
        
        return "\n\n".join(formatted_parts)
    
    def _format_html(self, response_data: Dict[str, Any]) -> str:
        """Format response as HTML"""
        response_text = response_data.get("response_text", "")
        intent_info = response_data.get("intent_classification", {})
        tools_used = response_data.get("tools_used", [])
        
        html_parts = ['<div class="customer-response">']
        
        # Add intent classification
        if intent_info:
            intent = intent_info.get("primary_intent", "")
            if intent:
                html_parts.append(f'<p><strong>Intent:</strong> {intent}</p>')
        
        # Add main response
        html_parts.append(f'<p>{response_text}</p>')
        
        # Add tools used
        if tools_used:
            html_parts.append(f'<p><strong>Tools Used:</strong> {", ".join(tools_used)}</p>')
        
        # Add metadata
        html_parts.append('<div class="response-metadata">')
        confidence_score = response_data.get("confidence_score", 0.0)
        html_parts.append(f'<span class="confidence">Confidence: {confidence_score:.2f}</span>')
        
        sop_compliance = response_data.get("sop_compliance", True)
        html_parts.append(f'<span class="sop-compliance">SOP Compliant: {"Yes" if sop_compliance else "No"}</span>')
        html_parts.append('</div>')
        
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def _format_structured(self, response_data: Dict[str, Any]) -> str:
        """Format response in structured format"""
        # Create a structured representation
        structured = {
            "response": response_data.get("response_text", ""),
            "metadata": {
                "intent": response_data.get("intent_classification", {}),
                "confidence": response_data.get("confidence_score", 0.0),
                "sop_compliance": response_data.get("sop_compliance", True),
                "tools_used": response_data.get("tools_used", []),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Convert to formatted string
        import json
        return json.dumps(structured, indent=2)
    
    def _clean_text(self, text: str) -> str:
        """Clean and format text"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common formatting issues
        cleaned = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', cleaned)
        
        # Ensure proper paragraph spacing
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        
        return cleaned
    
    def add_emphasis(self, text: str, emphasis_type: str = "bold") -> str:
        """Add emphasis to text"""
        if emphasis_type == "bold":
            return f"**{text}**"
        elif emphasis_type == "italic":
            return f"*{text}*"
        elif emphasis_type == "underline":
            return f"__{text}__"
        else:
            return text
    
    def format_list(self, items: List[str], list_type: str = "bullet") -> str:
        """Format a list of items"""
        if list_type == "bullet":
            return '\n'.join([f"{self.formatting_rules['bullet_style']} {item}" for item in items])
        elif list_type == "numbered":
            return '\n'.join([f"{i+1}. {item}" for i, item in enumerate(items)])
        else:
            return '\n'.join(items)
    
    def add_contact_information(self, response: str, contact_details: Dict[str, str]) -> str:
        """Add contact information to response"""
        contact_text = "\n\nFor further assistance, please contact us:\n"
        
        if "phone" in contact_details:
            contact_text += f"Phone: {contact_details['phone']}\n"
        if "email" in contact_details:
            contact_text += f"Email: {contact_details['email']}\n"
        if "hours" in contact_details:
            contact_text += f"Hours: {contact_details['hours']}\n"
        
        return response + contact_text
    
    def format_escalation_response(self, response_data: Dict[str, Any]) -> str:
        """Format escalation response"""
        escalation_reason = response_data.get("escalation_reason", "Technical issue")
        
        formatted = f"I understand your concern about: {escalation_reason}\n\n"
        formatted += "I'm escalating this to our specialist team who will be able to assist you better.\n\n"
        formatted += "You should receive a response within 2-4 hours during business hours.\n\n"
        formatted += "Thank you for your patience."
        
        return formatted
    
    def get_formatting_options(self) -> Dict[str, Any]:
        """Get available formatting options"""
        return {
            "format_types": ["plain", "markdown", "html", "structured"],
            "templates": list(self.templates.keys()),
            "formatting_rules": self.formatting_rules,
            "emphasis_types": ["bold", "italic", "underline"],
            "list_types": ["bullet", "numbered", "plain"]
        }
