"""
Fallback handler for managing system failures and escalations
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import json

logger = logging.getLogger(__name__)

class FallbackReason(Enum):
    """Reasons for fallback"""
    LOW_CONFIDENCE = "low_confidence"
    MODEL_FAILURE = "model_failure"
    TOOL_FAILURE = "tool_failure"
    TIMEOUT = "timeout"
    INVALID_SOP = "invalid_sop"
    HALLUCINATION_DETECTED = "hallucination_detected"
    SYSTEM_ERROR = "system_error"
    ESCALATION_REQUESTED = "escalation_requested"

class FallbackLevel(Enum):
    """Fallback levels"""
    RETRY = "retry"
    SIMPLIFIED_RESPONSE = "simplified_response"
    HUMAN_ESCALATION = "human_escalation"
    SYSTEM_SHUTDOWN = "system_shutdown"

class FallbackAction:
    """Represents a fallback action"""
    def __init__(self, reason: FallbackReason, level: FallbackLevel, description: str, 
                 timestamp: datetime, context: Dict[str, Any]):
        self.reason = reason
        self.level = level
        self.description = description
        self.timestamp = timestamp
        self.context = context
        self.resolved = False
        self.resolution_time = None
    
    def dict(self):
        """Convert to dictionary"""
        return {
            "reason": self.reason.value,
            "level": self.level.value,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None
        }

class FallbackHandler:
    """Handles system fallbacks and escalations"""
    
    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        confidence_threshold: float = 0.7,
        timeout_threshold: float = 30.0,
        escalation_threshold: int = 3
    ):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.confidence_threshold = confidence_threshold
        self.timeout_threshold = timeout_threshold
        self.escalation_threshold = escalation_threshold
        
        self.fallback_history: List[FallbackAction] = []
        self.retry_counts: Dict[str, int] = {}
        self.escalation_count = 0
        
        # Fallback strategies
        self.fallback_strategies = {
            FallbackReason.LOW_CONFIDENCE: self._handle_low_confidence,
            FallbackReason.MODEL_FAILURE: self._handle_model_failure,
            FallbackReason.TOOL_FAILURE: self._handle_tool_failure,
            FallbackReason.TIMEOUT: self._handle_timeout,
            FallbackReason.INVALID_SOP: self._handle_invalid_sop,
            FallbackReason.HALLUCINATION_DETECTED: self._handle_hallucination,
            FallbackReason.SYSTEM_ERROR: self._handle_system_error,
            FallbackReason.ESCALATION_REQUESTED: self._handle_escalation_request
        }
    
    async def handle_fallback(
        self,
        reason: FallbackReason,
        context: Dict[str, Any],
        original_request: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle a fallback situation"""
        
        try:
            # Create fallback action
            fallback_action = FallbackAction(
                reason=reason,
                level=FallbackLevel.RETRY,
                description=f"Fallback triggered: {reason.value}",
                timestamp=datetime.now(),
                context=context
            )
            
            # Add to history
            self.fallback_history.append(fallback_action)
            
            # Get appropriate strategy
            strategy = self.fallback_strategies.get(reason)
            if strategy:
                result = await strategy(fallback_action, context, original_request)
            else:
                result = await self._handle_generic_fallback(fallback_action, context)
            
            # Update fallback action
            fallback_action.resolved = True
            fallback_action.resolution_time = datetime.now()
            
            logger.info(f"Fallback handled successfully: {reason.value}")
            return result
            
        except Exception as e:
            logger.error(f"Error handling fallback: {str(e)}")
            return await self._handle_critical_fallback(reason, context, str(e))
    
    async def _handle_low_confidence(
        self, 
        fallback_action: FallbackAction, 
        context: Dict[str, Any],
        original_request: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle low confidence fallback"""
        
        # Check if we should escalate
        if self.escalation_count >= self.escalation_threshold:
            fallback_action.level = FallbackLevel.HUMAN_ESCALATION
            return await self._escalate_to_human(fallback_action, context)
        
        # Try simplified response
        fallback_action.level = FallbackLevel.SIMPLIFIED_RESPONSE
        
        try:
            # Generate a more conservative response
            simplified_response = await self._generate_simplified_response(context)
            
            return {
                "success": True,
                "response": simplified_response,
                "fallback_type": "simplified_response",
                "confidence": 0.8,  # Conservative confidence
                "escalation_required": False
            }
            
        except Exception as e:
            logger.warning(f"Simplified response failed: {str(e)}")
            return await self._escalate_to_human(fallback_action, context)
    
    async def _handle_model_failure(
        self, 
        fallback_action: FallbackAction, 
        context: Dict[str, Any],
        original_request: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle model failure fallback"""
        
        # Check retry count
        request_id = context.get("request_id", "unknown")
        retry_count = self.retry_counts.get(request_id, 0)
        
        if retry_count < self.max_retries:
            # Retry with delay
            fallback_action.level = FallbackLevel.RETRY
            self.retry_counts[request_id] = retry_count + 1
            
            await asyncio.sleep(self.retry_delay * (2 ** retry_count))  # Exponential backoff
            
            return {
                "success": False,
                "fallback_type": "retry",
                "retry_count": retry_count + 1,
                "message": f"Model failure, retrying... (attempt {retry_count + 1})"
            }
        else:
            # Escalate to human
            fallback_action.level = FallbackLevel.HUMAN_ESCALATION
            return await self._escalate_to_human(fallback_action, context)
    
    async def _handle_tool_failure(
        self, 
        fallback_action: FallbackAction, 
        context: Dict[str, Any],
        original_request: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle tool failure fallback"""
        
        tool_id = context.get("tool_id", "unknown")
        error_message = context.get("error", "Unknown tool error")
        
        # Try alternative tools if available
        alternative_tools = await self._find_alternative_tools(tool_id, context)
        
        if alternative_tools:
            fallback_action.level = FallbackLevel.RETRY
            return {
                "success": False,
                "fallback_type": "alternative_tool",
                "alternative_tools": alternative_tools,
                "message": f"Tool {tool_id} failed, trying alternatives"
            }
        else:
            # Escalate to human
            fallback_action.level = FallbackLevel.HUMAN_ESCALATION
            return await self._escalate_to_human(fallback_action, context)
    
    async def _handle_timeout(
        self, 
        fallback_action: FallbackAction, 
        context: Dict[str, Any],
        original_request: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle timeout fallback"""
        
        # Check if this is a repeated timeout
        request_id = context.get("request_id", "unknown")
        timeout_count = context.get("timeout_count", 0)
        
        if timeout_count < 2:
            # Retry with shorter timeout
            fallback_action.level = FallbackLevel.RETRY
            context["timeout_count"] = timeout_count + 1
            
            return {
                "success": False,
                "fallback_type": "retry_timeout",
                "timeout_count": timeout_count + 1,
                "message": "Request timed out, retrying with shorter timeout"
            }
        else:
            # Escalate to human
            fallback_action.level = FallbackLevel.HUMAN_ESCALATION
            return await self._escalate_to_human(fallback_action, context)
    
    async def _handle_invalid_sop(
        self, 
        fallback_action: FallbackAction, 
        context: Dict[str, Any],
        original_request: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle invalid SOP fallback"""
        
        # Try to find relevant SOPs
        relevant_sops = await self._find_relevant_sops(context)
        
        if relevant_sops:
            fallback_action.level = FallbackLevel.RETRY
            return {
                "success": False,
                "fallback_type": "alternative_sop",
                "relevant_sops": relevant_sops,
                "message": "Invalid SOP found, trying alternatives"
            }
        else:
            # Escalate to human
            fallback_action.level = FallbackLevel.HUMAN_ESCALATION
            return await self._escalate_to_human(fallback_action, context)
    
    async def _handle_hallucination(
        self, 
        fallback_action: FallbackAction, 
        context: Dict[str, Any],
        original_request: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle hallucination detection fallback"""
        
        # Always escalate hallucinations to human
        fallback_action.level = FallbackLevel.HUMAN_ESCALATION
        
        return await self._escalate_to_human(fallback_action, context)
    
    async def _handle_system_error(
        self, 
        fallback_action: FallbackAction, 
        context: Dict[str, Any],
        original_request: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle system error fallback"""
        
        error_severity = context.get("severity", "medium")
        
        if error_severity == "critical":
            fallback_action.level = FallbackLevel.SYSTEM_SHUTDOWN
            return await self._handle_system_shutdown(fallback_action, context)
        else:
            fallback_action.level = FallbackLevel.HUMAN_ESCALATION
            return await self._escalate_to_human(fallback_action, context)
    
    async def _handle_escalation_request(
        self, 
        fallback_action: FallbackAction, 
        context: Dict[str, Any],
        original_request: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle explicit escalation request"""
        
        fallback_action.level = FallbackLevel.HUMAN_ESCALATION
        return await self._escalate_to_human(fallback_action, context)
    
    async def _handle_generic_fallback(
        self, 
        fallback_action: FallbackAction, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle generic fallback"""
        
        fallback_action.level = FallbackLevel.HUMAN_ESCALATION
        return await self._escalate_to_human(fallback_action, context)
    
    async def _handle_critical_fallback(
        self, 
        reason: FallbackReason, 
        context: Dict[str, Any], 
        error_message: str
    ) -> Dict[str, Any]:
        """Handle critical fallback when normal handling fails"""
        
        logger.critical(f"Critical fallback failure: {reason.value} - {error_message}")
        
        return {
            "success": False,
            "fallback_type": "critical_failure",
            "error": error_message,
            "escalation_required": True,
            "message": "System experiencing critical failure, immediate human intervention required"
        }
    
    async def _escalate_to_human(
        self, 
        fallback_action: FallbackAction, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Escalate to human agent"""
        
        self.escalation_count += 1
        
        # Create escalation ticket
        escalation_ticket = await self._create_escalation_ticket(fallback_action, context)
        
        return {
            "success": False,
            "fallback_type": "human_escalation",
            "escalation_ticket": escalation_ticket,
            "escalation_required": True,
            "message": "Issue escalated to human agent",
            "estimated_wait_time": "5-10 minutes"
        }
    
    async def _handle_system_shutdown(
        self, 
        fallback_action: FallbackAction, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle system shutdown"""
        
        logger.critical("System shutdown initiated due to critical error")
        
        # Perform cleanup
        await self._perform_system_cleanup()
        
        return {
            "success": False,
            "fallback_type": "system_shutdown",
            "escalation_required": True,
            "message": "System shutdown initiated, contact system administrator immediately"
        }
    
    async def _generate_simplified_response(self, context: Dict[str, Any]) -> str:
        """Generate a simplified, conservative response"""
        
        customer_message = context.get("customer_message", "")
        
        # Simple template-based responses
        if "order" in customer_message.lower():
            return "I understand you have an order-related question. Let me connect you with a specialist who can better assist you."
        elif "billing" in customer_message.lower() or "payment" in customer_message.lower():
            return "I can see you have a billing question. Let me transfer you to our billing department for immediate assistance."
        elif "technical" in customer_message.lower() or "issue" in customer_message.lower():
            return "I understand you're experiencing a technical issue. Let me connect you with our technical support team."
        else:
            return "I understand your question and want to ensure you get the best possible assistance. Let me connect you with a customer service specialist."
    
    async def _find_alternative_tools(self, failed_tool_id: str, context: Dict[str, Any]) -> List[str]:
        """Find alternative tools for a failed tool"""
        
        # This would typically query a tool registry
        # For now, return empty list
        return []
    
    async def _find_relevant_sops(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find relevant SOPs for the context"""
        
        # This would typically query an SOP database
        # For now, return empty list
        return []
    
    async def _create_escalation_ticket(
        self, 
        fallback_action: FallbackAction, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an escalation ticket"""
        
        ticket = {
            "ticket_id": f"ESC{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "created_at": datetime.now(),
            "priority": "high",
            "reason": fallback_action.reason.value,
            "context": context,
            "status": "open",
            "assigned_to": None
        }
        
        # In a real system, this would be saved to a database
        logger.info(f"Escalation ticket created: {ticket['ticket_id']}")
        
        return ticket
    
    async def _perform_system_cleanup(self):
        """Perform system cleanup before shutdown"""
        
        logger.info("Performing system cleanup...")
        
        # Close connections, save state, etc.
        await asyncio.sleep(1)  # Simulate cleanup time
        
        logger.info("System cleanup completed")
    
    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get fallback statistics"""
        
        if not self.fallback_history:
            return {}
        
        # Calculate statistics
        total_fallbacks = len(self.fallback_history)
        resolved_fallbacks = len([f for f in self.fallback_history if f.resolved])
        
        # Count by reason
        reason_counts = {}
        for fallback in self.fallback_history:
            reason = fallback.reason.value
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        # Count by level
        level_counts = {}
        for fallback in self.fallback_history:
            level = fallback.level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            "total_fallbacks": total_fallbacks,
            "resolved_fallbacks": resolved_fallbacks,
            "resolution_rate": resolved_fallbacks / total_fallbacks if total_fallbacks > 0 else 0,
            "reason_distribution": reason_counts,
            "level_distribution": level_counts,
            "escalation_count": self.escalation_count,
            "retry_counts": dict(self.retry_counts)
        }
    
    def reset_fallback_counts(self):
        """Reset fallback counters"""
        self.retry_counts.clear()
        self.escalation_count = 0
    
    def export_fallback_history(self, file_path: str = "fallback_history.json"):
        """Export fallback history to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump([f.dict() for f in self.fallback_history], f, indent=2, default=str, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error exporting fallback history: {e}")
            return False
