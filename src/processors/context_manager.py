"""
Context Management Module
"""
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

class ConversationTurn(BaseModel):
    """Single conversation turn"""
    turn_id: str
    timestamp: datetime
    role: str  # "customer" or "agent"
    message: str
    intent: Optional[str] = None
    entities: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ConversationContext(BaseModel):
    """Complete conversation context"""
    session_id: str
    customer_id: Optional[str] = None
    domain: str = "general"
    turns: List[ConversationTurn] = Field(default_factory=list)
    context_summary: str = ""
    active_entities: Dict[str, Any] = Field(default_factory=dict)
    conversation_state: str = "active"  # active, resolved, escalated
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ContextManager:
    """Manages conversation context and history"""
    
    def __init__(
        self,
        max_context_age_hours: int = 1,
        max_tokens: int = 5000,
        max_turns: int = 20
    ):
        self.max_context_age = timedelta(hours=max_context_age_hours)
        self.max_tokens = max_tokens
        self.max_turns = max_turns
        self.logger = logging.getLogger(__name__)
        
        # In-memory storage (replace with Redis/DB in production)
        self.contexts: Dict[str, ConversationContext] = {}
    
    def get_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context by session ID"""
        try:
            context = self.contexts.get(session_id)
            
            if context is None:
                return None
            
            # Check if context is expired
            if datetime.now() - context.updated_at > self.max_context_age:
                self.logger.info(f"Context {session_id} expired, removing")
                del self.contexts[session_id]
                return None
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to get context {session_id}: {str(e)}")
            return None
    
    def create_context(
        self,
        session_id: str,
        customer_id: Optional[str] = None,
        domain: str = "general"
    ) -> ConversationContext:
        """Create new conversation context"""
        try:
            context = ConversationContext(
                session_id=session_id,
                customer_id=customer_id,
                domain=domain
            )
            
            self.contexts[session_id] = context
            self.logger.info(f"Created new context for session {session_id}")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to create context {session_id}: {str(e)}")
            raise
    
    def add_turn(
        self,
        session_id: str,
        role: str,
        message: str,
        intent: Optional[str] = None,
        entities: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationContext:
        """Add a conversation turn"""
        try:
            context = self.get_context(session_id)
            if context is None:
                context = self.create_context(session_id)
            
            # Create turn
            turn = ConversationTurn(
                turn_id=f"{session_id}_{len(context.turns)}",
                timestamp=datetime.now(),
                role=role,
                message=message,
                intent=intent,
                entities=entities or {},
                metadata=metadata or {}
            )
            
            # Add turn to context
            context.turns.append(turn)
            context.updated_at = datetime.now()
            
            # Update active entities
            if entities:
                context.active_entities.update(entities)
            
            # Trim context if too long
            context = self._trim_context(context)
            
            # Update summary
            context.context_summary = self._generate_context_summary(context)
            
            self.contexts[session_id] = context
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to add turn to context {session_id}: {str(e)}")
            raise
    
    def get_context_for_agent(self, session_id: str) -> Dict[str, Any]:
        """Get context formatted for agent consumption"""
        try:
            context = self.get_context(session_id)
            if context is None:
                return {
                    "conversation_history": [],
                    "context_summary": "",
                    "active_entities": {},
                    "domain": "general"
                }
            
            # Format conversation history
            history = []
            for turn in context.turns[-10:]:  # Last 10 turns
                history.append({
                    "role": turn.role,
                    "content": turn.message,
                    "timestamp": turn.timestamp.isoformat(),
                    "intent": turn.intent
                })
            
            return {
                "conversation_history": history,
                "context_summary": context.context_summary,
                "active_entities": dict(context.active_entities),
                "domain": context.domain,
                "session_metadata": dict(context.metadata)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to format context for agent {session_id}: {str(e)}")
            return {
                "conversation_history": [],
                "context_summary": "Context unavailable",
                "active_entities": {},
                "domain": "general"
            }
    
    def update_conversation_state(self, session_id: str, state: str, metadata: Optional[Dict[str, Any]] = None):
        """Update conversation state"""
        try:
            context = self.get_context(session_id)
            if context:
                context.conversation_state = state
                context.updated_at = datetime.now()
                
                if metadata:
                    context.metadata.update(metadata)
                
                self.contexts[session_id] = context
                self.logger.info(f"Updated context {session_id} state to {state}")
            
        except Exception as e:
            self.logger.error(f"Failed to update context state {session_id}: {str(e)}")
    
    def _trim_context(self, context: ConversationContext) -> ConversationContext:
        """Trim context if it exceeds limits"""
        # Trim by number of turns
        if len(context.turns) > self.max_turns:
            context.turns = context.turns[-self.max_turns:]
        
        # Estimate token count and trim if needed
        estimated_tokens = self._estimate_token_count(context)
        while estimated_tokens > self.max_tokens and len(context.turns) > 5:
            # Remove oldest turns but keep some for context
            context.turns = context.turns[2:]
            estimated_tokens = self._estimate_token_count(context)
        
        return context
    
    def _estimate_token_count(self, context: ConversationContext) -> int:
        """Rough estimate of token count"""
        # Rough approximation: 1 token ≈ 4 characters
        total_chars = 0
        for turn in context.turns:
            total_chars += len(turn.message)
        total_chars += len(context.context_summary)
        
        return total_chars // 4
    
    def _generate_context_summary(self, context: ConversationContext) -> str:
        """Generate a summary of the conversation context"""
        if not context.turns:
            return ""
        
        # Simple summary generation
        customer_turns = [t for t in context.turns if t.role == "customer"]
        if not customer_turns:
            return ""
        
        # Extract main topics/intents
        intents = [t.intent for t in customer_turns if t.intent]
        entities = list(context.active_entities.keys())
        
        summary_parts = []
        
        if intents:
            unique_intents = list(dict.fromkeys(intents))
            summary_parts.append(f"Customer inquiries: {', '.join(unique_intents)}")
        
        if entities:
            summary_parts.append(f"Key entities: {', '.join(entities)}")
        
        return "; ".join(summary_parts)
    
    def cleanup_expired_contexts(self):
        """Clean up expired contexts"""
        try:
            current_time = datetime.now()
            expired_sessions = [
                session_id for session_id, context in self.contexts.items()
                if current_time - context.updated_at > self.max_context_age
            ]
            
            for session_id in expired_sessions:
                del self.contexts[session_id]
                self.logger.info(f"Cleaned up expired context: {session_id}")
            
            if expired_sessions:
                self.logger.info(f"Cleaned up {len(expired_sessions)} expired contexts")
            
        except Exception as e:
            self.logger.error(f"Context cleanup failed: {str(e)}")
