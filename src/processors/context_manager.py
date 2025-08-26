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
