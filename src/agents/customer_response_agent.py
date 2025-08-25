from typing import Dict, Any, List, Type, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph import StateGraph, END
import asyncio
import json

from .base_agent import BaseAgent, AgentState
from .intent_classifier_agent import IntentClassifierAgent
from .sop_retrieval_agent import SOPRetrievalAgent
from .tool_orchestrator_agent import ToolOrchestratorAgent
from .quality_checker_agent import QualityCheckerAgent

class CustomerResponseInput(BaseModel):
    """Input schema for customer response generation"""
    customer_message: str = Field(..., description="Customer's message")
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    domain: str = Field(default="general", description="Business domain")
    session_id: Optional[str] = Field(None, description="Conversation session ID")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

class CustomerResponseOutput(BaseModel):
    """Output schema for customer response"""
    response_text: str = Field(..., description="Generated response to customer")
    intent_classification: Dict[str, Any] = Field(default_factory=dict)
    sop_compliance: bool = Field(default=True)
    tools_used: List[str] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    escalation_required: bool = Field(default=False)
    escalation_reason: Optional[str] = Field(None)
    response_metadata: Dict[str, Any] = Field(default_factory=dict)

