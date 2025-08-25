from typing import Dict, Any, List, Type, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
import asyncio

from .base_agent import BaseAgent, AgentState

class ToolOrchestrationInput(BaseModel):
    """Input schema for tool orchestration"""
    intent: str = Field(..., description="Customer intent")
    entities: Dict[str, str] = Field(default_factory=dict)
    customer_message: str = Field(..., description="Original customer message")
    domain: str = Field(..., description="Business domain")

class ToolOrchestrationOutput(BaseModel):
    """Output schema for tool orchestration"""
    required_tools: List[str] = Field(default_factory=list)
    tool_parameters: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    execution_order: List[str] = Field(default_factory=list)
    tool_results: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error_details: Optional[str] = None

