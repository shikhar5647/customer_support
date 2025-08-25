from typing import Dict, Any, List, Type, Optional
from pydantic import BaseModel, Field
import asyncio

from .base_agent import BaseAgent, AgentState

class SOPRetrievalInput(BaseModel):
    """Input schema for SOP retrieval"""
    intent: str = Field(..., description="Customer intent")
    domain: str = Field(..., description="Business domain")
    entities: Dict[str, str] = Field(default_factory=dict)
    urgency_level: str = Field(default="medium")

class SOPRetrievalOutput(BaseModel):
    """Output schema for SOP retrieval"""
    relevant_sops: List[Dict[str, Any]] = Field(default_factory=list)
    sop_snippets: List[str] = Field(default_factory=list)
    confidence_scores: List[float] = Field(default_factory=list)
    total_sops_found: int = 0