from typing import Dict, Any, List, Type, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
import asyncio
import re
import json

from .base_agent import BaseAgent, AgentState

class QualityCheckInput(BaseModel):
    """Input schema for quality checking"""
    response_text: str = Field(..., description="Generated response to validate")
    customer_message: str = Field(..., description="Original customer message")
    intent_classification: Dict[str, Any] = Field(default_factory=dict)
    sop_snippets: List[str] = Field(default_factory=list)
    tool_results: Dict[str, Any] = Field(default_factory=dict)

class QualityCheckOutput(BaseModel):
    """Output schema for quality check"""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    sop_compliance: bool = Field(default=True, description="Adherence to SOPs")
    accuracy_score: float = Field(..., ge=0.0, le=1.0)
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    professionalism_score: float = Field(..., ge=0.0, le=1.0)
    completeness_score: float = Field(..., ge=0.0, le=1.0)
    safety_issues: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    contains_pii: bool = Field(default=False)
    toxicity_detected: bool = Field(default=False)
    hallucination_risk: str = Field(default="low", description="low, medium, high")
    
