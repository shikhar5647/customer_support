from typing import Dict, Any, List, Type
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
import asyncio

from .base_agent import BaseAgent, AgentState

class IntentClassificationInput(BaseModel):
    """Input schema for intent classification"""
    customer_message: str = Field(..., description="Customer's raw message")
    conversation_history: List[str] = Field(default_factory=list)
    domain: str = Field(default="general", description="Business domain (ecommerce, telecom, utilities)")

class IntentClassificationOutput(BaseModel):
    """Output schema for intent classification"""
    primary_intent: str = Field(..., description="Primary customer intent")
    secondary_intents: List[str] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    entities: Dict[str, str] = Field(default_factory=dict)
    urgency_level: str = Field(default="medium", description="low, medium, high, critical")
    sentiment: str = Field(default="neutral", description="positive, neutral, negative")

