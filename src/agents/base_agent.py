"""
Base Agent class for all agents in the system
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
import logging
import time
import uuid

logger = logging.getLogger(__name__)

class AgentState(BaseModel):
    """Base state model for agents"""
    agent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = Field(default_factory=time.time)
    messages: List[BaseMessage] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    confidence: float = 0.0

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the customer response system
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        max_retries: int = 3,
        timeout: int = 30
    ):
        self.name = name
        self.description = description
        self.max_retries = max_retries
        self.timeout = timeout
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    async def execute(self, state: AgentState) -> AgentState:
        """
        Execute the agent's main functionality
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Type[BaseModel]:
        """Get the input schema for this agent"""
        pass
    
    async def run_with_retry(self, state: AgentState) -> AgentState:
        """Execute agent with retry logic"""
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Executing {self.name}, attempt {attempt + 1}")
                result = await self.execute(state)
                self.logger.info(f"{self.name} completed successfully")
                return result
            except Exception as e:
                self.logger.error(f"{self.name} failed on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    state.error = f"Agent {self.name} failed after {self.max_retries} attempts: {str(e)}"
                    state.confidence = 0.0
                    return state
                await self._handle_retry_delay(attempt)
        
        return state
    
    async def _handle_retry_delay(self, attempt: int):
        """Handle delay between retries with exponential backoff"""
        delay = min(2 ** attempt, 10)  # Cap at 10 seconds
        await asyncio.sleep(delay)
    
    def validate_input(self, state: AgentState) -> bool:
        """Validate input state"""
        try:
            schema = self.get_schema()
            # Basic validation - can be extended
            return True
        except Exception as e:
            self.logger.error(f"Input validation failed: {str(e)}")
            return False
