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
    
class SOPRetrievalAgent(BaseAgent):
    """Agent for retrieving relevant SOPs"""
    
    def __init__(
        self,
        sop_manager=None,  # Will be injected
        max_sops: int = 3,
        min_similarity_threshold: float = 0.7,
        **kwargs
    ):
        super().__init__(name="SOPRetrieval", **kwargs)
        self.sop_manager = sop_manager
        self.max_sops = max_sops
        self.min_similarity_threshold = min_similarity_threshold
    
    def get_schema(self) -> Type[BaseModel]:
        return SOPRetrievalInput
    
    async def execute(self, state: AgentState) -> AgentState:
        """Execute SOP retrieval"""
        try:
            # Extract parameters from state
            intent = state.context.get("primary_intent", "")
            domain = state.context.get("domain", "general")
            entities = state.context.get("entities", {})
            urgency_level = state.context.get("urgency_level", "medium")
            
            if not intent:
                raise ValueError("Intent is required for SOP retrieval")
            
            if not self.sop_manager:
                # Fallback: return empty results
                self.logger.warning("No SOP manager configured, returning empty results")
                state.context["sop_retrieval"] = {
                    "relevant_sops": [],
                    "sop_snippets": [],
                    "confidence_scores": [],
                    "total_sops_found": 0
                }
                state.confidence = 0.0
                return state
            
            # Search for relevant SOPs
            search_results = await self._search_sops(intent, domain, entities, urgency_level)
            
            # Filter by confidence threshold
            filtered_results = [
                result for result in search_results 
                if result.get("similarity_score", 0) >= self.min_similarity_threshold
            ]
            
            # Limit results
            top_results = filtered_results[:self.max_sops]
            
            # Extract snippets and scores
            sop_snippets = [result.get("content", "") for result in top_results]
            confidence_scores = [result.get("similarity_score", 0.0) for result in top_results]
            
            # Update state
            retrieval_output = {
                "relevant_sops": top_results,
                "sop_snippets": sop_snippets,
                "confidence_scores": confidence_scores,
                "total_sops_found": len(filtered_results)
            }
            
            state.context["sop_retrieval"] = retrieval_output
            state.confidence = max(confidence_scores) if confidence_scores else 0.0
            
            self.logger.info(f"Retrieved {len(top_results)} SOPs for intent: {intent}")
            
        except Exception as e:
            self.logger.error(f"SOP retrieval failed: {str(e)}")
            state.error = f"SOP retrieval error: {str(e)}"
            state.confidence = 0.0
        
        return state
    
    async def _search_sops(
        self, 
        intent: str, 
        domain: str, 
        entities: Dict[str, str], 
        urgency_level: str
    ) -> List[Dict[str, Any]]:
        """Search for relevant SOPs"""
        try:
            # Create search query
            search_query = f"{intent} {domain}"
            if entities:
                entity_text = " ".join(entities.values())
                search_query += f" {entity_text}"
            
            # Use SOP manager for similarity search
            results = await self.sop_manager.search_similar_sops(
                query=search_query,
                domain=domain,
                intent=intent,
                top_k=self.max_sops * 2  # Get more candidates for filtering
            )
            
            # Add urgency-based scoring boost
            for result in results:
                if urgency_level == "critical":
                    result["similarity_score"] *= 1.2
                elif urgency_level == "high":
                    result["similarity_score"] *= 1.1
            
            # Sort by similarity score
            results.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            
            return results
