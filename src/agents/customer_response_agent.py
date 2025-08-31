from typing import Dict, Any, List, Type, Optional, Union
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
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

class CustomerResponseAgent(BaseAgent):
    """
    Main orchestrator agent that coordinates the entire customer response workflow
    """
    
    def __init__(
        self,
        gemini_api_key: str,
        sop_manager=None,
        tool_registry=None,
        guardrails_config=None,
        model_name: str = "gemini-1.5-pro",
        min_confidence_threshold: float = 0.7,
        **kwargs
    ):
        super().__init__(name="CustomerResponseOrchestrator", **kwargs)
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key,
            temperature=0.3
        )
        
        # Initialize sub-agents
        self.intent_classifier = IntentClassifierAgent(gemini_api_key=gemini_api_key)
        self.sop_retrieval = SOPRetrievalAgent(sop_manager=sop_manager)
        self.tool_orchestrator = ToolOrchestratorAgent(
            gemini_api_key=gemini_api_key, 
            tool_registry=tool_registry
        )
        self.quality_checker = QualityCheckerAgent(
            gemini_api_key=gemini_api_key,
            guardrails_config=guardrails_config
        )
        
        self.min_confidence_threshold = min_confidence_threshold
        self.output_parser = JsonOutputParser(pydantic_object=CustomerResponseOutput)
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
    
    def get_schema(self) -> Type[BaseModel]:
        return CustomerResponseInput
    
    async def execute(self, state: Union[AgentState, Dict[str, Any]]) -> AgentState:
        """Execute the complete customer response workflow"""
        try:
            # Convert dictionary to AgentState if needed
            if isinstance(state, dict):
                agent_state = AgentState(context=state)
            else:
                agent_state = state
            
            # Initialize workflow state
            workflow_state = self._initialize_workflow_state(agent_state)
            
            # Run the LangGraph workflow
            final_state = await self.workflow.ainvoke(workflow_state)
            
            # Extract final response
            response_output = final_state.get("final_response", {})
            
            # Update main state
            agent_state.context.update(final_state)
            agent_state.context["customer_response"] = response_output
            agent_state.confidence = response_output.get("confidence_score", 0.0)
            
            # Check if escalation is needed
            if response_output.get("escalation_required", False):
                agent_state.context["escalation_required"] = True
                agent_state.context["escalation_reason"] = response_output.get("escalation_reason", "Low confidence response")
            
            self.logger.info(f"Customer response generated successfully. Confidence: {agent_state.confidence}")
            
        except Exception as e:
            self.logger.error(f"Customer response workflow failed: {str(e)}")
            agent_state.error = f"Workflow error: {str(e)}"
            agent_state.confidence = 0.0
        
        return agent_state
    
    def _initialize_workflow_state(self, state: Union[AgentState, Dict[str, Any]]) -> Dict[str, Any]:
        """Initialize state for LangGraph workflow"""
        # Handle both AgentState and dict inputs
        if isinstance(state, AgentState):
            context = state.context
        else:
            context = state
        
        return {
            "customer_message": context.get("customer_message", ""),
            "conversation_history": context.get("conversation_history", []),
            "domain": context.get("domain", "general"),
            "customer_id": context.get("customer_id"),
            "session_id": context.get("session_id"),
            "context": context.get("context", {}),
            "workflow_step": "intent_classification",
            "intent_classification": {},
            "sop_retrieval": {},
            "tool_orchestration": {},
            "response_generation": {},
            "quality_check": {},
            "final_response": {},
            "errors": []
        }
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Define workflow nodes
        async def classify_intent_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Intent classification node"""
            agent_state = AgentState(context=state)
            result_state = await self.intent_classifier.run_with_retry(agent_state)
            
            state.update(result_state.context)
            if result_state.error:
                state["errors"].append(f"Intent classification: {result_state.error}")
            
            return state
        
        async def retrieve_sops_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """SOP retrieval node"""
            agent_state = AgentState(context=state)
            result_state = await self.sop_retrieval.run_with_retry(agent_state)
            
            state.update(result_state.context)
            if result_state.error:
                state["errors"].append(f"SOP retrieval: {result_state.error}")
            
            return state
        
        async def orchestrate_tools_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Tool orchestration node"""
            agent_state = AgentState(context=state)
            result_state = await self.tool_orchestrator.run_with_retry(agent_state)
            
            state.update(result_state.context)
            if result_state.error:
                state["errors"].append(f"Tool orchestration: {result_state.error}")
            
            return state
        
        async def generate_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Response generation node"""
            try:
                response = await self._generate_response(state)
                state["response_generation"] = response
            except Exception as e:
                state["errors"].append(f"Response generation: {str(e)}")
                state["response_generation"] = {
                    "response_text": "I apologize, but I'm having trouble processing your request right now. Please try again or contact support for assistance.",
                    "confidence_score": 0.0
                }
            
            return state
        
        async def quality_check_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Quality check node"""
            agent_state = AgentState(context=state)
            result_state = await self.quality_checker.run_with_retry(agent_state)
            
            state.update(result_state.context)
            if result_state.error:
                state["errors"].append(f"Quality check: {result_state.error}")
            
            return state
        
        async def finalize_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
            """Finalize response node"""
            final_response = self._create_final_response(state)
            state["final_response"] = final_response
            return state
        
        # Build the graph
        workflow = StateGraph(dict)
        
        # Add nodes
        workflow.add_node("classify_intent", classify_intent_node)
        workflow.add_node("retrieve_sops", retrieve_sops_node)
        workflow.add_node("orchestrate_tools", orchestrate_tools_node)
        workflow.add_node("generate_response", generate_response_node)
        workflow.add_node("quality_check", quality_check_node)
        workflow.add_node("finalize_response", finalize_response_node)
        
        # Add edges
        workflow.set_entry_point("classify_intent")
        workflow.add_edge("classify_intent", "retrieve_sops")
        workflow.add_edge("retrieve_sops", "orchestrate_tools")
        workflow.add_edge("orchestrate_tools", "generate_response")
        workflow.add_edge("generate_response", "quality_check")
        workflow.add_edge("quality_check", "finalize_response")
        workflow.add_edge("finalize_response", END)
        
        return workflow.compile()
    
    async def _generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using all collected information"""
        try:
            # Collect information from previous steps
            customer_message = state.get("customer_message", "")
            intent_info = state.get("intent_classification", {})
            sop_info = state.get("sop_retrieval", {})
            tool_info = state.get("tool_orchestration", {})
            conversation_history = state.get("conversation_history", [])
            
            # Create response generation prompt
            system_prompt = self._create_response_system_prompt()
            human_prompt = self._create_response_human_prompt(
                customer_message, intent_info, sop_info, tool_info, conversation_history
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Calculate confidence based on available information
            confidence = self._calculate_response_confidence(intent_info, sop_info, tool_info)
            
            return {
                "response_text": response.content.strip(),
                "confidence_score": confidence,
                "generation_method": "llm_with_context"
            }
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {str(e)}")
            return {
                "response_text": "I apologize, but I'm experiencing technical difficulties. Please contact our support team for immediate assistance.",
                "confidence_score": 0.0,
                "generation_method": "fallback"
            }
    
    def _create_response_system_prompt(self) -> str:
        """Create system prompt for response generation"""
        return """You are a professional customer service representative AI assistant.

Your role is to provide helpful, accurate, and empathetic responses to customers based on:
1. Customer intent and sentiment
2. Relevant Standard Operating Procedures (SOPs)  
3. Real-time data from system tools/APIs
4. Conversation history and context

Guidelines:
- Be professional, helpful, and empathetic
- Follow SOPs when provided
- Use tool/API data when available to provide specific information
- Acknowledge customer concerns and emotions
- Provide clear, actionable information
- If you cannot fully address the request, explain what you can do and offer next steps
- Keep responses concise but comprehensive
- Maintain consistency with previous conversation turns

Important: Generate ONLY the response text - no JSON, no metadata, just the customer-facing message."""
    
    def _create_response_human_prompt(
        self,
        customer_message: str,
        intent_info: Dict[str, Any],
        sop_info: Dict[str, Any], 
        tool_info: Dict[str, Any],
        conversation_history: List[Dict[str, str]]
    ) -> str:
        """Create human prompt with all context"""
        
        prompt_parts = [f"Customer Message: {customer_message}"]
        
        # Add intent information
        if intent_info:
            intent = intent_info.get("primary_intent", "unknown")
            sentiment = intent_info.get("sentiment", "neutral")
            urgency = intent_info.get("urgency_level", "medium")
            prompt_parts.append(f"\nIntent: {intent} | Sentiment: {sentiment} | Urgency: {urgency}")
        
        # Add conversation history
        if conversation_history:
            history_text = "\n".join([
                f"{'Customer' if msg.get('role') == 'user' else 'Agent'}: {msg.get('content', '')}"
                for msg in conversation_history[-3:]  # Last 3 exchanges
            ])
            prompt_parts.append(f"\nRecent Conversation:\n{history_text}")
        
        # Add SOP information
        if sop_info and sop_info.get("sop_snippets"):
            sop_text = "\n".join(sop_info["sop_snippets"])
            prompt_parts.append(f"\nRelevant Procedures:\n{sop_text}")
        
        # Add tool results
        if tool_info and tool_info.get("tool_results"):
            results_text = []
            for tool_name, result in tool_info["tool_results"].items():
                if isinstance(result, dict) and result.get("status") == "success":
                    results_text.append(f"{tool_name}: {result.get('data', 'No data')}")
            
            if results_text:
                prompt_parts.append(f"\nSystem Information:\n" + "\n".join(results_text))
        
        return "\n".join(prompt_parts)
    
    def _calculate_response_confidence(
        self,
        intent_info: Dict[str, Any],
        sop_info: Dict[str, Any],
        tool_info: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for response"""
        base_confidence = 0.5
        
        # Intent classification confidence
        intent_confidence = intent_info.get("confidence_score", 0.0)
        base_confidence += intent_confidence * 0.3
        
        # SOP availability boost
        if sop_info.get("sop_snippets"):
            sop_confidence = max(sop_info.get("confidence_scores", [0.0]))
            base_confidence += sop_confidence * 0.2
        
        # Tool results boost
        if tool_info.get("tool_results"):
            successful_tools = sum(1 for result in tool_info["tool_results"].values() 
                                 if isinstance(result, dict) and result.get("status") == "success")
            total_tools = len(tool_info["tool_results"])
            if total_tools > 0:
                tool_success_rate = successful_tools / total_tools
                base_confidence += tool_success_rate * 0.2
        
        return min(base_confidence, 1.0)
    
    def _create_final_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create the final response output"""
        response_gen = state.get("response_generation", {})
        intent_info = state.get("intent_classification", {})
        tool_info = state.get("tool_orchestration", {})
        quality_info = state.get("quality_check", {})
        
        confidence = response_gen.get("confidence_score", 0.0)
        escalation_required = confidence < self.min_confidence_threshold
        
        # Determine escalation reason
        escalation_reason = None
        if escalation_required:
            if state.get("errors"):
                escalation_reason = f"Technical errors occurred: {'; '.join(state['errors'])}"
            else:
                escalation_reason = f"Low confidence response ({confidence:.2f} < {self.min_confidence_threshold})"
        
        return {
            "response_text": response_gen.get("response_text", "I apologize, but I cannot process your request at this time."),
            "intent_classification": intent_info,
            "sop_compliance": quality_info.get("sop_compliance", True),
            "tools_used": tool_info.get("required_tools", []),
            "confidence_score": confidence,
            "escalation_required": escalation_required,
            "escalation_reason": escalation_reason,
            "response_metadata": {
                "workflow_errors": state.get("errors", []),
                "processing_steps": ["intent_classification", "sop_retrieval", "tool_orchestration", "response_generation", "quality_check"],
                "generation_method": response_gen.get("generation_method", "unknown")
            }
        }

