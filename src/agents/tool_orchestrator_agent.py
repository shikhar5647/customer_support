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

class ToolOrchestratorAgent(BaseAgent):
    """Agent for orchestrating external tool/API calls"""
    
    def __init__(
        self,
        gemini_api_key: str,
        tool_registry=None,  # Will be injected
        model_name: str = "gemini-1.5-flash",
        **kwargs
    ):
        super().__init__(name="ToolOrchestrator", **kwargs)
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key,
            temperature=0.1
        )
        self.tool_registry = tool_registry or {}
        self.output_parser = JsonOutputParser(pydantic_object=ToolOrchestrationOutput)
    
    def get_schema(self) -> Type[BaseModel]:
        return ToolOrchestrationInput

    async def run_with_retry(self, agent_state) -> Any:
        """Run tool orchestration with retry logic for compatibility with workflow"""
        try:
            # Run tool orchestration
            result_state = await self.execute(agent_state)
            return result_state
            
        except Exception as e:
            self.logger.error(f"Tool orchestration failed: {str(e)}")
            agent_state.error = f"Tool orchestration error: {str(e)}"
            return agent_state

    async def execute(self, state: AgentState) -> AgentState:
        """Execute tool orchestration"""
        try:
            # Extract parameters
            intent = state.context.get("primary_intent", "")
            entities = state.context.get("entities", {})
            customer_message = state.context.get("customer_message", "")
            domain = state.context.get("domain", "general")
            
            # Determine required tools
            tool_plan = await self._create_tool_execution_plan(
                intent, entities, customer_message, domain
            )
            
            if not tool_plan.get("required_tools"):
                # No tools required
                state.context["tool_orchestration"] = {
                    "required_tools": [],
                    "tool_results": {},
                    "success": True
                }
                state.confidence = 1.0
                return state
            
            # Execute tools
            tool_results = await self._execute_tools(tool_plan)
            
            # Update state
            orchestration_output = {
                "required_tools": tool_plan.get("required_tools", []),
                "tool_parameters": tool_plan.get("tool_parameters", {}),
                "execution_order": tool_plan.get("execution_order", []),
                "tool_results": tool_results,
                "success": len(tool_results) > 0
            }
            
            state.context["tool_orchestration"] = orchestration_output
            state.context["tool_results"] = tool_results
            state.confidence = 1.0 if orchestration_output["success"] else 0.5
            
            self.logger.info(f"Tool orchestration completed. Tools used: {tool_plan.get('required_tools', [])}")
            
        except Exception as e:
            self.logger.error(f"Tool orchestration failed: {str(e)}")
            state.error = f"Tool orchestration error: {str(e)}"
            state.confidence = 0.0
        
        return state
    
    async def _create_tool_execution_plan(
        self,
        intent: str,
        entities: Dict[str, str],
        customer_message: str,
        domain: str
    ) -> Dict[str, Any]:
        """Create execution plan for tools"""
        try:
            # Get available tools for domain
            available_tools = self._get_available_tools(domain)
            
            if not available_tools:
                return {"required_tools": [], "tool_parameters": {}, "execution_order": []}
            
            # Create prompt for tool selection
            system_prompt = self._create_tool_selection_prompt(domain, available_tools)
            human_prompt = f"""
Customer Intent: {intent}
Customer Message: {customer_message}
Extracted Entities: {entities}

Determine which tools are needed and their parameters.
"""
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse response - simplified version
            # In production, you'd want more sophisticated parsing
            tool_plan = self._parse_tool_plan(response.content, intent, entities)
            
            return tool_plan
            
        except Exception as e:
            self.logger.error(f"Tool plan creation failed: {str(e)}")
            return {"required_tools": [], "tool_parameters": {}, "execution_order": []}
    
    def _get_available_tools(self, domain: str) -> Dict[str, str]:
        """Get available tools for domain"""
        tools_by_domain = {
            "ecommerce": {
                "order_lookup": "Look up order status and details",
                "inventory_check": "Check product inventory",
                "return_status": "Check return/refund status",
                "shipping_tracker": "Track shipping information"
            },
            "telecom": {
                "bill_lookup": "Look up billing information",
                "plan_details": "Get plan details and usage",
                "network_status": "Check network status",
                "service_status": "Check service status"
            },
            "utilities": {
                "service_status": "Check service status",
                "billing_lookup": "Look up billing information",
                "outage_checker": "Check for service outages",
                "meter_reading": "Get meter readings"
            }
        }
        return tools_by_domain.get(domain, {})
    
    def _create_tool_selection_prompt(self, domain: str, available_tools: Dict[str, str]) -> str:
        """Create prompt for tool selection"""
        tools_description = "\n".join([f"- {tool}: {desc}" for tool, desc in available_tools.items()])
        
        return f"""You are a tool selection expert for {domain} customer service.

Available tools:
{tools_description}

Your task is to determine:
1. Which tools are needed to address the customer's request
2. What parameters each tool needs
3. The optimal execution order

Rules:
- Only select tools that are actually needed
- Extract parameters from customer message and entities
- Consider dependencies between tools
- If no tools are needed, return empty lists

Respond with a clear plan indicating required tools and their parameters."""
    
    def _parse_tool_plan(self, response_content: str, intent: str, entities: Dict[str, str]) -> Dict[str, Any]:
        """Parse LLM response into tool execution plan"""
        # Simplified parsing - in production, use structured output
        plan = {
            "required_tools": [],
            "tool_parameters": {},
            "execution_order": []
        }
        
        # Basic intent-to-tool mapping
        intent_tool_mapping = {
            "order_status": ["order_lookup"],
            "order_cancel": ["order_lookup"],
            "return_request": ["order_lookup", "return_status"],
            "bill_inquiry": ["bill_lookup"],
            "technical_support": ["network_status", "service_status"],
            "outage_report": ["outage_checker"],
            "service_request": ["service_status"]
        }
        
        if intent in intent_tool_mapping:
            plan["required_tools"] = intent_tool_mapping[intent]
            plan["execution_order"] = intent_tool_mapping[intent]
            
            # Extract parameters from entities
            for tool in plan["required_tools"]:
                plan["tool_parameters"][tool] = self._extract_tool_parameters(tool, entities)
        
        return plan
    
    def _extract_tool_parameters(self, tool: str, entities: Dict[str, str]) -> Dict[str, Any]:
        """Extract parameters for specific tools"""
        parameters = {}
        
        if tool == "order_lookup":
            parameters["order_id"] = entities.get("order_number", entities.get("order_id", ""))
            
        elif tool == "bill_lookup":
            parameters["account_id"] = entities.get("account_number", entities.get("account_id", ""))
            
        elif tool == "service_status":
            parameters["service_address"] = entities.get("address", "")
            parameters["account_id"] = entities.get("account_number", "")
        
        return parameters
    
    async def _execute_tools(self, tool_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the planned tools"""
        results = {}
        
        try:
            for tool_name in tool_plan.get("execution_order", []):
                if tool_name in self.tool_registry:
                    tool_instance = self.tool_registry[tool_name]
                    parameters = tool_plan.get("tool_parameters", {}).get(tool_name, {})
                    
                    self.logger.info(f"Executing tool: {tool_name} with parameters: {parameters}")
                    
                    # Execute tool
                    result = await tool_instance.execute(parameters)
                    results[tool_name] = result
                    
                else:
                    # Tool not available - create mock result
                    self.logger.warning(f"Tool {tool_name} not available, creating mock result")
                    results[tool_name] = {"status": "tool_unavailable", "message": f"Tool {tool_name} is not configured"}
                    
        except Exception as e:
            self.logger.error(f"Tool execution failed: {str(e)}")
            results["execution_error"] = str(e)
        
        return results