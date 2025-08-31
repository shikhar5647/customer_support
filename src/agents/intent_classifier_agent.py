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

class IntentClassifierAgent(BaseAgent):
    """Agent for classifying customer intents"""
    
    def __init__(
        self,
        gemini_api_key: str,
        model_name: str = "gemini-1.5-flash",
        **kwargs
    ):
        super().__init__(name="IntentClassifier", **kwargs)
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key,
            temperature=0.1
        )
        self.output_parser = JsonOutputParser(pydantic_object=IntentClassificationOutput)
        
        # Intent categories by domain
        self.intent_categories = {
            "ecommerce": [
                "order_status", "order_cancel", "order_modify", "return_request",
                "refund_inquiry", "product_inquiry", "shipping_issue", "payment_issue",
                "account_management", "complaint", "general_inquiry"
            ],
            "telecom": [
                "bill_inquiry", "payment_issue", "plan_change", "technical_support",
                "network_issue", "device_support", "service_activation", "service_cancellation",
                "account_management", "complaint", "general_inquiry"
            ],
            "utilities": [
                "service_request", "billing_inquiry", "payment_issue", "outage_report",
                "meter_reading", "service_connection", "service_disconnection",
                "account_management", "complaint", "general_inquiry"
            ]
        }
    
    def get_schema(self) -> Type[BaseModel]:
        return IntentClassificationInput

    async def run_with_retry(self, agent_state) -> Any:
        """Run intent classification with retry logic for compatibility with workflow"""
        try:
            # Extract inputs from agent state
            inputs = IntentClassificationInput(
                customer_message=agent_state.context.get("customer_message", ""),
                conversation_history=agent_state.context.get("conversation_history", []),
                domain=agent_state.context.get("domain", "general"),
                customer_id=agent_state.context.get("customer_id")
            )
            
            # Run intent classification
            result_state = await self.execute(agent_state)
            
            return result_state
            
        except Exception as e:
            self.logger.error(f"Intent classification failed: {str(e)}")
            agent_state.error = f"Intent classification error: {str(e)}"
            return agent_state

    async def execute(self, state: AgentState) -> AgentState:
        """Execute intent classification"""
        try:
            # Extract input from state
            customer_message = state.context.get("customer_message", "")
            conversation_history = state.context.get("conversation_history", [])
            domain = state.context.get("domain", "general")
            
            if not customer_message:
                raise ValueError("Customer message is required for intent classification")
            
            # Get domain-specific intents
            available_intents = self.intent_categories.get(domain, self.intent_categories["ecommerce"])
            
            # Create prompt
            system_prompt = self._create_system_prompt(domain, available_intents)
            human_prompt = self._create_human_prompt(customer_message, conversation_history)
            
            # Generate response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            parsed_output = self.output_parser.parse(response.content)
            
            # Update state
            state.context["intent_classification"] = parsed_output.dict()
            state.context["primary_intent"] = parsed_output.primary_intent
            state.context["urgency_level"] = parsed_output.urgency_level
            state.context["sentiment"] = parsed_output.sentiment
            state.confidence = parsed_output.confidence_score
            
            self.logger.info(f"Intent classified as: {parsed_output.primary_intent} (confidence: {parsed_output.confidence_score})")
            
        except Exception as e:
            self.logger.error(f"Intent classification failed: {str(e)}")
            state.error = f"Intent classification error: {str(e)}"
            state.confidence = 0.0
        
        return state
    
    def _create_system_prompt(self, domain: str, available_intents: List[str]) -> str:
        """Create system prompt for intent classification"""
        intents_str = ", ".join(available_intents)
        
        return f"""You are an expert intent classifier for a {domain} customer service system.

Your task is to analyze customer messages and classify their intent accurately.

Available intents for {domain}: {intents_str}

Guidelines:
1. Identify the PRIMARY intent (most important customer need)
2. Identify any SECONDARY intents (additional needs in the same message)
3. Extract relevant entities (order numbers, account IDs, product names, etc.)
4. Assess urgency level: critical (immediate attention), high (same day), medium (1-2 days), low (flexible)
5. Determine sentiment: positive, neutral, negative
6. Provide confidence score (0.0 to 1.0)

Output your response as valid JSON matching the required schema.
Be precise and consistent in your classifications."""
    
    def _create_human_prompt(self, customer_message: str, conversation_history: List[str]) -> str:
        """Create human prompt with customer message and context"""
        prompt = f"Customer Message: {customer_message}\n"
        
        if conversation_history:
            history_str = "\n".join([f"- {msg}" for msg in conversation_history[-3:]])  # Last 3 messages
            prompt += f"\nConversation Context:\n{history_str}\n"
        
        prompt += f"\n{self.output_parser.get_format_instructions()}"
        return prompt