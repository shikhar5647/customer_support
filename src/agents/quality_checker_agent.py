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
    
class QualityCheckerAgent(BaseAgent):
    """Agent for validating response quality and compliance"""
    
    def __init__(
        self,
        gemini_api_key: str,
        guardrails_config: Optional[Dict[str, Any]] = None,
        model_name: str = "gemini-1.5-flash",
        **kwargs
    ):
        super().__init__(name="QualityChecker", **kwargs)
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key,
            temperature=0.1
        )
        self.guardrails_config = guardrails_config or {}
        self.output_parser = JsonOutputParser(pydantic_object=QualityCheckOutput)
        
        # Quality thresholds
        self.min_quality_score = self.guardrails_config.get("min_quality_score", 0.7)
        self.pii_patterns = self._load_pii_patterns()
        self.toxic_keywords = self._load_toxic_keywords()
    
    def get_schema(self) -> Type[BaseModel]:
        return QualityCheckInput
    
    async def execute(self, state: AgentState) -> AgentState:
        """Execute quality checking"""
        try:
            # Extract information from state
            response_text = state.context.get("response_generation", {}).get("response_text", "")
            customer_message = state.context.get("customer_message", "")
            intent_info = state.context.get("intent_classification", {})
            sop_info = state.context.get("sop_retrieval", {})
            tool_info = state.context.get("tool_orchestration", {})
            
            if not response_text:
                raise ValueError("No response text provided for quality checking")
            
            # Perform comprehensive quality checks
            quality_results = await self._perform_quality_checks(
                response_text=response_text,
                customer_message=customer_message,
                intent_info=intent_info,
                sop_snippets=sop_info.get("sop_snippets", []),
                tool_results=tool_info.get("tool_results", {})
            )
            
            # Update state with quality check results
            state.context["quality_check"] = quality_results
            state.confidence = quality_results.get("overall_score", 0.0)
            
            # Flag for human review if quality is low
            if quality_results.get("overall_score", 0.0) < self.min_quality_score:
                state.context["requires_human_review"] = True
                state.context["review_reason"] = f"Quality score ({quality_results.get('overall_score', 0.0):.2f}) below threshold ({self.min_quality_score})"
            
            # Flag safety issues
            if quality_results.get("safety_issues") or quality_results.get("toxicity_detected") or quality_results.get("contains_pii"):
                state.context["safety_flagged"] = True
                state.context["safety_reasons"] = quality_results.get("safety_issues", [])
            
            self.logger.info(f"Quality check completed. Overall score: {quality_results.get('overall_score', 0.0):.2f}")
            
        except Exception as e:
            self.logger.error(f"Quality check failed: {str(e)}")
            state.error = f"Quality check error: {str(e)}"
            state.confidence = 0.0
            
            # Set default quality check results
            state.context["quality_check"] = {
                "overall_score": 0.0,
                "sop_compliance": False,
                "accuracy_score": 0.0,
                "relevance_score": 0.0,
                "professionalism_score": 0.0,
                "completeness_score": 0.0,
                "safety_issues": ["Quality check system error"],
                "improvement_suggestions": ["System requires maintenance"],
                "contains_pii": False,
                "toxicity_detected": False,
                "hallucination_risk": "high"
            }
        
        return state
    
    async def _perform_quality_checks(
        self,
        response_text: str,
        customer_message: str,
        intent_info: Dict[str, Any],
        sop_snippets: List[str],
        tool_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive quality checks"""
        
        # 1. Basic safety checks
        pii_detected = self._check_pii(response_text)
        toxicity_detected = self._check_toxicity(response_text)
        safety_issues = []
        
        if pii_detected:
            safety_issues.append("Potential PII detected in response")
        if toxicity_detected:
            safety_issues.append("Toxic content detected")
        
        # 2. LLM-based quality assessment
        llm_assessment = await self._llm_quality_assessment(
            response_text, customer_message, intent_info, sop_snippets, tool_results
        )
        
        # 3. SOP compliance check
        sop_compliance = self._check_sop_compliance(response_text, sop_snippets)
        
        # 4. Hallucination risk assessment
        hallucination_risk = self._assess_hallucination_risk(response_text, tool_results)
        
        # 5. Calculate overall score
        scores = {
            "accuracy_score": llm_assessment.get("accuracy_score", 0.0),
            "relevance_score": llm_assessment.get("relevance_score", 0.0),
            "professionalism_score": llm_assessment.get("professionalism_score", 0.0),
            "completeness_score": llm_assessment.get("completeness_score", 0.0)
        }
        
        # Apply penalties for safety issues
        overall_score = sum(scores.values()) / len(scores)
        if safety_issues:
            overall_score *= 0.5  # Heavy penalty for safety issues
        if not sop_compliance:
            overall_score *= 0.8  # Penalty for SOP non-compliance
        
        return {
            "overall_score": min(overall_score, 1.0),
            "sop_compliance": sop_compliance,
            "accuracy_score": scores["accuracy_score"],
            "relevance_score": scores["relevance_score"],
            "professionalism_score": scores["professionalism_score"],
            "completeness_score": scores["completeness_score"],
            "safety_issues": safety_issues,
            "improvement_suggestions": llm_assessment.get("improvement_suggestions", []),
            "contains_pii": pii_detected,
            "toxicity_detected": toxicity_detected,
            "hallucination_risk": hallucination_risk
        }
    
    async def _llm_quality_assessment(
        self,
        response_text: str,
        customer_message: str,
        intent_info: Dict[str, Any],
        sop_snippets: List[str],
        tool_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use LLM for quality assessment"""
        try:
            system_prompt = self._create_quality_assessment_prompt()
            human_prompt = self._create_quality_human_prompt(
                response_text, customer_message, intent_info, sop_snippets, tool_results
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Parse response as JSON
            try:
                assessment = json.loads(response.content)
                return assessment
            except json.JSONDecodeError:
                # Fallback parsing
                return self._parse_quality_assessment_fallback(response.content)
            
        except Exception as e:
            self.logger.error(f"LLM quality assessment failed: {str(e)}")
            return {
                "accuracy_score": 0.5,
                "relevance_score": 0.5,
                "professionalism_score": 0.5,
                "completeness_score": 0.5,
                "improvement_suggestions": ["Quality assessment system error"]
            }
    
    def _create_quality_assessment_prompt(self) -> str:
        """Create system prompt for quality assessment"""
        return """You are a quality assessment expert for customer service responses.

Evaluate the generated response on these dimensions:
1. **Accuracy** (0.0-1.0): Is the information provided correct and factual?
2. **Relevance** (0.0-1.0): Does the response address the customer's specific request?
3. **Professionalism** (0.0-1.0): Is the tone appropriate, polite, and professional?
4. **Completeness** (0.0-1.0): Does the response fully address the customer's needs?

Also provide improvement suggestions if the scores are below 0.8.

Respond in JSON format:
{
  "accuracy_score": 0.0-1.0,
  "relevance_score": 0.0-1.0,
  "professionalism_score": 0.0-1.0,
  "completeness_score": 0.0-1.0,
  "improvement_suggestions": ["suggestion1", "suggestion2"]
}"""
    
    def _create_quality_human_prompt(
        self,
        response_text: str,
        customer_message: str,
        intent_info: Dict[str, Any],
        sop_snippets: List[str],
        tool_results: Dict[str, Any]
    ) -> str:
        """Create human prompt for quality assessment"""
        prompt_parts = [
            f"Customer Message: {customer_message}",
            f"Generated Response: {response_text}"
        ]
        
        if intent_info:
            intent = intent_info.get("primary_intent", "unknown")
            sentiment = intent_info.get("sentiment", "neutral")
            prompt_parts.append(f"Customer Intent: {intent} | Sentiment: {sentiment}")
        
        if sop_snippets:
            sop_text = "\n".join(sop_snippets[:2])  # First 2 SOPs
            prompt_parts.append(f"Relevant SOPs:\n{sop_text}")
        
        if tool_results:
            tools_info = [f"{tool}: {result.get('status', 'unknown')}" for tool, result in tool_results.items()]
            prompt_parts.append(f"Tools Used: {', '.join(tools_info)}")
        
        return "\n\n".join(prompt_parts)
    
    def _parse_quality_assessment_fallback(self, content: str) -> Dict[str, Any]:
        """Fallback parsing for quality assessment"""
        # Extract scores using regex patterns
        accuracy_match = re.search(r'"accuracy_score":\s*([\d.]+)', content)
        relevance_match = re.search(r'"relevance_score":\s*([\d.]+)', content)
        professionalism_match = re.search(r'"professionalism_score":\s*([\d.]+)', content)
        completeness_match = re.search(r'"completeness_score":\s*([\d.]+)', content)
        
        return {
            "accuracy_score": float(accuracy_match.group(1)) if accuracy_match else 0.5,
            "relevance_score": float(relevance_match.group(1)) if relevance_match else 0.5,
            "professionalism_score": float(professionalism_match.group(1)) if professionalism_match else 0.5,
            "completeness_score": float(completeness_match.group(1)) if completeness_match else 0.5,
            "improvement_suggestions": ["Unable to parse detailed suggestions due to format error"]
        }
    
    def _check_pii(self, text: str) -> bool:
        """Check for potential PII in text"""
        for pattern in self.pii_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _check_toxicity(self, text: str) -> bool:
        """Check for toxic content"""
        text_lower = text.lower()
        for keyword in self.toxic_keywords:
            if keyword in text_lower:
                return True
        return False
    
    def _check_sop_compliance(self, response_text: str, sop_snippets: List[str]) -> bool:
        """Check if response follows SOPs"""
        if not sop_snippets:
            return True  # No SOPs to validate against
        
        # Simple compliance check - look for key phrases from SOPs in response
        response_lower = response_text.lower()
        
        # Check if response mentions key SOP elements
        compliance_indicators = [
            "policy", "procedure", "guidelines", "process", "steps",
            "please", "thank you", "apologize", "understand"
        ]
        
        compliance_score = 0
        for indicator in compliance_indicators:
            if indicator in response_lower:
                compliance_score += 1
        
        # If response contains at least 3 compliance indicators, consider it compliant
        return compliance_score >= 3
    
    def _assess_hallucination_risk(self, response_text: str, tool_results: Dict[str, Any]) -> str:
        """Assess risk of hallucination in response"""
        # High risk indicators
        high_risk_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',  # Specific dates
            r'\$[\d,]+\.?\d*',  # Specific dollar amounts
            r'order #?\w+',  # Order numbers not from tools
            r'account #?\w+',  # Account numbers not from tools
        ]
        
        # Check if response contains specific details not backed by tools
        has_specific_details = any(re.search(pattern, response_text, re.IGNORECASE) for pattern in high_risk_patterns)
        
        # Check if tool results are available to back claims
        has_tool_backing = bool(tool_results and any(
            result.get("status") == "success" for result in tool_results.values()
            if isinstance(result, dict)
        ))
        
        if has_specific_details and not has_tool_backing:
            return "high"
        elif has_specific_details and has_tool_backing:
            return "low"
        else:
            return "medium"
    
    def _load_pii_patterns(self) -> List[str]:
        """Load PII detection patterns"""
        return [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN pattern
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card pattern
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email pattern
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone number pattern
            r'\b\d{1,5}\s+([A-Za-z]+\s+){1,3}(St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Ct|Court|Pl|Place)\b'  # Address pattern
        ]
    
    def _load_toxic_keywords(self) -> List[str]:
        """Load toxic keywords list"""
        return [
            "hate", "stupid", "idiot", "moron", "dumb", "pathetic",
            "worthless", "useless", "incompetent", "terrible", "awful",
            # Add more as needed, but keep it professional
        ]
        except Exception as e:
            self.logger.error(f"SOP search failed: {str(e)}")
            return []