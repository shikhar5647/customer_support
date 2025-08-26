from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser
import re
import json
import logging


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


class QualityCheckerAgent:
    """LangGraph-compatible Quality Checker Agent"""

    def __init__(
        self,
        gemini_api_key: str,
        guardrails_config: Optional[Dict[str, Any]] = None,
        model_name: str = "gemini-1.5-flash",
    ):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key,
            temperature=0.1
        )
        self.guardrails_config = guardrails_config or {}
        self.output_parser = JsonOutputParser(pydantic_object=QualityCheckOutput)
        self.min_quality_score = self.guardrails_config.get("min_quality_score", 0.7)
        self.pii_patterns = self._load_pii_patterns()
        self.toxic_keywords = self._load_toxic_keywords()
        self.logger = logging.getLogger(__name__)

    async def __call__(self, inputs: QualityCheckInput) -> QualityCheckOutput:
        """
        LangGraph requires nodes to be callable with input -> output
        """

        try:
            # 1. Safety checks
            pii_detected = self._check_pii(inputs.response_text)
            toxicity_detected = self._check_toxicity(inputs.response_text)
            safety_issues = []
            if pii_detected:
                safety_issues.append("Potential PII detected in response")
            if toxicity_detected:
                safety_issues.append("Toxic content detected")

            # 2. LLM-based quality assessment
            llm_assessment = await self._llm_quality_assessment(
                inputs.response_text,
                inputs.customer_message,
                inputs.intent_classification,
                inputs.sop_snippets,
                inputs.tool_results
            )

            # 3. SOP compliance
            sop_compliance = self._check_sop_compliance(
                inputs.response_text, inputs.sop_snippets
            )

            # 4. Hallucination risk
            hallucination_risk = self._assess_hallucination_risk(
                inputs.response_text, inputs.tool_results
            )

            # 5. Aggregate scores
            scores = {
                "accuracy_score": llm_assessment.get("accuracy_score", 0.0),
                "relevance_score": llm_assessment.get("relevance_score", 0.0),
                "professionalism_score": llm_assessment.get("professionalism_score", 0.0),
                "completeness_score": llm_assessment.get("completeness_score", 0.0),
            }

            overall_score = sum(scores.values()) / len(scores)
            if safety_issues:
                overall_score *= 0.5
            if not sop_compliance:
                overall_score *= 0.8

            return QualityCheckOutput(
                overall_score=min(overall_score, 1.0),
                sop_compliance=sop_compliance,
                accuracy_score=scores["accuracy_score"],
                relevance_score=scores["relevance_score"],
                professionalism_score=scores["professionalism_score"],
                completeness_score=scores["completeness_score"],
                safety_issues=safety_issues,
                improvement_suggestions=llm_assessment.get("improvement_suggestions", []),
                contains_pii=pii_detected,
                toxicity_detected=toxicity_detected,
                hallucination_risk=hallucination_risk,
            )

        except Exception as e:
            self.logger.error(f"Quality check failed: {str(e)}")
            return QualityCheckOutput(
                overall_score=0.0,
                sop_compliance=False,
                accuracy_score=0.0,
                relevance_score=0.0,
                professionalism_score=0.0,
                completeness_score=0.0,
                safety_issues=["Quality check system error"],
                improvement_suggestions=["System requires maintenance"],
                contains_pii=False,
                toxicity_detected=False,
                hallucination_risk="high"
            )

    # ---------------- Helper methods ---------------- #

    async def _llm_quality_assessment(
        self,
        response_text: str,
        customer_message: str,
        intent_info: Dict[str, Any],
        sop_snippets: List[str],
        tool_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            system_prompt = self._create_quality_assessment_prompt()
            human_prompt = self._create_quality_human_prompt(
                response_text, customer_message, intent_info, sop_snippets, tool_results
            )
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
            response = await self.llm.ainvoke(messages)

            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return self._parse_quality_assessment_fallback(response.content)

        except Exception as e:
            self.logger.error(f"LLM quality assessment failed: {str(e)}")
            return {
                "accuracy_score": 0.5,
                "relevance_score": 0.5,
                "professionalism_score": 0.5,
                "completeness_score": 0.5,
                "improvement_suggestions": ["Quality assessment system error"],
            }

    def _create_quality_assessment_prompt(self) -> str:
        return """You are a quality assessment expert for customer service responses.
Evaluate the generated response on these dimensions:
1. Accuracy (0.0-1.0)
2. Relevance (0.0-1.0)
3. Professionalism (0.0-1.0)
4. Completeness (0.0-1.0)
Also provide improvement suggestions if the scores are below 0.8.
Respond in JSON format:
{
  "accuracy_score": 0.0-1.0,
  "relevance_score": 0.0-1.0,
  "professionalism_score": 0.0-1.0,
  "completeness_score": 0.0-1.0,
  "improvement_suggestions": ["suggestion1"]
}"""

    def _create_quality_human_prompt(
        self, response_text, customer_message, intent_info, sop_snippets, tool_results
    ) -> str:
        parts = [f"Customer Message: {customer_message}", f"Generated Response: {response_text}"]
        if intent_info:
            parts.append(
                f"Customer Intent: {intent_info.get('primary_intent', 'unknown')} | Sentiment: {intent_info.get('sentiment', 'neutral')}"
            )
        if sop_snippets:
            parts.append(f"Relevant SOPs:\n{'\n'.join(sop_snippets[:2])}")
        if tool_results:
            tools_info = [f"{tool}: {result.get('status', 'unknown')}" for tool, result in tool_results.items()]
            parts.append(f"Tools Used: {', '.join(tools_info)}")
        return "\n\n".join(parts)

    def _parse_quality_assessment_fallback(self, content: str) -> Dict[str, Any]:
        accuracy = re.search(r'"accuracy_score":\s*([\d.]+)', content)
        relevance = re.search(r'"relevance_score":\s*([\d.]+)', content)
        professionalism = re.search(r'"professionalism_score":\s*([\d.]+)', content)
        completeness = re.search(r'"completeness_score":\s*([\d.]+)', content)
        return {
            "accuracy_score": float(accuracy.group(1)) if accuracy else 0.5,
            "relevance_score": float(relevance.group(1)) if relevance else 0.5,
            "professionalism_score": float(professionalism.group(1)) if professionalism else 0.5,
            "completeness_score": float(completeness.group(1)) if completeness else 0.5,
            "improvement_suggestions": ["Unable to parse detailed suggestions"],
        }

    def _check_pii(self, text: str) -> bool:
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in self.pii_patterns)

    def _check_toxicity(self, text: str) -> bool:
        return any(keyword in text.lower() for keyword in self.toxic_keywords)

    def _check_sop_compliance(self, response_text: str, sop_snippets: List[str]) -> bool:
        if not sop_snippets:
            return True
        indicators = ["policy", "procedure", "guidelines", "process", "steps", "please", "thank you", "apologize", "understand"]
        return sum(1 for ind in indicators if ind in response_text.lower()) >= 3

    def _assess_hallucination_risk(self, response_text: str, tool_results: Dict[str, Any]) -> str:
        high_risk_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\$[\d,]+\.?\d*',
            r'order #?\w+',
            r'account #?\w+',
        ]
        has_details = any(re.search(p, response_text, re.IGNORECASE) for p in high_risk_patterns)
        has_tool_backing = any(
            isinstance(result, dict) and result.get("status") == "success"
            for result in tool_results.values()
        )
        if has_details and not has_tool_backing:
            return "high"
        elif has_details and has_tool_backing:
            return "low"
        return "medium"

    def _load_pii_patterns(self) -> List[str]:
        return [
            r'\b\d{3}-\d{2}-\d{4}\b',
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\b\d{1,5}\s+([A-Za-z]+\s+){1,3}(St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Dr|Drive|Ln|Lane|Way|Ct|Court|Pl|Place)\b'
        ]

    def _load_toxic_keywords(self) -> List[str]:
        return ["hate", "stupid", "idiot", "moron", "dumb", "pathetic", "worthless", "useless", "incompetent", "terrible", "awful"]
