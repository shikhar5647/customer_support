"""
Metrics calculator for evaluating customer response quality
"""

import re
import time
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

class MetricsCalculator:
    """Calculate various metrics for response quality assessment"""
    
    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []
        self.response_times: List[float] = []
        self.confidence_scores: List[float] = []
        self.escalation_rates: List[bool] = []
        self.tool_usage: Dict[str, int] = defaultdict(int)
        self.sop_compliance: List[bool] = []
        
    def calculate_response_quality(
        self, 
        customer_message: str, 
        agent_response: str,
        expected_response: Optional[str] = None,
        response_time: float = 0.0,
        confidence_score: float = 0.0,
        tools_used: List[str] = None,
        sop_followed: bool = True,
        escalation_required: bool = False
    ) -> Dict[str, Any]:
        """Calculate comprehensive response quality metrics"""
        
        metrics = {
            "timestamp": datetime.now(),
            "customer_message": customer_message,
            "agent_response": agent_response,
            "response_time": response_time,
            "confidence_score": confidence_score,
            "escalation_required": escalation_required,
            "sop_compliance": sop_followed,
            "tools_used": tools_used or [],
            "metrics": {}
        }
        
        # Calculate various quality metrics
        metrics["metrics"]["readability"] = self._calculate_readability(agent_response)
        metrics["metrics"]["relevance"] = self._calculate_relevance(customer_message, agent_response)
        metrics["metrics"]["completeness"] = self._calculate_completeness(agent_response)
        metrics["metrics"]["professionalism"] = self._calculate_professionalism(agent_response)
        metrics["metrics"]["actionability"] = self._calculate_actionability(agent_response)
        
        if expected_response:
            metrics["metrics"]["accuracy"] = self._calculate_accuracy(agent_response, expected_response)
        
        # Calculate overall quality score
        quality_score = self._calculate_overall_quality(metrics["metrics"])
        metrics["metrics"]["overall_quality"] = quality_score
        
        # Store metrics
        self.metrics_history.append(metrics)
        self.response_times.append(response_time)
        self.confidence_scores.append(confidence_score)
        self.escalation_rates.append(escalation_required)
        self.sop_compliance.append(sop_followed)
        
        # Update tool usage
        for tool in tools_used or []:
            self.tool_usage[tool] += 1
        
        return metrics
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score using Flesch Reading Ease"""
        try:
            sentences = len(re.split(r'[.!?]+', text))
            words = len(text.split())
            syllables = self._count_syllables(text)
            
            if words == 0 or sentences == 0:
                return 0.0
            
            # Flesch Reading Ease formula
            flesch_score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
            
            # Normalize to 0-1 scale
            return max(0.0, min(1.0, flesch_score / 100.0))
            
        except Exception:
            return 0.5
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (simplified approach)"""
        text = text.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
        
        return max(1, count)
    
    def _calculate_relevance(self, customer_message: str, agent_response: str) -> float:
        """Calculate relevance between customer message and agent response"""
        try:
            # Simple keyword matching approach
            customer_words = set(re.findall(r'\b\w+\b', customer_message.lower()))
            response_words = set(re.findall(r'\b\w+\b', agent_response.lower()))
            
            if not customer_words:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(customer_words.intersection(response_words))
            union = len(customer_words.union(response_words))
            
            if union == 0:
                return 0.0
            
            return intersection / union
            
        except Exception:
            return 0.5
    
    def _calculate_completeness(self, response: str) -> float:
        """Calculate completeness of the response"""
        try:
            # Check for various completeness indicators
            indicators = {
                "greeting": bool(re.search(r'\b(hi|hello|good|thank)\b', response.lower())),
                "acknowledgment": bool(re.search(r'\b(understand|see|hear)\b', response.lower())),
                "solution": bool(re.search(r'\b(here|solution|answer|help)\b', response.lower())),
                "next_steps": bool(re.search(r'\b(next|step|process|procedure)\b', response.lower())),
                "closing": bool(re.search(r'\b(anything|else|help|thank)\b', response.lower()))
            }
            
            # Calculate completeness score
            score = sum(indicators.values()) / len(indicators)
            return score
            
        except Exception:
            return 0.5
    
    def _calculate_professionalism(self, response: str) -> float:
        """Calculate professionalism score"""
        try:
            # Check for professional language indicators
            professional_indicators = [
                r'\b(professional|polite|courteous)\b',
                r'\b(please|thank you|appreciate)\b',
                r'\b(understand|clarify|confirm)\b',
                r'\b(assist|help|support)\b'
            ]
            
            # Check for unprofessional language
            unprofessional_indicators = [
                r'\b(slang|casual|informal)\b',
                r'\b(angry|frustrated|annoyed)\b',
                r'\b(impatient|rude|disrespectful)\b'
            ]
            
            professional_score = sum(bool(re.search(pattern, response.lower())) for pattern in professional_indicators)
            unprofessional_score = sum(bool(re.search(pattern, response.lower())) for pattern in unprofessional_indicators)
            
            # Normalize score
            total_indicators = len(professional_indicators) + len(unprofessional_indicators)
            if total_indicators == 0:
                return 0.5
            
            score = (professional_score - unprofessional_score) / total_indicators
            return max(0.0, min(1.0, (score + 1) / 2))  # Normalize to 0-1
            
        except Exception:
            return 0.5
    
    def _calculate_actionability(self, response: str) -> float:
        """Calculate how actionable the response is"""
        try:
            # Check for actionable elements
            action_indicators = [
                r'\b(can|will|able to)\b',
                r'\b(provide|give|send)\b',
                r'\b(check|verify|confirm)\b',
                r'\b(process|handle|resolve)\b',
                r'\b(contact|call|email)\b',
                r'\b(follow|next step|procedure)\b'
            ]
            
            action_count = sum(bool(re.search(pattern, response.lower())) for pattern in action_indicators)
            
            # Normalize to 0-1 scale
            max_actions = len(action_indicators)
            return min(1.0, action_count / max_actions) if max_actions > 0 else 0.0
            
        except Exception:
            return 0.5
    
    def _calculate_accuracy(self, actual_response: str, expected_response: str) -> float:
        """Calculate accuracy compared to expected response"""
        try:
            # Simple text similarity using word overlap
            actual_words = set(re.findall(r'\b\w+\b', actual_response.lower()))
            expected_words = set(re.findall(r'\b\w+\b', expected_response.lower()))
            
            if not expected_words:
                return 0.0
            
            intersection = len(actual_words.intersection(expected_words))
            return intersection / len(expected_words)
            
        except Exception:
            return 0.5
    
    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from individual metrics"""
        try:
            # Weighted average of all metrics
            weights = {
                "readability": 0.15,
                "relevance": 0.25,
                "completeness": 0.20,
                "professionalism": 0.20,
                "actionability": 0.20
            }
            
            # Only include metrics that exist
            available_metrics = {k: v for k, v in metrics.items() if k in weights and v is not None}
            
            if not available_metrics:
                return 0.0
            
            total_weight = sum(weights[k] for k in available_metrics.keys())
            weighted_sum = sum(metrics[k] * weights[k] for k in available_metrics.keys())
            
            return weighted_sum / total_weight if total_weight > 0 else 0.0
            
        except Exception:
            return 0.5
    
    def get_performance_summary(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get performance summary for a time window"""
        try:
            if time_window:
                cutoff_time = datetime.now() - time_window
                recent_metrics = [m for m in self.metrics_history if m["timestamp"] >= cutoff_time]
            else:
                recent_metrics = self.metrics_history
            
            if not recent_metrics:
                return {}
            
            # Calculate summary statistics
            summary = {
                "total_responses": len(recent_metrics),
                "average_response_time": np.mean([m["response_time"] for m in recent_metrics]),
                "average_confidence": np.mean([m["confidence_score"] for m in recent_metrics]),
                "escalation_rate": np.mean([m["escalation_required"] for m in recent_metrics]),
                "sop_compliance_rate": np.mean([m["sop_compliance"] for m in recent_metrics]),
                "average_quality_score": np.mean([m["metrics"].get("overall_quality", 0) for m in recent_metrics]),
                "tool_usage": dict(self.tool_usage),
                "quality_distribution": self._get_quality_distribution(recent_metrics)
            }
            
            return summary
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_quality_distribution(self, metrics: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get distribution of quality scores"""
        try:
            quality_scores = [m["metrics"].get("overall_quality", 0) for m in metrics]
            
            distribution = {
                "excellent": len([s for s in quality_scores if s >= 0.9]),
                "good": len([s for s in quality_scores if 0.7 <= s < 0.9]),
                "fair": len([s for s in quality_scores if 0.5 <= s < 0.7]),
                "poor": len([s for s in quality_scores if s < 0.5])
            }
            
            return distribution
            
        except Exception:
            return {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
    
    def get_trend_analysis(self, days: int = 7) -> Dict[str, Any]:
        """Get trend analysis over specified days"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            daily_metrics = defaultdict(list)
            
            for metric in self.metrics_history:
                if start_date <= metric["timestamp"] <= end_date:
                    date_key = metric["timestamp"].date()
                    daily_metrics[date_key].append(metric)
            
            trends = {}
            for date, metrics in sorted(daily_metrics.items()):
                if metrics:
                    trends[str(date)] = {
                        "count": len(metrics),
                        "avg_quality": np.mean([m["metrics"].get("overall_quality", 0) for m in metrics]),
                        "avg_confidence": np.mean([m["confidence_score"] for m in metrics]),
                        "escalation_rate": np.mean([m["escalation_required"] for m in metrics])
                    }
            
            return trends
            
        except Exception as e:
            return {"error": str(e)}
    
    def export_metrics(self, file_path: str = "metrics_export.json"):
        """Export metrics to JSON file"""
        try:
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.metrics_history, f, indent=2, default=str, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error exporting metrics: {e}")
            return False
    
    def reset_metrics(self):
        """Reset all metrics"""
        self.metrics_history.clear()
        self.response_times.clear()
        self.confidence_scores.clear()
        self.escalation_rates.clear()
        self.tool_usage.clear()
        self.sop_compliance.clear()
