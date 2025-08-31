"""
SOP Validation Module
"""
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError
import re
from datetime import datetime
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class ValidationResult(BaseModel):
    """Result of SOP validation"""
    sop_id: str
    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    validation_score: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class SOPValidator:
    """Validates SOP documents for quality and compliance"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Validation rules
        self.required_fields = ["sop_id", "title", "domain", "content"]
        self.min_content_length = 50
        self.max_content_length = 10000
        
        # Content quality patterns
        self.quality_patterns = {
            "actionable_verbs": [
                "check", "verify", "confirm", "ensure", "follow", "use", "apply",
                "contact", "notify", "escalate", "resolve", "complete", "submit"
            ],
            "clear_indicators": [
                "step", "procedure", "process", "guideline", "policy", "rule",
                "requirement", "standard", "protocol", "workflow"
            ],
            "contact_info_patterns": [
                r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # Phone numbers
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Emails
                r'\b(?:contact|call|email|reach|support)\b'  # Contact words
            ]
        }
    
    async def validate_sop(self, sop_data: Dict[str, Any]) -> ValidationResult:
        """Validate a single SOP document"""
        try:
            sop_id = sop_data.get("sop_id", "unknown")
            errors = []
            warnings = []
            validation_score = 0.0
            
            # Check required fields
            field_errors = self._validate_required_fields(sop_data)
            errors.extend(field_errors)
            
            # Validate content quality
            content_errors, content_warnings, content_score = self._validate_content_quality(sop_data)
            errors.extend(content_errors)
            warnings.extend(content_warnings)
            
            # Validate structure
            structure_errors, structure_warnings, structure_score = self._validate_structure(sop_data)
            errors.extend(structure_errors)
            warnings.extend(structure_warnings)
            
            # Validate metadata
            metadata_errors, metadata_warnings, metadata_score = self._validate_metadata(sop_data)
            errors.extend(metadata_errors)
            warnings.extend(metadata_warnings)
            
            # Calculate overall validation score
            validation_score = (content_score + structure_score + metadata_score) / 3
            
            # Determine if SOP is valid
            is_valid = len(errors) == 0
            
            return ValidationResult(
                sop_id=sop_id,
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                validation_score=validation_score,
                metadata={
                    "validation_timestamp": datetime.now().isoformat(),
                    "total_checks": 4,
                    "passed_checks": 4 - len([e for e in errors if "required" in e.lower()])
                }
            )
            
        except Exception as e:
            self.logger.error(f"SOP validation failed: {str(e)}")
            return ValidationResult(
                sop_id=sop_data.get("sop_id", "unknown"),
                is_valid=False,
                errors=[f"Validation system error: {str(e)}"],
                validation_score=0.0
            )
    
    def _validate_required_fields(self, sop_data: Dict[str, Any]) -> List[str]:
        """Validate that all required fields are present"""
        errors = []
        
        for field in self.required_fields:
            if field not in sop_data or not sop_data[field]:
                errors.append(f"Missing required field: {field}")
        
        return errors
    
    def _validate_content_quality(self, sop_data: Dict[str, Any]) -> Tuple[List[str], List[str], float]:
        """Validate content quality"""
        errors = []
        warnings = []
        score = 0.0
        
        content = sop_data.get("content", "")
        
        # Check content length
        if len(content) < self.min_content_length:
            errors.append(f"Content too short: {len(content)} characters (minimum: {self.min_content_length})")
            score -= 0.3
        elif len(content) > self.max_content_length:
            warnings.append(f"Content very long: {len(content)} characters (recommended: < {self.max_content_length})")
            score -= 0.1
        
        # Check for actionable content
        actionable_count = sum(1 for verb in self.quality_patterns["actionable_verbs"] 
                              if verb.lower() in content.lower())
        if actionable_count < 3:
            warnings.append("Content may lack actionable steps")
            score -= 0.2
        else:
            score += 0.2
        
        # Check for clear structure indicators
        clear_indicators = sum(1 for indicator in self.quality_patterns["clear_indicators"] 
                              if indicator.lower() in content.lower())
        if clear_indicators < 2:
            warnings.append("Content may lack clear structure indicators")
            score -= 0.1
        else:
            score += 0.1
        
        # Check for contact information
        has_contact_info = any(re.search(pattern, content, re.IGNORECASE) 
                              for pattern in self.quality_patterns["contact_info_patterns"])
        if not has_contact_info:
            warnings.append("No contact information found")
            score -= 0.1
        
        # Normalize score to 0-1 range
        score = max(0.0, min(1.0, score + 0.5))
        
        return errors, warnings, score
    
    def _validate_structure(self, sop_data: Dict[str, Any]) -> Tuple[List[str], List[str], float]:
        """Validate SOP structure"""
        errors = []
        warnings = []
        score = 0.0
        
        content = sop_data.get("content", "")
        
        # Check for sections/headers
        if "sections" in sop_data:
            sections = sop_data["sections"]
            if len(sections) < 2:
                warnings.append("SOP has few sections - consider adding more structure")
                score -= 0.1
            else:
                score += 0.2
        else:
            # Check for markdown-style headers
            header_count = len(re.findall(r'^#+\s+', content, re.MULTILINE))
            if header_count < 2:
                warnings.append("Consider adding more section headers for better structure")
                score -= 0.1
            else:
                score += 0.2
        
        # Check for numbered or bulleted lists
        list_patterns = [
            r'^\d+\.\s+',  # Numbered lists
            r'^[-*]\s+',   # Bullet points
            r'^\*\s+',     # Asterisk lists
        ]
        
        list_count = sum(len(re.findall(pattern, content, re.MULTILINE)) 
                        for pattern in list_patterns)
        
        if list_count < 3:
            warnings.append("Consider using more lists for step-by-step procedures")
            score -= 0.1
        else:
            score += 0.2
        
        # Check for version information
        if "version" not in sop_data:
            warnings.append("No version information found")
            score -= 0.1
        
        # Normalize score
        score = max(0.0, min(1.0, score + 0.5))
        
        return errors, warnings, score
    
    def _validate_metadata(self, sop_data: Dict[str, Any]) -> Tuple[List[str], List[str], float]:
        """Validate SOP metadata"""
        errors = []
        warnings = []
        score = 0.0
        
        # Check for approval status
        if "metadata" in sop_data:
            metadata = sop_data["metadata"]
            
            if "approval_status" in metadata:
                status = metadata["approval_status"].lower()
                if status in ["draft", "pending"]:
                    warnings.append(f"SOP is in {status} status - not yet approved")
                    score -= 0.2
                elif status == "approved":
                    score += 0.2
            else:
                warnings.append("No approval status found")
                score -= 0.1
            
            # Check for review date
            if "last_review" in metadata:
                try:
                    review_date = datetime.fromisoformat(metadata["last_review"].replace('Z', '+00:00'))
                    days_since_review = (datetime.now() - review_date).days
                    if days_since_review > 365:
                        warnings.append(f"SOP was last reviewed {days_since_review} days ago")
                        score -= 0.1
                    else:
                        score += 0.1
                except:
                    warnings.append("Invalid review date format")
                    score -= 0.1
            else:
                warnings.append("No last review date found")
                score -= 0.1
            
            # Check for author information
            if "author" not in metadata:
                warnings.append("No author information found")
                score -= 0.1
        
        # Check for tags
        if "tags" in sop_data and sop_data["tags"]:
            score += 0.1
        else:
            warnings.append("No tags found - consider adding tags for better categorization")
            score -= 0.1
        
        # Normalize score
        score = max(0.0, min(1.0, score + 0.5))
        
        return errors, warnings, score
    
    async def validate_multiple_sops(self, sop_data_list: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Validate multiple SOPs"""
        try:
            results = []
            for sop_data in sop_data_list:
                result = await self.validate_sop(sop_data)
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Multiple SOP validation failed: {str(e)}")
            return []
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Get summary of validation results"""
        if not results:
            return {}
        
        total_sops = len(results)
        valid_sops = sum(1 for r in results if r.is_valid)
        total_errors = sum(len(r.errors) for r in results)
        total_warnings = sum(len(r.warnings) for r in results)
        avg_score = sum(r.validation_score for r in results) / total_sops
        
        return {
            "total_sops": total_sops,
            "valid_sops": valid_sops,
            "invalid_sops": total_sops - valid_sops,
            "validation_rate": valid_sops / total_sops if total_sops > 0 else 0,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "average_validation_score": avg_score,
            "score_distribution": {
                "excellent": len([r for r in results if r.validation_score >= 0.9]),
                "good": len([r for r in results if 0.7 <= r.validation_score < 0.9]),
                "fair": len([r for r in results if 0.5 <= r.validation_score < 0.7]),
                "poor": len([r for r in results if r.validation_score < 0.5])
            }
        }
