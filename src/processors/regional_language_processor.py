"""
Regional Language Processing Module
"""
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
import re
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class LanguageInfo(BaseModel):
    """Language information"""
    language_code: str
    language_name: str
    region: str
    script: str = "latin"
    direction: str = "ltr"  # left-to-right
    confidence: float = 1.0

class RegionalProcessor:
    """Processes text for regional language variations"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Load language configurations
        self.language_configs = self._load_language_configs(config_file)
        
        # Common regional variations
        self.regional_patterns = {
            "en_US": {
                "spelling": {
                    "color": "colour",
                    "center": "centre",
                    "favorite": "favourite",
                    "realize": "realise"
                },
                "date_format": "MM/DD/YYYY",
                "currency": "USD",
                "timezone": "UTC-5"
            },
            "en_GB": {
                "spelling": {
                    "colour": "color",
                    "centre": "center",
                    "favourite": "favorite",
                    "realise": "realize"
                },
                "date_format": "DD/MM/YYYY",
                "currency": "GBP",
                "timezone": "UTC+0"
            },
            "en_AU": {
                "spelling": {
                    "colour": "color",
                    "centre": "center",
                    "favourite": "favorite",
                    "realise": "realize"
                },
                "date_format": "DD/MM/YYYY",
                "currency": "AUD",
                "timezone": "UTC+10"
            }
        }
    
    def _load_language_configs(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load language configurations from file"""
        try:
            if config_file and Path(config_file).exists():
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Default configurations
                return {
                    "supported_languages": ["en", "es", "fr", "de", "it", "pt", "nl"],
                    "regional_variants": ["en_US", "en_GB", "en_AU", "es_ES", "es_MX", "fr_FR", "fr_CA"],
                    "fallback_language": "en"
                }
        except Exception as e:
            self.logger.warning(f"Failed to load language config: {str(e)}")
            return {}
    
    def detect_language(self, text: str) -> LanguageInfo:
        """Detect the language of the text"""
        try:
            # Simple language detection based on common words
            text_lower = text.lower()
            
            # English patterns
            en_words = ["the", "and", "or", "but", "in", "on", "at", "to", "for", "of"]
            en_score = sum(1 for word in en_words if word in text_lower)
            
            # Spanish patterns
            es_words = ["el", "la", "los", "las", "y", "o", "pero", "en", "con", "por"]
            es_score = sum(1 for word in es_words if word in text_lower)
            
            # French patterns
            fr_words = ["le", "la", "les", "et", "ou", "mais", "dans", "avec", "pour", "de"]
            fr_score = sum(1 for word in fr_words if word in text_lower)
            
            # German patterns
            de_words = ["der", "die", "das", "und", "oder", "aber", "in", "mit", "für", "von"]
            de_score = sum(1 for word in de_words if word in text_lower)
            
            # Determine language with highest score
            scores = {
                "en": en_score,
                "es": es_score,
                "fr": fr_score,
                "de": de_score
            }
            
            detected_lang = max(scores, key=scores.get)
            confidence = scores[detected_lang] / max(len(text.split()), 1)
            
            return LanguageInfo(
                language_code=detected_lang,
                language_name=self._get_language_name(detected_lang),
                region=self._get_default_region(detected_lang),
                confidence=min(confidence, 1.0)
            )
            
        except Exception as e:
            self.logger.error(f"Language detection failed: {str(e)}")
            return LanguageInfo(
                language_code="en",
                language_name="English",
                region="US",
                confidence=0.0
            )
    
    def _get_language_name(self, lang_code: str) -> str:
        """Get language name from code"""
        language_names = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "nl": "Dutch"
        }
        return language_names.get(lang_code, "Unknown")
    
    def _get_default_region(self, lang_code: str) -> str:
        """Get default region for language"""
        default_regions = {
            "en": "US",
            "es": "ES",
            "fr": "FR",
            "de": "DE",
            "it": "IT",
            "pt": "PT",
            "nl": "NL"
        }
        return default_regions.get(lang_code, "US")
    
    def normalize_regional_variations(self, text: str, target_region: str = "en_US") -> str:
        """Normalize text to target regional variation"""
        try:
            if target_region not in self.regional_patterns:
                return text
            
            normalized_text = text
            patterns = self.regional_patterns[target_region]
            
            # Apply spelling variations
            for variant, standard in patterns["spelling"].items():
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(variant) + r'\b'
                normalized_text = re.sub(pattern, standard, normalized_text, flags=re.IGNORECASE)
            
            return normalized_text
            
        except Exception as e:
            self.logger.error(f"Regional normalization failed: {str(e)}")
            return text
    
    def format_for_region(
        self, 
        text: str, 
        target_region: str,
        format_type: str = "date"
    ) -> str:
        """Format text for specific regional preferences"""
        try:
            if target_region not in self.regional_patterns:
                return text
            
            patterns = self.regional_patterns[target_region]
            
            if format_type == "date":
                return self._format_date_for_region(text, patterns["date_format"])
            elif format_type == "currency":
                return self._format_currency_for_region(text, patterns["currency"])
            elif format_type == "timezone":
                return self._format_timezone_for_region(text, patterns["timezone"])
            else:
                return text
                
        except Exception as e:
            self.logger.error(f"Regional formatting failed: {str(e)}")
            return text
    
    def _format_date_for_region(self, text: str, date_format: str) -> str:
        """Format date for regional preferences"""
        # This is a simplified implementation
        # In production, use proper date parsing and formatting libraries
        
        # Look for common date patterns
        date_patterns = [
            r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b',  # YYYY-MM-DD
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                if date_format == "MM/DD/YYYY":
                    # Convert to MM/DD/YYYY
                    if len(match.group(1)) == 4:  # YYYY-MM-DD format
                        year, month, day = match.groups()
                        new_date = f"{month}/{day}/{year}"
                    else:
                        month, day, year = match.groups()
                        new_date = f"{month}/{day}/{year}"
                else:
                    # Convert to DD/MM/YYYY
                    if len(match.group(1)) == 4:  # YYYY-MM-DD format
                        year, month, day = match.groups()
                        new_date = f"{day}/{month}/{year}"
                    else:
                        month, day, year = match.groups()
                        new_date = f"{day}/{month}/{year}"
                
                text = text.replace(match.group(0), new_date)
        
        return text
    
    def _format_currency_for_region(self, text: str, currency: str) -> str:
        """Format currency for regional preferences"""
        # This is a simplified implementation
        # In production, use proper currency formatting libraries
        
        currency_symbols = {
            "USD": "$",
            "GBP": "£",
            "EUR": "€",
            "AUD": "A$"
        }
        
        symbol = currency_symbols.get(currency, "$")
        
        # Look for currency amounts and format them
        currency_pattern = r'\b(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:dollars?|USD|GBP|EUR|AUD)\b'
        
        def format_currency(match):
            amount = match.group(1)
            return f"{symbol}{amount}"
        
        return re.sub(currency_pattern, format_currency, text, flags=re.IGNORECASE)
    
    def _format_timezone_for_region(self, text: str, timezone: str) -> str:
        """Format timezone for regional preferences"""
        # This is a simplified implementation
        # In production, use proper timezone formatting libraries
        
        # Look for timezone references and convert them
        timezone_pattern = r'\b(UTC[+-]\d{1,2})\b'
        
        def format_timezone(match):
            return timezone
        
        return re.sub(timezone_pattern, format_timezone, text)
    
    def get_regional_suggestions(self, text: str, detected_region: str) -> List[str]:
        """Get suggestions for regional adaptations"""
        suggestions = []
        
        try:
            # Check for potential regional issues
            if detected_region.startswith("en_"):
                # Check for mixed spelling
                us_spellings = ["color", "center", "favorite", "realize"]
                uk_spellings = ["colour", "centre", "favourite", "realise"]
                
                text_lower = text.lower()
                has_us = any(spelling in text_lower for spelling in us_spellings)
                has_uk = any(spelling in text_lower for spelling in uk_spellings)
                
                if has_us and has_uk:
                    suggestions.append("Mixed US/UK spelling detected. Consider standardizing to one regional variant.")
                
                # Check for date format consistency
                date_patterns = [
                    r'\b\d{1,2}/\d{1,2}/\d{4}\b',
                    r'\b\d{4}-\d{1,2}-\d{1,2}\b'
                ]
                
                if any(re.search(pattern, text) for pattern in date_patterns):
                    if detected_region == "en_US":
                        suggestions.append("Consider using MM/DD/YYYY date format for US audience.")
                    elif detected_region in ["en_GB", "en_AU"]:
                        suggestions.append("Consider using DD/MM/YYYY date format for international audience.")
            
            # Check for currency references
            currency_pattern = r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|USD|GBP|EUR|AUD)\b'
            if re.search(currency_pattern, text, re.IGNORECASE):
                suggestions.append("Consider specifying currency clearly for international audience.")
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Failed to generate regional suggestions: {str(e)}")
            return []
    
    def get_supported_regions(self) -> List[str]:
        """Get list of supported regions"""
        return list(self.regional_patterns.keys())
    
    def get_language_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """Get language statistics for multiple texts"""
        try:
            language_counts = {}
            total_texts = len(texts)
            
            for text in texts:
                lang_info = self.detect_language(text)
                lang_code = lang_info.language_code
                language_counts[lang_code] = language_counts.get(lang_code, 0) + 1
            
            return {
                "total_texts": total_texts,
                "language_distribution": language_counts,
                "primary_language": max(language_counts, key=language_counts.get) if language_counts else "unknown",
                "supported_languages": self.language_configs.get("supported_languages", [])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate language statistics: {str(e)}")
            return {}
