"""
Intent Processing Module
"""
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field
import re
import spacy
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ExtractedEntity(BaseModel):
    """Extracted entity model"""
    entity_type: str
    value: str
    confidence: float = 1.0
    start_pos: int = 0
    end_pos: int = 0

class ProcessedIntent(BaseModel):
    """Processed intent model"""
    original_message: str
    cleaned_message: str
    extracted_entities: List[ExtractedEntity] = Field(default_factory=list)
    message_type: str = "inquiry"  # inquiry, complaint, request, etc.
    keywords: List[str] = Field(default_factory=list)
    preprocessing_metadata: Dict[str, Any] = Field(default_factory=dict)

class IntentProcessor:
    """Processes raw customer messages for intent classification"""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.logger = logging.getLogger(__name__)
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            self.logger.warning(f"SpaCy model {spacy_model} not found. Using blank model.")
            self.nlp = spacy.blank("en")
        
        # Entity patterns for different domains
        self.entity_patterns = {
            "order_number": [
                r'\b(?:order|ord)\s*#?\s*([A-Z0-9]{6,})\b',
                r'\b([A-Z]{2}\d{6,})\b',
                r'\b(\d{10,})\b'
            ],
            "account_number": [
                r'\baccount\s*#?\s*([A-Z0-9]{6,})\b',
                r'\bacc\s*#?\s*([A-Z0-9]{6,})\b'
            ],
            "phone_number": [
                r'\b(\d{3}[-.]?\d{3}[-.]?\d{4})\b',
                r'\b(\(\d{3}\)\s?\d{3}[-.]?\d{4})\b'
            ],
            "email": [
                r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
            ],
            "amount": [
                r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',
                r'\b(\d+(?:,\d{3})*(?:\.\d{2})?\s?(?:dollars?|USD))\b'
            ],
            "date": [
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b',
                r'\b(\d{4}-\d{2}-\d{2})\b',
                r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b'
            ]
        }
        
        # Message type indicators
        self.message_type_patterns = {
            "complaint": ["complaint", "complain", "unhappy", "dissatisfied", "terrible", "awful", "horrible", "worst"],
            "inquiry": ["check", "status", "when", "how", "what", "where", "information", "details"],
            "request": ["want", "need", "please", "can you", "help", "assist", "request"],
            "urgent": ["urgent", "asap", "immediately", "emergency", "critical", "now"]
        }
    
    def process_message(self, message: str, domain: str = "general") -> ProcessedIntent:
        """Process raw customer message"""
        try:
            # Clean and normalize message
            cleaned_message = self._clean_message(message)
            
            # Extract entities
            entities = self._extract_entities(message, domain)
            
            # Determine message type
            message_type = self._classify_message_type(cleaned_message)
            
            # Extract keywords
            keywords = self._extract_keywords(cleaned_message)
            
            # Create metadata
            metadata = {
                "processing_timestamp": datetime.now().isoformat(),
                "domain": domain,
                "message_length": len(message),
                "entity_count": len(entities),
                "keyword_count": len(keywords)
            }
            
            return ProcessedIntent(
                original_message=message,
                cleaned_message=cleaned_message,
                extracted_entities=entities,
                message_type=message_type,
                keywords=keywords,
                preprocessing_metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Message processing failed: {str(e)}")
            return ProcessedIntent(
                original_message=message,
                cleaned_message=message,
                preprocessing_metadata={"error": str(e)}
            )
    
    def _clean_message(self, message: str) -> str:
        """Clean and normalize message"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', message.strip())
        
        # Normalize punctuation
        cleaned = re.sub(r'[!]{2,}', '!', cleaned)
        cleaned = re.sub(r'[?]{2,}', '?', cleaned)
        
        # Fix common typos and contractions
        cleaned = re.sub(r'\bu\b', 'you', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\bur\b', 'your', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\br\b', 'are', cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    def _extract_entities(self, message: str, domain: str) -> List[ExtractedEntity]:
        """Extract entities from message"""
        entities = []
        
        # Use regex patterns for common entities
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, message, re.IGNORECASE)
                for match in matches:
                    entities.append(ExtractedEntity(
                        entity_type=entity_type,
                        value=match.group(1) if match.groups() else match.group(0),
                        confidence=0.9,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
        
        # Use SpaCy NER for additional entities
        try:
            doc = self.nlp(message)
            for ent in doc.ents:
                # Map SpaCy entity types to our types
                entity_type = self._map_spacy_entity_type(ent.label_)
                if entity_type:
                    entities.append(ExtractedEntity(
                        entity_type=entity_type,
                        value=ent.text,
                        confidence=0.8,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char
                    ))
        except Exception as e:
            self.logger.warning(f"SpaCy NER failed: {str(e)}")
        
        # Remove duplicates and sort by confidence
        unique_entities = self._deduplicate_entities(entities)
        return sorted(unique_entities, key=lambda x: x.confidence, reverse=True)
    
    def _map_spacy_entity_type(self, spacy_label: str) -> Optional[str]:
        """Map SpaCy entity labels to our entity types"""
        mapping = {
            "PERSON": "person_name",
            "ORG": "organization",
            "MONEY": "amount",
            "DATE": "date",
            "TIME": "time",
            "CARDINAL": "number",
            "PRODUCT": "product_name"
        }
        return mapping.get(spacy_label)
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove duplicate entities"""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.entity_type, entity.value.lower())
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _classify_message_type(self, message: str) -> str:
        """Classify message type based on content"""
        message_lower = message.lower()
        
        # Score each message type
        type_scores = {}
        for msg_type, indicators in self.message_type_patterns.items():
            score = sum(1 for indicator in indicators if indicator in message_lower)
            if score > 0:
                type_scores[msg_type] = score
        
        # Return type with highest score, default to inquiry
        if type_scores:
            return max(type_scores, key=type_scores.get)
        return "inquiry"
    
    def _extract_keywords(self, message: str) -> List[str]:
        """Extract important keywords from message"""
        try:
            doc = self.nlp(message)
            
            # Extract important words (nouns, verbs, adjectives)
            keywords = []
            for token in doc:
                if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and 
                    not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2):
                    keywords.append(token.lemma_.lower())
            
            # Remove duplicates while preserving order
            return list(dict.fromkeys(keywords))
            
        except Exception as e:
            self.logger.warning(f"Keyword extraction failed: {str(e)}")
            # Fallback: simple word splitting
            words = re.findall(r'\b\w{3,}\b', message.lower())
            return list(set(words))[:10]  # Limit to 10 keywords