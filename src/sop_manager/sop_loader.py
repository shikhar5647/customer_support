"""
SOP Loading and Parsing Module
"""
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from pathlib import Path
import json
import yaml
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class SOPDocument(BaseModel):
    """SOP Document model"""
    sop_id: str
    title: str
    domain: str
    version: str = "1.0"
    content: str
    sections: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    @validator('domain')
    def validate_domain(cls, v):
        allowed_domains = ['ecommerce', 'telecom', 'utilities', 'general']
        if v not in allowed_domains:
            raise ValueError(f'Domain must be one of: {allowed_domains}')
        return v

