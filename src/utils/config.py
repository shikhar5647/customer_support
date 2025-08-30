"""
Configuration management for the customer support system
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import yaml

# Load environment variables
load_dotenv()

class GuardrailsConfig(BaseModel):
    """Configuration for AI guardrails and safety measures"""
    max_response_length: int = 500
    prohibited_topics: list = Field(default_factory=lambda: ["personal_info", "financial_data", "sensitive_data"])
    confidence_threshold: float = 0.7
    max_retries: int = 3
    timeout_seconds: int = 30

class ModelConfig(BaseModel):
    """Configuration for the LLM model"""
    model_name: str = "gemini-1.5-pro"
    temperature: float = 0.3
    max_tokens: int = 1000
    top_p: float = 0.9
    top_k: int = 40

class SystemConfig(BaseModel):
    """System-wide configuration"""
    max_conversation_length: int = 5000  # tokens
    max_session_duration: int = 3600  # seconds (1 hour)
    enable_logging: bool = True
    log_level: str = "INFO"
    enable_metrics: bool = True
    enable_fallback: bool = True

class Config:
    """Main configuration class for the customer support system"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config.yaml"
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file and environment variables"""
        # Default configurations
        self.guardrails_config = GuardrailsConfig()
        self.model_config = ModelConfig()
        self.system_config = SystemConfig()
        
        # Load from config file if exists
        if Path(self.config_file).exists():
            self._load_from_file()
        
        # Override with environment variables
        self._load_from_env()
    
    def _load_from_file(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if config_data:
                # Update guardrails config
                if 'guardrails' in config_data:
                    self.guardrails_config = GuardrailsConfig(**config_data['guardrails'])
                
                # Update model config
                if 'model' in config_data:
                    self.model_config = ModelConfig(**config_data['model'])
                
                # Update system config
                if 'system' in config_data:
                    self.system_config = SystemConfig(**config_data['system'])
                    
        except Exception as e:
            print(f"Warning: Could not load config file {self.config_file}: {e}")
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Gemini API key
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Override configurations with environment variables
        if os.getenv("GEMINI_MODEL_NAME"):
            self.model_config.model_name = os.getenv("GEMINI_MODEL_NAME")
        
        if os.getenv("GEMINI_TEMPERATURE"):
            self.model_config.temperature = float(os.getenv("GEMINI_TEMPERATURE"))
        
        if os.getenv("CONFIDENCE_THRESHOLD"):
            self.guardrails_config.confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD"))
        
        if os.getenv("MAX_RETRIES"):
            self.guardrails_config.max_retries = int(os.getenv("MAX_RETRIES"))
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return {
            "guardrails": self.guardrails_config.dict(),
            "model": self.model_config.dict(),
            "system": self.system_config.dict(),
            "gemini_api_key": "***" if self.gemini_api_key else None
        }
    
    def validate_config(self) -> bool:
        """Validate the configuration"""
        try:
            # Check required fields
            if not self.gemini_api_key:
                return False
            
            # Validate ranges
            if not (0 <= self.model_config.temperature <= 2):
                return False
            
            if not (0 <= self.guardrails_config.confidence_threshold <= 1):
                return False
            
            return True
            
        except Exception:
            return False
    
    def save_config(self, file_path: Optional[str] = None):
        """Save current configuration to file"""
        file_path = file_path or self.config_file
        
        config_data = {
            "guardrails": self.guardrails_config.dict(),
            "model": self.model_config.dict(),
            "system": self.system_config.dict()
        }
        
        try:
            with open(file_path, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False)
        except Exception as e:
            print(f"Error saving config: {e}")

# Global configuration instance
config = Config()
