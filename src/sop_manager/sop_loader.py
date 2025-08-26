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

class SOPLoader:
    """Loads and parses SOP documents from various sources"""
    
    def __init__(self, sop_directory: Optional[Path] = None):
        self.sop_directory = sop_directory or Path("data/sops")
        self.logger = logging.getLogger(__name__)
        
        # Supported file formats
        self.supported_formats = {'.json', '.yaml', '.yml', '.txt', '.md'}
    
    def load_sop_from_file(self, file_path: Union[str, Path]) -> Optional[SOPDocument]:
        """Load a single SOP from file"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                self.logger.error(f"SOP file not found: {file_path}")
                return None
            
            if file_path.suffix not in self.supported_formats:
                self.logger.error(f"Unsupported file format: {file_path.suffix}")
                return None
            
            # Load based on file type
            if file_path.suffix == '.json':
                return self._load_json_sop(file_path)
            elif file_path.suffix in ['.yaml', '.yml']:
                return self._load_yaml_sop(file_path)
            elif file_path.suffix in ['.txt', '.md']:
                return self._load_text_sop(file_path)
            
        except Exception as e:
            self.logger.error(f"Failed to load SOP from {file_path}: {str(e)}")
            return None
    
    def load_all_sops(self, domain: Optional[str] = None) -> List[SOPDocument]:
        """Load all SOPs from directory"""
        try:
            sops = []
            
            if not self.sop_directory.exists():
                self.logger.warning(f"SOP directory not found: {self.sop_directory}")
                return sops
            
            # Recursively find SOP files
            for file_path in self.sop_directory.rglob("*"):
                if file_path.suffix in self.supported_formats:
                    sop = self.load_sop_from_file(file_path)
                    if sop and (domain is None or sop.domain == domain):
                        sops.append(sop)
            
            self.logger.info(f"Loaded {len(sops)} SOPs" + (f" for domain '{domain}'" if domain else ""))
            return sops
            
        except Exception as e:
            self.logger.error(f"Failed to load SOPs: {str(e)}")
            return []
    
    def _load_json_sop(self, file_path: Path) -> Optional[SOPDocument]:
        """Load SOP from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate required fields
            if not all(key in data for key in ['sop_id', 'title', 'domain', 'content']):
                self.logger.error(f"Missing required fields in {file_path}")
                return None
            
            return SOPDocument(**data)
            
        except Exception as e:
            self.logger.error(f"Failed to parse JSON SOP {file_path}: {str(e)}")
            return None
    
    def _load_yaml_sop(self, file_path: Path) -> Optional[SOPDocument]:
        """Load SOP from YAML file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not all(key in data for key in ['sop_id', 'title', 'domain', 'content']):
                self.logger.error(f"Missing required fields in {file_path}")
                return None
            
            return SOPDocument(**data)
            
        except Exception as e:
            self.logger.error(f"Failed to parse YAML SOP {file_path}: {str(e)}")
            return None
    
    def _load_text_sop(self, file_path: Path) -> Optional[SOPDocument]:
        """Load SOP from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Generate basic metadata from filename and content
            sop_id = file_path.stem
            title = file_path.stem.replace('_', ' ').title()
            
            # Try to infer domain from path or filename
            domain = self._infer_domain_from_path(file_path)
            
            return SOPDocument(
                sop_id=sop_id,
                title=title,
                domain=domain,
                content=content,
                metadata={
                    "source_file": str(file_path),
                    "file_format": "text"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse text SOP {file_path}: {str(e)}")
            return None
    
    def _infer_domain_from_path(self, file_path: Path) -> str:
        """Infer domain from file path"""
        path_str = str(file_path).lower()
        
        if 'ecommerce' in path_str or 'e-commerce' in path_str:
            return 'ecommerce'
        elif 'telecom' in path_str or 'telecommunications' in path_str:
            return 'telecom'
        elif 'utilities' in path_str or 'utility' in path_str:
            return 'utilities'
        else:
            return 'general'
    
    def create_sop_template(self, domain: str) -> Dict[str, Any]:
        """Create an SOP template"""
        return {
            "sop_id": f"{domain}_sop_001",
            "title": f"Sample SOP for {domain.title()}",
            "domain": domain,
            "version": "1.0",
            "content": "This is a sample SOP content. Please replace with actual procedures.",
            "sections": [
                {
                    "title": "Overview",
                    "content": "Brief overview of the procedure"
                },
                {
                    "title": "Steps",
                    "content": "Step-by-step instructions"
                },
                {
                    "title": "Notes",
                    "content": "Additional notes and considerations"
                }
            ],
            "metadata": {
                "author": "System Administrator",
                "approval_status": "draft",
                "last_review": datetime.now().isoformat()
            },
            "tags": [domain, "template"]
        }