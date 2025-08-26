"""
SOP Text Chunking Module
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)