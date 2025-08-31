#!/usr/bin/env python3
"""
Test script to isolate import issues
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

print("Testing imports...")

try:
    print("1. Testing sop_manager imports...")
    from sop_manager.sop_loader import SOPLoader
    print("   ✓ SOPLoader imported successfully")
    
    from sop_manager.sop_chunker import SOPChunker
    print("   ✓ SOPChunker imported successfully")
    
    from sop_manager.sop_embedder import SOPEmbedder
    print("   ✓ SOPEmbedder imported successfully")
    
    from sop_manager.sop_retriever import SOPRetriever
    print("   ✓ SOPRetriever imported successfully")
    
    from sop_manager.sop_validator import SOPValidator
    print("   ✓ SOPValidator imported successfully")
    
except Exception as e:
    print(f"   ✗ SOP manager import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("2. Testing processors imports...")
    from processors.context_manager import ContextManager
    print("   ✓ ContextManager imported successfully")
    
    from processors.intent_processor import IntentProcessor
    print("   ✓ IntentProcessor imported successfully")
    
    from processors.response_formatter import ResponseFormatter
    print("   ✓ ResponseFormatter imported successfully")
    
    from processors.regional_language_processor import RegionalProcessor
    print("   ✓ RegionalProcessor imported successfully")
    
    from processors.sop_processor import SOPProcessor
    print("   ✓ SOPProcessor imported successfully")
    
except Exception as e:
    print(f"   ✗ Processors import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("3. Testing utils imports...")
    from utils.config import Config
    print("   ✓ Config imported successfully")
    
    from utils.tool_registry import ToolRegistry
    print("   ✓ ToolRegistry imported successfully")
    
    from utils.sample_data import SampleData
    print("   ✓ SampleData imported successfully")
    
except Exception as e:
    print(f"   ✗ Utils import failed: {e}")
    import traceback
    traceback.print_exc()

try:
    print("4. Testing agents imports...")
    from agents.base_agent import BaseAgent
    print("   ✓ BaseAgent imported successfully")
    
    from agents.intent_classifier_agent import IntentClassifierAgent
    print("   ✓ IntentClassifierAgent imported successfully")
    
    from agents.sop_retrieval_agent import SOPRetrievalAgent
    print("   ✓ SOPRetrievalAgent imported successfully")
    
    from agents.tool_orchestrator_agent import ToolOrchestratorAgent
    print("   ✓ ToolOrchestratorAgent imported successfully")
    
    from agents.quality_checker_agent import QualityCheckerAgent
    print("   ✓ QualityCheckerAgent imported successfully")
    
    from agents.customer_response_agent import CustomerResponseAgent
    print("   ✓ CustomerResponseAgent imported successfully")
    
except Exception as e:
    print(f"   ✗ Agents import failed: {e}")
    import traceback
    traceback.print_exc()

print("Import test completed!")
