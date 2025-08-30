#!/usr/bin/env python3
"""
Test script for the Customer Support System
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing module imports...")
    
    try:
        from utils.config import Config
        print("✓ Config module imported successfully")
    except Exception as e:
        print(f"✗ Config module import failed: {e}")
        return False
    
    try:
        from utils.tool_registry import ToolRegistry
        print("✓ ToolRegistry module imported successfully")
    except Exception as e:
        print(f"✗ ToolRegistry module import failed: {e}")
        return False
    
    try:
        from utils.sample_data import SampleData
        print("✓ SampleData module imported successfully")
    except Exception as e:
        print(f"✗ SampleData module import failed: {e}")
        return False
    
    try:
        from utils.metrics import MetricsCalculator
        print("✓ MetricsCalculator module imported successfully")
    except Exception as e:
        print(f"✗ MetricsCalculator module import failed: {e}")
        return False
    
    try:
        from utils.fallback_handler import FallbackHandler
        print("✓ FallbackHandler module imported successfully")
    except Exception as e:
        print(f"✗ FallbackHandler module import failed: {e}")
        return False
    
    try:
        from sop_manager.sop_loader import SOPLoader
        print("✓ SOPLoader module imported successfully")
    except Exception as e:
        print(f"✗ SOPLoader module import failed: {e}")
        return False
    
    try:
        from processors.context_manager import ContextManager
        print("✓ ContextManager module imported successfully")
    except Exception as e:
        print(f"✗ ContextManager module import failed: {e}")
        return False
    
    try:
        from processors.intent_processor import IntentProcessor
        print("✓ IntentProcessor module imported successfully")
    except Exception as e:
        print(f"✗ IntentProcessor module import failed: {e}")
        return False
    
    return True

def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from utils.config import Config
        config = Config()
        print("✓ Configuration loaded successfully")
        print(f"  - Model: {config.model_config.model_name}")
        print(f"  - Confidence threshold: {config.guardrails_config.confidence_threshold}")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_tool_registry():
    """Test tool registry"""
    print("\nTesting tool registry...")
    
    try:
        from utils.tool_registry import ToolRegistry
        registry = ToolRegistry()
        
        # Check if tools are registered
        tools = registry.get_tools_by_domain("ecommerce")
        print(f"✓ Tool registry initialized with {len(tools)} e-commerce tools")
        
        # Test tool descriptions
        descriptions = registry.get_tool_descriptions("ecommerce")
        print(f"✓ Retrieved {len(descriptions)} tool descriptions")
        
        return True
    except Exception as e:
        print(f"✗ Tool registry test failed: {e}")
        return False

def test_sample_data():
    """Test sample data"""
    print("\nTesting sample data...")
    
    try:
        from utils.sample_data import SampleData
        data = SampleData()
        
        # Test queries
        queries = data.get_sample_queries("ecommerce")
        print(f"✓ Retrieved {len(queries)} e-commerce sample queries")
        
        # Test SOPs
        sops = data.get_sample_sops("ecommerce")
        print(f"✓ Retrieved {len(sops)} e-commerce sample SOPs")
        
        return True
    except Exception as e:
        print(f"✗ Sample data test failed: {e}")
        return False

def test_metrics():
    """Test metrics calculator"""
    print("\nTesting metrics calculator...")
    
    try:
        from utils.metrics import MetricsCalculator
        calculator = MetricsCalculator()
        
        # Test quality calculation
        metrics = calculator.calculate_response_quality(
            customer_message="Where is my order?",
            agent_response="I can help you track your order. Please provide your order ID.",
            response_time=1.5,
            confidence_score=0.8,
            tools_used=["order_lookup"],
            sop_compliance=True
        )
        
        print("✓ Metrics calculation successful")
        print(f"  - Overall quality: {metrics['metrics']['overall_quality']:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ Metrics test failed: {e}")
        return False

def test_fallback_handler():
    """Test fallback handler"""
    print("\nTesting fallback handler...")
    
    try:
        from utils.fallback_handler import FallbackHandler, FallbackReason
        handler = FallbackHandler()
        
        # Test fallback handling
        result = asyncio.run(handler.handle_fallback(
            reason=FallbackReason.LOW_CONFIDENCE,
            context={"customer_message": "Test message"}
        ))
        
        print("✓ Fallback handler test successful")
        print(f"  - Fallback type: {result.get('fallback_type', 'unknown')}")
        
        return True
    except Exception as e:
        print(f"✗ Fallback handler test failed: {e}")
        return False

async def test_async_components():
    """Test async components"""
    print("\nTesting async components...")
    
    try:
        from utils.tool_registry import ToolRegistry
        registry = ToolRegistry()
        
        # Test async tool execution
        result = await registry.execute_tool(
            tool_id="order_lookup",
            input_data={"order_id": "TEST123"}
        )
        
        print("✓ Async tool execution successful")
        print(f"  - Tool result: {result.success}")
        
        await registry.close()
        return True
    except Exception as e:
        print(f"✗ Async components test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Customer Support System Test Suite")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your installation.")
        return False
    
    # Test configuration
    if not test_config():
        print("\n❌ Configuration test failed.")
        return False
    
    # Test tool registry
    if not test_tool_registry():
        print("\n❌ Tool registry test failed.")
        return False
    
    # Test sample data
    if not test_sample_data():
        print("\n❌ Sample data test failed.")
        return False
    
    # Test metrics
    if not test_metrics():
        print("\n❌ Metrics test failed.")
        return False
    
    # Test fallback handler
    if not test_fallback_handler():
        print("\n❌ Fallback handler test failed.")
        return False
    
    # Test async components
    if not asyncio.run(test_async_components()):
        print("\n❌ Async components test failed.")
        return False
    
    print("\n🎉 All tests passed successfully!")
    print("\n✅ The Customer Support System is ready to use.")
    print("\nTo start the application, run:")
    print("  streamlit run main.py")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
