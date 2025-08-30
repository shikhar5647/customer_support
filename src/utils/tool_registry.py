"""
Tool Registry for managing external tools and APIs
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Union
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)

class ToolMetadata(BaseModel):
    """Metadata for a tool/API"""
    tool_id: str
    name: str
    description: str
    category: str
    domain: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    api_endpoint: Optional[str] = None
    function: Optional[Callable] = None
    rate_limit: Optional[int] = None
    timeout: int = 30
    requires_auth: bool = False
    auth_type: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    usage_count: int = 0
    success_rate: float = 1.0

class ToolExecutionResult(BaseModel):
    """Result of tool execution"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ToolRegistry:
    """Registry for managing tools and APIs"""
    
    def __init__(self):
        self.tools: Dict[str, ToolMetadata] = {}
        self.execution_history: List[ToolExecutionResult] = []
        self.rate_limit_trackers: Dict[str, List[datetime]] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Register default tools
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools for common customer support scenarios"""
        
        # E-commerce tools
        self.register_tool(ToolMetadata(
            tool_id="order_lookup",
            name="Order Lookup",
            description="Look up order details by order ID or customer ID",
            category="ecommerce",
            domain="ecommerce",
            input_schema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "customer_id": {"type": "string"},
                    "email": {"type": "string"}
                },
                "required": ["order_id"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "status": {"type": "string"},
                    "items": {"type": "array"},
                    "total": {"type": "number"},
                    "shipping_address": {"type": "object"}
                }
            },
            function=self._mock_order_lookup
        ))
        
        # Telecom tools
        self.register_tool(ToolMetadata(
            tool_id="balance_check",
            name="Balance Check",
            description="Check account balance and usage",
            category="telecom",
            domain="telecom",
            input_schema={
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string"},
                    "account_id": {"type": "string"}
                },
                "required": ["phone_number"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "balance": {"type": "number"},
                    "data_usage": {"type": "object"},
                    "plan_details": {"type": "object"}
                }
            },
            function=self._mock_balance_check
        ))
        
        # Utility tools
        self.register_tool(ToolMetadata(
            tool_id="bill_inquiry",
            name="Bill Inquiry",
            description="Check bill status and amount",
            category="utilities",
            domain="utilities",
            input_schema={
                "type": "object",
                "properties": {
                    "account_number": {"type": "string"},
                    "meter_number": {"type": "string"}
                },
                "required": ["account_number"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "bill_amount": {"type": "number"},
                    "due_date": {"type": "string"},
                    "usage": {"type": "object"}
                }
            },
            function=self._mock_bill_inquiry
        ))
        
        # General tools
        self.register_tool(ToolMetadata(
            tool_id="customer_validation",
            name="Customer Validation",
            description="Validate customer information",
            category="general",
            domain="general",
            input_schema={
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "email": {"type": "string"},
                    "phone": {"type": "string"}
                },
                "required": ["customer_id"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "valid": {"type": "boolean"},
                    "customer_info": {"type": "object"},
                    "verification_status": {"type": "string"}
                }
            },
            function=self._mock_customer_validation
        ))
    
    def register_tool(self, tool: ToolMetadata):
        """Register a new tool"""
        if tool.tool_id in self.tools:
            logger.warning(f"Tool {tool.tool_id} already exists, updating...")
        
        self.tools[tool.tool_id] = tool
        logger.info(f"Registered tool: {tool.name} ({tool.tool_id})")
    
    def unregister_tool(self, tool_id: str):
        """Unregister a tool"""
        if tool_id in self.tools:
            del self.tools[tool_id]
            logger.info(f"Unregistered tool: {tool_id}")
    
    def get_tool(self, tool_id: str) -> Optional[ToolMetadata]:
        """Get tool by ID"""
        return self.tools.get(tool_id)
    
    def get_tools_by_domain(self, domain: str) -> List[ToolMetadata]:
        """Get tools by domain"""
        return [tool for tool in self.tools.values() if tool.domain == domain]
    
    def get_tools_by_category(self, category: str) -> List[ToolMetadata]:
        """Get tools by category"""
        return [tool for tool in self.tools.values() if tool.category == category]
    
    async def execute_tool(
        self, 
        tool_id: str, 
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ToolExecutionResult:
        """Execute a tool with given input"""
        
        tool = self.get_tool(tool_id)
        if not tool:
            return ToolExecutionResult(
                success=False,
                error=f"Tool {tool_id} not found"
            )
        
        if not tool.is_active:
            return ToolExecutionResult(
                success=False,
                error=f"Tool {tool_id} is not active"
            )
        
        # Check rate limiting
        if not self._check_rate_limit(tool_id):
            return ToolExecutionResult(
                success=False,
                error=f"Rate limit exceeded for tool {tool_id}"
            )
        
        start_time = datetime.now()
        
        try:
            # Execute tool
            if tool.function:
                # Local function
                if asyncio.iscoroutinefunction(tool.function):
                    result = await tool.function(input_data, context)
                else:
                    result = tool.function(input_data, context)
            elif tool.api_endpoint:
                # API call
                result = await self._call_api(tool, input_data)
            else:
                return ToolExecutionResult(
                    success=False,
                    error=f"Tool {tool_id} has no execution method"
                )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update tool usage
            tool.last_used = datetime.now()
            tool.usage_count += 1
            
            # Record success
            self._update_success_rate(tool_id, True)
            
            return ToolExecutionResult(
                success=True,
                data=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            
            # Record failure
            self._update_success_rate(tool_id, False)
            
            logger.error(f"Tool {tool_id} execution failed: {error_msg}")
            
            return ToolExecutionResult(
                success=False,
                error=error_msg,
                execution_time=execution_time
            )
    
    def _check_rate_limit(self, tool_id: str) -> bool:
        """Check if tool is within rate limit"""
        tool = self.tools.get(tool_id)
        if not tool or not tool.rate_limit:
            return True
        
        now = datetime.now()
        if tool_id not in self.rate_limit_trackers:
            self.rate_limit_trackers[tool_id] = []
        
        # Remove old timestamps
        self.rate_limit_trackers[tool_id] = [
            ts for ts in self.rate_limit_trackers[tool_id]
            if now - ts < timedelta(minutes=1)
        ]
        
        # Check if under limit
        if len(self.rate_limit_trackers[tool_id]) >= tool.rate_limit:
            return False
        
        # Add current timestamp
        self.rate_limit_trackers[tool_id].append(now)
        return True
    
    def _update_success_rate(self, tool_id: str, success: bool):
        """Update tool success rate"""
        tool = self.tools.get(tool_id)
        if tool:
            # Simple moving average
            tool.success_rate = (tool.success_rate * 0.9) + (0.1 if success else 0.0)
    
    async def _call_api(self, tool: ToolMetadata, input_data: Dict[str, Any]) -> Any:
        """Make API call to external service"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        try:
            async with self.session.post(
                tool.api_endpoint,
                json=input_data,
                timeout=aiohttp.ClientTimeout(total=tool.timeout)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"API call failed with status {response.status}")
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")
    
    # Mock tool implementations for demo
    def _mock_order_lookup(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock order lookup tool"""
        order_id = input_data.get("order_id", "UNKNOWN")
        
        # Simulate different order statuses
        if "CANCEL" in order_id.upper():
            status = "Cancelled"
        elif "DELIVERED" in order_id.upper():
            status = "Delivered"
        elif "SHIPPED" in order_id.upper():
            status = "Shipped"
        else:
            status = "Processing"
        
        return {
            "order_id": order_id,
            "status": status,
            "items": [
                {"name": "Sample Product", "quantity": 1, "price": 29.99}
            ],
            "total": 29.99,
            "shipping_address": {
                "street": "123 Main St",
                "city": "Sample City",
                "state": "CA",
                "zip": "12345"
            }
        }
    
    def _mock_balance_check(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock balance check tool"""
        phone_number = input_data.get("phone_number", "UNKNOWN")
        
        return {
            "balance": 45.67,
            "data_usage": {
                "used": "2.5 GB",
                "total": "10 GB",
                "remaining": "7.5 GB"
            },
            "plan_details": {
                "plan_name": "Unlimited Plus",
                "monthly_fee": 59.99,
                "features": ["Unlimited calls", "10GB data", "Unlimited texts"]
            }
        }
    
    def _mock_bill_inquiry(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock bill inquiry tool"""
        account_number = input_data.get("account_number", "UNKNOWN")
        
        return {
            "bill_amount": 125.50,
            "due_date": "2024-02-15",
            "usage": {
                "electricity": "450 kWh",
                "water": "12 CCF",
                "gas": "85 therms"
            }
        }
    
    def _mock_customer_validation(self, input_data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Mock customer validation tool"""
        customer_id = input_data.get("customer_id", "UNKNOWN")
        
        return {
            "valid": True,
            "customer_info": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "+1-555-0123",
                "account_status": "Active"
            },
            "verification_status": "Verified"
        }
    
    def get_tool_descriptions(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get tool descriptions for LLM context"""
        tools = self.get_tools_by_domain(domain) if domain else list(self.tools.values())
        
        descriptions = []
        for tool in tools:
            descriptions.append({
                "tool_id": tool.tool_id,
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
                "category": tool.category
            })
        
        return descriptions
    
    async def close(self):
        """Close the tool registry and cleanup resources"""
        if self.session:
            await self.session.close()
    
    def __del__(self):
        """Cleanup on deletion"""
        if self.session and not self.session.closed:
            asyncio.create_task(self.close())
