"""
Sample data for the customer support system demo
"""

from typing import Dict, List, Any
import json
from pathlib import Path

class SampleData:
    """Sample data for demonstration purposes"""
    
    def __init__(self):
        self.sample_queries = {
            "general": [
                "I need help with my account",
                "How do I contact customer service?",
                "What are your business hours?",
                "I want to speak to a human agent",
                "Can you help me with a complaint?",
                "What is your return policy?",
                "How do I update my contact information?",
                "I forgot my password"
            ],
            "ecommerce": [
                "Where is my order?",
                "I want to track my package",
                "Can I cancel my order?",
                "I received a damaged item",
                "How do I return a product?",
                "What's the status of my refund?",
                "I want to change my shipping address",
                "Do you offer free shipping?",
                "What payment methods do you accept?",
                "I have a coupon code"
            ],
            "telecom": [
                "What's my current balance?",
                "I need to check my data usage",
                "How do I upgrade my plan?",
                "I'm having trouble with my service",
                "Can you help me with billing?",
                "I want to add international calling",
                "How do I reset my password?",
                "I need to report a technical issue",
                "What are my plan features?",
                "I want to cancel my service"
            ],
            "utilities": [
                "What's my current bill amount?",
                "When is my bill due?",
                "I want to set up automatic payments",
                "How do I read my meter?",
                "I'm experiencing a power outage",
                "Can you help me with my account?",
                "I want to change my billing address",
                "How do I report a gas leak?",
                "What are my usage patterns?",
                "I need to start new service"
            ]
        }
        
        self.sample_sops = {
            "ecommerce": [
                {
                    "sop_id": "ECO001",
                    "title": "Order Status Inquiry",
                    "domain": "ecommerce",
                    "content": """
                    When a customer asks about their order status:
                    1. Ask for order ID or customer email
                    2. Use order lookup tool to check status
                    3. Provide current status and estimated delivery
                    4. If delayed, explain reason and offer solutions
                    5. Offer to send tracking information
                    """,
                    "tags": ["order", "tracking", "delivery"]
                },
                {
                    "sop_id": "ECO002",
                    "title": "Return and Refund Process",
                    "domain": "ecommerce",
                    "content": """
                    For return and refund requests:
                    1. Verify customer identity and order details
                    2. Check if item is eligible for return (within 30 days)
                    3. Explain return process and requirements
                    4. Provide return shipping label if applicable
                    5. Set expectations for refund timeline (5-7 business days)
                    6. Offer alternatives like exchange or store credit
                    """,
                    "tags": ["return", "refund", "customer_service"]
                }
            ],
            "telecom": [
                {
                    "sop_id": "TEL001",
                    "title": "Account Balance Check",
                    "domain": "telecom",
                    "content": """
                    When checking account balance:
                    1. Verify customer identity (phone number or account ID)
                    2. Use balance check tool to get current information
                    3. Provide balance, data usage, and plan details
                    4. Explain any pending charges or credits
                    5. Offer payment options if balance is low
                    6. Suggest plan optimization if usage is high
                    """,
                    "tags": ["balance", "billing", "account"]
                },
                {
                    "sop_id": "TEL002",
                    "title": "Service Issue Resolution",
                    "domain": "telecom",
                    "content": """
                    For service-related issues:
                    1. Listen to customer description of the problem
                    2. Ask clarifying questions to understand the issue
                    3. Check for known outages in the area
                    4. Guide customer through basic troubleshooting steps
                    5. If issue persists, create a support ticket
                    6. Provide ticket number and expected resolution time
                    7. Offer compensation if appropriate
                    """,
                    "tags": ["technical_support", "troubleshooting", "outage"]
                }
            ],
            "utilities": [
                {
                    "sop_id": "UTL001",
                    "title": "Bill Inquiry and Payment",
                    "domain": "utilities",
                    "content": """
                    For bill-related inquiries:
                    1. Verify customer account number or meter number
                    2. Use bill inquiry tool to get current bill information
                    3. Explain bill breakdown and charges
                    4. Provide due date and payment options
                    5. Offer payment plans if customer is struggling
                    6. Explain late fees and consequences
                    7. Help set up automatic payments if requested
                    """,
                    "tags": ["billing", "payment", "account"]
                },
                {
                    "sop_id": "UTL002",
                    "title": "Emergency Service Issues",
                    "domain": "utilities",
                    "content": """
                    For emergency service issues:
                    1. Assess urgency (gas leak, power outage, water main break)
                    2. For gas leaks: instruct customer to evacuate and call emergency services
                    3. For power outages: check for known issues and estimated restoration
                    4. For water issues: check for maintenance or emergency work
                    5. Create high-priority service ticket
                    6. Provide emergency contact numbers
                    7. Follow up within 2 hours for critical issues
                    """,
                    "tags": ["emergency", "safety", "urgent"]
                }
            ],
            "general": [
                {
                    "sop_id": "GEN001",
                    "title": "Customer Authentication",
                    "domain": "general",
                    "content": """
                    For all customer interactions:
                    1. Greet customer professionally
                    2. Ask for customer ID, email, or phone number
                    3. Use customer validation tool to verify identity
                    4. Confirm account status and any restrictions
                    5. Note any special account flags or notes
                    6. Proceed with customer request
                    7. If verification fails, offer alternative identification methods
                    """,
                    "tags": ["authentication", "security", "verification"]
                },
                {
                    "sop_id": "GEN002",
                    "title": "Escalation to Human Agent",
                    "domain": "general",
                    "content": """
                    When escalation is needed:
                    1. Acknowledge customer's frustration or complex request
                    2. Explain why human assistance is needed
                    3. Assure customer that their issue will be resolved
                    4. Provide estimated wait time for human agent
                    5. Offer to schedule a callback if preferred
                    6. Summarize the issue for the human agent
                    7. Transfer customer with warm handoff
                    """,
                    "tags": ["escalation", "human_agent", "customer_service"]
                }
            ]
        }
        
        self.sample_conversations = {
            "ecommerce": [
                {
                    "customer": "Hi, I ordered something last week and I'm wondering where it is",
                    "agent": "I'd be happy to help you track your order. Could you please provide your order ID or the email address you used for the purchase?",
                    "customer": "My order ID is ORD12345",
                    "agent": "Thank you! Let me check the status of your order ORD12345 for you.",
                    "tools_used": ["order_lookup"],
                    "sop_followed": "ECO001"
                }
            ],
            "telecom": [
                {
                    "customer": "I need to check my account balance",
                    "agent": "I can help you check your account balance. Could you please provide your phone number or account ID?",
                    "customer": "My phone number is 555-0123",
                    "agent": "Perfect! Let me check your account balance and usage for the number 555-0123.",
                    "tools_used": ["balance_check"],
                    "sop_followed": "TEL001"
                }
            ]
        }
    
    def get_sample_queries(self, domain: str = "general") -> List[str]:
        """Get sample queries for a specific domain"""
        return self.sample_queries.get(domain, self.sample_queries["general"])
    
    def get_sample_sops(self, domain: str = "general") -> List[Dict[str, Any]]:
        """Get sample SOPs for a specific domain"""
        domain_sops = self.sample_sops.get(domain, [])
        general_sops = self.sample_sops.get("general", [])
        return domain_sops + general_sops
    
    def get_sample_conversations(self, domain: str = "general") -> List[Dict[str, Any]]:
        """Get sample conversations for a specific domain"""
        return self.sample_conversations.get(domain, [])
    
    def get_all_sample_data(self) -> Dict[str, Any]:
        """Get all sample data"""
        return {
            "queries": self.sample_queries,
            "sops": self.sample_sops,
            "conversations": self.sample_conversations
        }
    
    def save_sample_data_to_file(self, file_path: str = "sample_data.json"):
        """Save sample data to a JSON file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.get_all_sample_data(), f, indent=2, ensure_ascii=False)
            print(f"Sample data saved to {file_path}")
        except Exception as e:
            print(f"Error saving sample data: {e}")
    
    def load_sample_data_from_file(self, file_path: str = "sample_data.json"):
        """Load sample data from a JSON file"""
        try:
            if Path(file_path).exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'queries' in data:
                    self.sample_queries = data['queries']
                if 'sops' in data:
                    self.sample_sops = data['sops']
                if 'conversations' in data:
                    self.sample_conversations = data['conversations']
                
                print(f"Sample data loaded from {file_path}")
            else:
                print(f"Sample data file {file_path} not found")
        except Exception as e:
            print(f"Error loading sample data: {e}")
    
    def add_sample_query(self, domain: str, query: str):
        """Add a new sample query"""
        if domain not in self.sample_queries:
            self.sample_queries[domain] = []
        
        if query not in self.sample_queries[domain]:
            self.sample_queries[domain].append(query)
    
    def add_sample_sop(self, domain: str, sop: Dict[str, Any]):
        """Add a new sample SOP"""
        if domain not in self.sample_sops:
            self.sample_sops[domain] = []
        
        self.sample_sops[domain].append(sop)
    
    def get_domain_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about sample data across domains"""
        stats = {}
        
        for domain in self.sample_queries.keys():
            stats[domain] = {
                "queries": len(self.sample_queries.get(domain, [])),
                "sops": len(self.sample_sops.get(domain, [])),
                "conversations": len(self.sample_conversations.get(domain, []))
            }
        
        return stats
