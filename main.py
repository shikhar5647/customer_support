#!/usr/bin/env python3
"""
Main Customer Support System Application
AI-based customer response system using LangGraph agents and Gemini LLM
"""

import streamlit as st
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from agents.customer_response_agent import CustomerResponseAgent
from sop_manager.sop_loader import SOPLoader
from processors.context_manager import ContextManager
from processors.intent_processor import IntentProcessor
from utils.tool_registry import ToolRegistry
from utils.config import Config
from utils.sample_data import SampleData

class CustomerSupportSystem:
    """Main customer support system class"""
    
    def __init__(self):
        self.config = Config()
        self.sop_loader = SOPLoader()
        self.tool_registry = ToolRegistry()
        self.context_manager = ContextManager()
        self.intent_processor = IntentProcessor()
        
        # Initialize the main agent
        self.customer_agent = CustomerResponseAgent(
            gemini_api_key=self.config.gemini_api_key,
            sop_manager=self.sop_loader,
            tool_registry=self.tool_registry,
            guardrails_config=self.config.guardrails_config
        )
        
        # Load sample data
        self.sample_data = SampleData()
    
    async def process_customer_query(
        self, 
        customer_message: str, 
        conversation_history: List[Dict[str, str]] = None,
        customer_id: str = None,
        domain: str = "general"
    ) -> Dict[str, Any]:
        """Process customer query and generate response"""
        try:
            # Create input state
            input_data = {
                "customer_message": customer_message,
                "conversation_history": conversation_history or [],
                "customer_id": customer_id,
                "domain": domain,
                "session_id": f"session_{len(conversation_history) if conversation_history else 0}",
                "context": {}
            }
            
            # Process through the agent
            response = await self.customer_agent.execute(input_data)
            
            return response.context.get("customer_response", {})
            
        except Exception as e:
            logger.error(f"Error processing customer query: {str(e)}")
            return {
                "response_text": "I apologize, but I'm experiencing technical difficulties. Please try again or contact human support.",
                "confidence_score": 0.0,
                "escalation_required": True,
                "escalation_reason": f"System error: {str(e)}"
            }

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="AI Customer Support System",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 AI Customer Support System")
    st.markdown("AI-powered customer response system using LangGraph agents and Gemini LLM")
    
    # Initialize system
    if 'support_system' not in st.session_state:
        st.session_state.support_system = CustomerSupportSystem()
        st.session_state.conversation_history = []
        st.session_state.current_domain = "general"
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Domain selection
        domain = st.selectbox(
            "Business Domain",
            ["general", "ecommerce", "telecom", "utilities"],
            index=0
        )
        
        if domain != st.session_state.current_domain:
            st.session_state.current_domain = domain
            st.session_state.conversation_history = []
        
        # Customer ID
        customer_id = st.text_input("Customer ID (Optional)", value="CUST001")
        
        # Sample queries
        st.header("Sample Queries")
        sample_queries = st.session_state.support_system.sample_data.get_sample_queries(domain)
        
        for query in sample_queries:
            if st.button(query, key=f"sample_{query[:20]}"):
                st.session_state.sample_query = query
        
        # Clear conversation
        if st.button("Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Customer Support Chat")
        
        # Display conversation history
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.conversation_history):
                if message["role"] == "customer":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])
        
        # Input area
        if 'sample_query' in st.session_state:
            customer_message = st.text_input(
                "Customer Message",
                value=st.session_state.sample_query,
                key="customer_input"
            )
            del st.session_state.sample_query
        else:
            customer_message = st.text_input(
                "Customer Message",
                placeholder="Type your customer query here...",
                key="customer_input"
            )
        
        # Process button
        if st.button("Process Query", type="primary") and customer_message:
            with st.spinner("Processing customer query..."):
                # Add customer message to history
                st.session_state.conversation_history.append({
                    "role": "customer",
                    "content": customer_message,
                    "timestamp": "now"
                })
                
                # Process query
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    response = loop.run_until_complete(
                        st.session_state.support_system.process_customer_query(
                            customer_message=customer_message,
                            conversation_history=st.session_state.conversation_history,
                            customer_id=customer_id,
                            domain=domain
                        )
                    )
                    
                    # Add response to history
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": response.get("response_text", "No response generated"),
                        "timestamp": "now"
                    })
                    
                    # Rerun to display new messages
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                finally:
                    loop.close()
    
    with col2:
        st.header("System Status")
        
        # Display current configuration
        st.subheader("Current Settings")
        st.write(f"**Domain:** {domain}")
        st.write(f"**Customer ID:** {customer_id}")
        st.write(f"**Conversation Length:** {len(st.session_state.conversation_history)} messages")
        
        # Display recent conversation metrics
        if st.session_state.conversation_history:
            st.subheader("Recent Metrics")
            
            # Get last response metrics
            last_response = None
            for message in reversed(st.session_state.conversation_history):
                if message["role"] == "assistant":
                    last_response = message
                    break
            
            if last_response:
                st.write(f"**Last Response:** {last_response['content'][:100]}...")
        
        # System information
        st.subheader("System Info")
        st.write("**Model:** Gemini 1.5 Pro")
        st.write("**Framework:** LangGraph")
        st.write("**Status:** Active")

if __name__ == "__main__":
    main()
