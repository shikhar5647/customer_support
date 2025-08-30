# AI Customer Support System

An intelligent, AI-powered customer response system built with LangGraph agents and Gemini LLM, designed for BPO (Business Process Outsourcing) contact centers.

## 🚀 Features

- **Multi-Domain Support**: Handles e-commerce, telecom, and utility customer queries
- **LangGraph Agent Architecture**: Modular, scalable agent-based system
- **Gemini LLM Integration**: Powered by Google's Gemini 1.5 Pro model
- **SOP Compliance**: Follows Standard Operating Procedures for consistent responses
- **Tool Integration**: Seamlessly integrates with external APIs and systems
- **Quality Assessment**: Comprehensive metrics for response quality evaluation
- **Fallback Handling**: Robust error handling and human escalation
- **Streamlit Interface**: User-friendly web interface for testing and demonstration

## 🏗️ System Architecture

### Section A: Overall Architecture & Approach

The system follows a **LangGraph-based agent architecture** with the following components:

1. **Intent Classification Agent**: Identifies customer intent from natural language
2. **SOP Retrieval Agent**: Retrieves relevant Standard Operating Procedures
3. **Tool Orchestrator Agent**: Executes external tools and APIs
4. **Quality Checker Agent**: Validates response quality and compliance
5. **Customer Response Agent**: Orchestrates the entire workflow

**Model Choice**: Gemini 1.5 Pro for its:
- Strong reasoning capabilities
- Multimodal understanding
- Cost-effectiveness for production use
- Integration with Google's ecosystem

### Section B: Input Handling & Preprocessing

- **Customer Intent**: Derived from natural language processing and intent classification
- **SOPs**: Chunked, indexed, and embedded for efficient retrieval
- **Tools/APIs**: Metadata-driven tool execution with input/output validation
- **Context Preservation**: Maintains conversation context for 1 hour or 5000 tokens

### Section C: Prompt Design / Query Strategy

- **RAG-based approach** for SOP retrieval and context building
- **Structured prompts** with clear input/output schemas
- **Multi-turn conversation** support with memory management
- **Domain-specific prompting** for specialized knowledge

### Section D: Token Optimization Strategy

- **Dynamic context window** management
- **SOP compression** and summarization
- **Tool description optimization**
- **Context reuse** across conversation turns

### Section E: Failover & Fallback Plan

- **Multi-level fallback** strategies
- **Automatic retry** with exponential backoff
- **Human escalation** for complex issues
- **System monitoring** and alerting

### Section F: Model Quality Assessment

- **BLEU, ROUGE, and custom metrics**
- **Human evaluation** integration
- **Feedback loops** for continuous improvement
- **Regional language** support

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Gemini API key
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd customer_support
   ```

2. **Create virtual environment**
   ```bash
   python -m venv virtual_env
   source virtual_env/bin/activate  # On Windows: virtual_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp env.example .env
   # Edit .env with your Gemini API key
   ```

5. **Run the application**
   ```bash
   streamlit run main.py
   ```

## 📁 Project Structure

```
customer_support/
├── main.py                          # Main Streamlit application
├── config.yaml                      # Configuration file
├── requirements.txt                 # Python dependencies
├── env.example                      # Environment variables template
├── src/                            # Source code
│   ├── agents/                     # LangGraph agents
│   │   ├── base_agent.py          # Base agent class
│   │   ├── customer_response_agent.py  # Main orchestrator
│   │   ├── intent_classifier_agent.py  # Intent classification
│   │   ├── sop_retrieval_agent.py      # SOP retrieval
│   │   ├── tool_orchestrator_agent.py  # Tool execution
│   │   └── quality_checker_agent.py    # Quality validation
│   ├── sop_manager/                # SOP management
│   │   ├── sop_loader.py          # SOP loading and parsing
│   │   └── sop_chunker.py         # SOP text chunking
│   ├── processors/                 # Data processors
│   │   ├── context_manager.py     # Context management
│   │   └── intent_processor.py    # Intent processing
│   └── utils/                      # Utility modules
│       ├── config.py               # Configuration management
│       ├── tool_registry.py        # Tool registry
│       ├── sample_data.py          # Sample data for demo
│       ├── metrics.py              # Quality metrics
│       └── fallback_handler.py     # Fallback management
└── data/                           # Data directory
    └── sops/                       # SOP documents
```

## 🎯 Usage

### Starting the System

1. **Launch Streamlit Interface**
   ```bash
   streamlit run main.py
   ```

2. **Configure Settings**
   - Select business domain (ecommerce, telecom, utilities)
   - Set customer ID (optional)
   - Choose sample queries for testing

3. **Interact with the System**
   - Type customer queries or use sample queries
   - View generated responses and metrics
   - Monitor system performance

### Sample Queries

**E-commerce Domain:**
- "Where is my order?"
- "I want to return a product"
- "What's the status of my refund?"

**Telecom Domain:**
- "What's my current balance?"
- "I need to check my data usage"
- "How do I upgrade my plan?"

**Utilities Domain:**
- "What's my current bill amount?"
- "When is my bill due?"
- "I'm experiencing a power outage"

## 🔧 Configuration

### Environment Variables

- `GEMINI_API_KEY`: Your Gemini API key (required)
- `GEMINI_MODEL_NAME`: Model to use (default: gemini-1.5-pro)
- `CONFIDENCE_THRESHOLD`: Minimum confidence for responses (default: 0.7)

### Configuration File

Edit `config.yaml` to customize:
- Guardrails and safety settings
- Model parameters
- System thresholds
- Domain-specific configurations

## 📊 Quality Metrics

The system provides comprehensive quality assessment:

- **Readability**: Flesch Reading Ease score
- **Relevance**: Semantic similarity to customer query
- **Completeness**: Response structure and coverage
- **Professionalism**: Language quality and tone
- **Actionability**: Clear next steps and instructions

## 🚨 Fallback & Escalation

### Automatic Fallbacks

1. **Retry**: Automatic retry with exponential backoff
2. **Simplified Response**: Conservative, template-based responses
3. **Human Escalation**: Transfer to human agent when needed
4. **System Shutdown**: Emergency shutdown for critical failures

### Escalation Triggers

- Low confidence scores
- Model failures
- Tool execution errors
- Timeout violations
- Hallucination detection

## 🔌 API Integration

### Available Tools

- **Order Lookup**: E-commerce order status and details
- **Balance Check**: Telecom account balance and usage
- **Bill Inquiry**: Utility bill status and amounts
- **Customer Validation**: Identity verification

### Adding New Tools

1. Define tool metadata in `ToolRegistry`
2. Implement tool execution logic
3. Update domain configurations
4. Test integration

## 🧪 Testing & Development

### Running Tests

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run with coverage
python -m pytest --cov=src tests/
```

### Development Mode

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run linting
flake8 src/
black src/

# Run type checking
mypy src/
```

## 📈 Performance Monitoring

### Metrics Dashboard

- Response quality trends
- Tool usage statistics
- Fallback rates
- Escalation patterns

### Logging

- Structured logging with loguru
- Performance metrics collection
- Error tracking and alerting

## 🚀 Deployment

### Production Setup

1. **Environment Configuration**
   ```bash
   export GEMINI_API_KEY="your_production_key"
   export LOG_LEVEL="INFO"
   export ENABLE_METRICS="true"
   ```

2. **Service Management**
   ```bash
   # Using systemd
   sudo systemctl enable customer-support
   sudo systemctl start customer-support
   ```

3. **Monitoring**
   - Set up health checks
   - Configure alerting
   - Monitor resource usage

### Docker Deployment

```bash
# Build image
docker build -t customer-support .

# Run container
docker run -p 8501:8501 -e GEMINI_API_KEY=your_key customer-support
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Contact the development team

## 🔮 Roadmap

- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] Integration with CRM systems
- [ ] Voice interface support
- [ ] Advanced fallback strategies
- [ ] Performance optimization
- [ ] Extended tool ecosystem

---

**Built with ❤️ using LangGraph, Gemini, and Streamlit**
