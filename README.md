# BirLab AI

BirLab AI - The ultimate Python framework with **100+ AI models** that enables massive multi-agent coordination for complex tasks through intelligent selection and delegation.

## 🚀 NEW: 100+ AI MODELS ARMY!

BirLab AI now supports **100+ different AI models** from 8+ providers with intelligent auto-registration and task-specific agent selection!

### 📊 **Model Coverage:**
- **OpenAI**: 15 models (GPT-4 Turbo, GPT-3.5, Codex, DALL-E, Whisper)
- **Anthropic**: 8 models (Claude-3 Opus/Sonnet/Haiku, Claude-2.1)
- **Google AI**: 12 models (Gemini 1.5 Pro/Flash, PaLM 2, Imagen)
- **Cohere**: 6 models (Command-R+, Command Light/Nightly)
- **Mistral AI**: 8 models (Large/Medium/Small, Mixtral 8x7B/8x22B)
- **Hugging Face**: 15 models (Llama 2, Falcon, Vicuna, Code Llama)
- **Together AI**: 10 models (Optimized inference for popular models)
- **Ollama**: 20 models (Local privacy-focused models)

### 🎯 **Intelligent Features:**
- 🧠 **Smart Agent Selection** - Automatically picks the best model for each task
- ⚖️ **Load Balancing** - Distributes tasks acr oss multiple providers
- 🎛️ **Capability Matching** - Routes coding/multimodal/analysis tasks to specialized agents
- 📊 **Performance Analytics** - Real-time stats on model usage and performance
- 🔒 **Privacy Options** - Local models available via Ollama (no API keys needed)

## 🌟 Features

- **Modular Architecture**: Clean, extensible design with clear separation of concerns
- **MASSIVE AI Provider Support**: Connect to **14 different AI services** with 50+ models!
  - 🔥 **OpenAI** (GPT-3.5, GPT-4)
  - 🎨 **Anthropic** (Claude-3 Opus/Sonnet/Haiku)
  - 🌟 **Google AI** (Gemini Pro, PaLM)
  - 💬 **Cohere** (Command models)
  - 🤗 **Hugging Face** (Thousands of models via Inference API)
  - ⚡ **Mistral AI** (Mistral, Mixtral models)
  - 🤝 **Together AI** (Open-source models)
  - 🔄 **Replicate** (Community models)
  - 🤖 **Grok** (xAI - Rebellious AI with real-time data)
  - 🔍 **Perplexity AI** (Search-enhanced responses)
  - 🧠 **AI21 Labs** (Jamba, Jurassic models)
  - ⚡ **Groq** (Ultra-fast LPU inference)
  - 🔥 **Fireworks AI** (Fast open-source serving)
  - 🏠 **Ollama** (Local/private models)
- **Intelligent Coordination**: Main controller agent that delegates tasks to the most suitable specialized agents
- **Task Decomposition**: Break complex tasks into manageable subtasks with dependency management
- **Async by Design**: Built with asyncio for high performance and concurrent task execution
- **Comprehensive Monitoring**: Real-time status tracking and performance metrics
- **Easy Integration**: Simple APIs for adding new agents and capabilities

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  CoordinatorAgent                       │
│  (Main controller that delegates tasks)                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   OpenAI    │  │  Anthropic  │  │   Ollama    │     │
│  │    Agent    │  │    Agent    │  │    Agent    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                                         │
├─────────────────────────────────────────────────────────┤
│              Task & Message System                      │
├─────────────────────────────────────────────────────────┤
│            AI Model Connectors                          │
└─────────────────────────────────────────────────────────┘
```

## 🌐 **FULL-STACK WEB APPLICATION**

### 🚀 **One-Command Deployment**

Deploy both frontend and backend with a single command:

```bash
# Interactive deployment (recommended)
./deploy.sh

# Quick Docker deployment
docker-compose up --build

# Manual deployment
export GOOGLE_AI_API_KEY="your-key"
python backend/fastapi_server.py &  # Backend at :8000
cd frontend && npm start &          # Frontend at :3000
```

### **🎨 Frontend Features**
- **💬 Chat Interface**: Real-time chat with any AI agent
- **🎯 Task Management**: Execute complex multi-step tasks
- **👁️ Vision Analysis**: Upload and analyze images with Gemini Vision
- **🤖 Agent Dashboard**: Monitor all 14 AI providers
- **📊 Real-time Metrics**: System status and performance
- **📱 Responsive Design**: Works on desktop, tablet, and mobile

### **🌐 Backend API**
- **REST API**: All agents accessible via HTTP endpoints
- **WebSocket**: Real-time chat connections
- **Multimodal**: File upload for image analysis
- **OpenAPI Docs**: Interactive API documentation at `/docs`
- **Health Monitoring**: System status and metrics
- **Production Ready**: FastAPI with async support

**📚 Complete guide:** [`FULLSTACK_GUIDE.md`](FULLSTACK_GUIDE.md)

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd BirLab-AI-assistant
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. **🚀 Quick Setup (Recommended)**:
```bash
python setup_env.py  # Interactive setup helper
```

4. **OR** Set up environment variables for the AI services you want to use:
```bash
# OpenAI (Required for GPT models)
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic (Required for Claude models)  
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Google AI (Required for Gemini/PaLM models)
export GOOGLE_AI_API_KEY="your-google-ai-api-key"

# Cohere (Required for Command models)
export COHERE_API_KEY="your-cohere-api-key"

# Hugging Face (Required for HF models)
export HUGGINGFACE_API_KEY="your-huggingface-api-key"

# Mistral AI (Required for Mistral models)
export MISTRAL_API_KEY="your-mistral-api-key"

# Together AI (Required for Together models)
export TOGETHER_API_KEY="your-together-api-key"

# Replicate (Required for Replicate models)
export REPLICATE_API_TOKEN="your-replicate-token"

# Grok (Required for xAI models)
export GROK_API_KEY="your-grok-api-key"

# And more... see CONFIG_GUIDE.md for complete list
```

📚 **For detailed configuration help, see:** [`CONFIG_GUIDE.md`](CONFIG_GUIDE.md)

**🌟 To get started with just Gemini:**
```bash
export GOOGLE_AI_API_KEY="your-key"  # Get from: https://aistudio.google.com/app/apikey
python examples/gemini_showcase.py
```

**Complete Environment Variables:**
```bash
# Perplexity AI (Required for search-enhanced responses)
export PERPLEXITY_API_KEY="your-perplexity-key"

# AI21 Labs (Required for Jamba/Jurassic models)
export AI21_API_KEY="your-ai21-key"

# Groq (Required for ultra-fast inference)
export GROQ_API_KEY="your-groq-key"

# Fireworks AI (Required for fast open-source serving)
export FIREWORKS_API_KEY="your-fireworks-key"
```

4. For local models, ensure Ollama is running:
```bash
# Install Ollama first: https://ollama.ai
ollama serve
ollama pull llama2:7b  # or other models you want to use
```

### Basic Usage

```python
import asyncio
from multi_agent_system import CoordinatorAgent, Task, TaskPriority
from multi_agent_system.connectors import create_openai_agent, create_anthropic_agent

async def main():
    # Create coordinator
    coordinator = CoordinatorAgent(name="MainCoordinator")
    
    # Create and register agents
    gpt_agent = create_openai_agent(
        model_name="gpt-3.5-turbo",
        capabilities=["text_generation", "analysis"]
    )
    coordinator.register_agent(gpt_agent)
    
    # Create a task
    task = Task(
        description="Analyze the benefits of renewable energy",
        task_type="analysis",
        priority=TaskPriority.HIGH
    )
    
    # Execute task
    result = await coordinator.start_task(task)
    print(f"Result: {result.result}")
    
    # Cleanup
    await coordinator.shutdown()

asyncio.run(main())
```

## 🤖 AI Provider Gallery

### All Supported AI Services

```python
from multi_agent_system.connectors import *

# 🔥 OpenAI - GPT Models
gpt4_agent = create_openai_agent("gpt-4", api_key="...", capabilities=["reasoning", "analysis"])
gpt35_agent = create_openai_agent("gpt-3.5-turbo", api_key="...", capabilities=["conversation"])

# 🎨 Anthropic - Claude Models  
claude_opus = create_anthropic_agent("claude-3-opus-20240229", api_key="...", capabilities=["research", "analysis"])
claude_sonnet = create_anthropic_agent("claude-3-sonnet-20240229", api_key="...", capabilities=["reasoning"])

# 🌟 Google AI - ENHANCED GEMINI MODELS!
# Gemini 1.5 Pro - 2 MILLION token context window!
gemini_pro = create_gemini_pro_agent(api_key="...")  # Ultra-long context + multimodal
gemini_flash = create_gemini_flash_agent(api_key="...")  # Lightning-fast responses
gemini_vision = create_gemini_vision_agent(api_key="...")  # Multimodal vision expert
gemini_coder = create_gemini_coder_agent(api_key="...")  # Programming specialist
gemini_researcher = create_gemini_researcher_agent(api_key="...")  # Research powerhouse

# 💬 Cohere - Command Models
cohere_agent = create_cohere_agent("command", api_key="...", capabilities=["conversation", "analysis"])

# 🤗 Hugging Face - Thousands of Models
flan_agent = create_flan_agent(api_key="...")  # Instruction-following
codegen_agent = create_codegen_agent(api_key="...")  # Code generation
hf_llama = create_huggingface_agent("meta-llama/Llama-2-7b-chat-hf", api_key="...")

# ⚡ Mistral AI - European Models
mistral_agent = create_mistral_agent("mistral-medium", api_key="...", capabilities=["multilingual"])

# 🤝 Together AI - Open Source Models
llama2_together = create_llama2_agent("13b", api_key="...")  # Llama via Together
falcon_agent = create_falcon_agent("7b", api_key="...")  # Falcon via Together

# 🔄 Replicate - Community Models
llama2_replicate = create_llama2_replicate_agent("7b", api_key="...")
stable_diffusion = create_stable_diffusion_agent("xl", api_key="...")  # Image generation!

# 🤖 Grok - Rebellious AI from xAI
grok_agent = create_grok_agent("grok-beta", api_key="...", capabilities=["humor", "real_time_info"])

# 🔍 Perplexity AI - Search-Enhanced Responses
perplexity_agent = create_perplexity_agent("llama-3.1-sonar-large-128k-online", api_key="...")
perplexity_research = create_perplexity_research_agent(api_key="...")  # Academic research

# 🧠 AI21 Labs - Jamba & Jurassic Models
jamba_agent = create_jamba_agent(api_key="...")  # 256k context window!
jurassic_agent = create_jurassic_agent("ultra", api_key="...")  # Creative writing

# ⚡ Groq - Ultra-Fast LPU Inference
groq_speed = create_groq_speed_demon(api_key="...")  # Fastest responses
groq_power = create_groq_powerhouse(api_key="...")  # 70B model at blazing speed
groq_multilingual = create_groq_multilingual(api_key="...")  # Mixtral MoE

# 🔥 Fireworks AI - Fast Open Source Serving
fireworks_llama = create_fireworks_llama_agent("405b", api_key="...")  # Largest Llama!
fireworks_code = create_fireworks_code_agent(api_key="...")  # DeepSeek Coder
fireworks_math = create_fireworks_math_agent(api_key="...")  # Qwen Math

# 🏠 Ollama - Local/Private Models
local_llama = create_ollama_agent("llama2:7b")  # Runs locally
local_codellama = create_ollama_agent("codellama:13b")
local_mistral = create_ollama_agent("mistral:7b")
```

## 📚 Examples

### Basic Usage
Run the basic example to see simple task delegation:
```bash
python examples/basic_usage.py
```

### Advanced Workflow
See complex task decomposition and multi-agent coordination:
```bash
python examples/advanced_workflow.py
```

### 🤖 MEGA AI ARMY (Epic Demo!)
Experience ALL AI providers working together in an epic battle royale:
```bash
python examples/mega_ai_army.py
```
This demonstrates agents from ALL 14 providers (OpenAI, Anthropic, Google, Cohere, Hugging Face, Mistral, Together, Replicate, Grok, Perplexity, AI21, Groq, Fireworks, and Ollama) all coordinating together!

### 🚀 100+ AI MODELS ARMY DEMO!
Experience the massive BirLab AI system with intelligent coordination of 100+ models:
```bash
python examples/birlab_ai_army_demo.py
```
This demonstrates:
- 🤖 **100+ AI Models**: Automatic registration from 8+ providers (OpenAI, Anthropic, Google, Cohere, Mistral, Hugging Face, Together, Ollama)
- 🧠 **Intelligent Selection**: Smart routing to the best model for each task type
- ⚖️ **Load Balancing**: Concurrent task distribution across multiple agents
- 🎯 **Specialized Routing**: Coding → CodeLlama, Vision → Gemini, Analysis → Claude, etc.
- 📊 **Performance Analytics**: Real-time stats, provider comparison, and capability mapping
- 🔒 **Privacy Options**: Local models (Ollama) alongside cloud providers

### 🌟 GEMINI SHOWCASE (Google AI Demo!)
Experience the full power of Google's Gemini models:
```bash
python examples/gemini_showcase.py
```
This demonstrates:
- 🧠 **Gemini 1.5 Pro**: 2 million token context window for ultra-long documents
- ⚡ **Gemini 1.5 Flash**: Lightning-fast responses with multimodal capabilities  
- 👁️ **Vision Analysis**: Image understanding and multimodal reasoning
- 💻 **Code Generation**: Advanced programming and architecture design
- 📚 **Research Synthesis**: Academic-level analysis and document processing

## 🔧 Core Components

### CoordinatorAgent
The main controller that:
- Manages registered agents
- Delegates tasks to appropriate agents
- Handles task decomposition and workflows
- Monitors system performance

### AI Agents
Specialized agents that:
- Connect to different AI services
- Handle specific task types
- Maintain conversation context
- Report performance metrics

### Task Management
- **Task**: Represents work to be done
- **TaskResult**: Contains execution results and metadata
- **Dependencies**: Support for complex workflows

### Connectors
- **OpenAI**: GPT-3.5, GPT-4 via OpenAI API
- **Anthropic**: Claude-3 Opus/Sonnet/Haiku via Anthropic API  
- **Google AI**: Gemini Pro, PaLM via Google AI API
- **Cohere**: Command models via Cohere API
- **Hugging Face**: Thousands of models via Inference API
- **Mistral AI**: Mistral, Mixtral models via Mistral API
- **Together AI**: Open-source models (Llama, Falcon, etc.)
- **Replicate**: Community models with async prediction handling
- **Grok**: xAI's rebellious AI with real-time X/Twitter data
- **Perplexity AI**: Search-enhanced responses with citations
- **AI21 Labs**: Jamba (256k context) & Jurassic models
- **Groq**: Ultra-fast LPU inference for lightning speed
- **Fireworks AI**: Optimized open-source model serving
- **Ollama**: Local models via Ollama server (privacy-first)

## 🎯 Use Cases

1. **Content Creation Pipeline**
   - Research → Analysis → Writing → Review
   - Each step handled by specialized agents

2. **Software Development**
   - Requirements → Design → Code → Tests → Documentation
   - Automated project workflow

3. **Research & Analysis**
   - Data gathering → Analysis → Report generation
   - Comprehensive research automation

4. **Customer Support**
   - Intent analysis → Response generation → Quality check
   - Multi-step customer interaction

## 🛠️ Extension Guide

### Adding New AI Connectors

1. Create a new connector class inheriting from `AIModelConnector`:

```python
from multi_agent_system.connectors.base_connector import AIModelConnector

class CustomConnector(AIModelConnector):
    async def generate_response(self, prompt: str, **kwargs) -> str:
        # Implement your AI service integration
        pass
    
    async def validate_connection(self) -> bool:
        # Implement connection validation
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        # Return model information
        pass
```

2. Create a convenience function:

```python
def create_custom_agent(model_name: str, **config):
    connector = CustomConnector(model_name, **config)
    return AIAgent(connector, capabilities, name=f"Custom_{model_name}")
```

### Custom Task Decomposition

```python
async def custom_decomposition_strategy(main_task: Task) -> List[Task]:
    # Break down the main task into subtasks
    subtasks = [
        Task(description="Subtask 1", task_type="analysis"),
        Task(description="Subtask 2", task_type="generation", 
             dependencies=[subtasks[0].task_id])
    ]
    return subtasks

# Register with coordinator
coordinator.register_decomposition_strategy("custom_task_type", custom_decomposition_strategy)
```

## 📊 Monitoring & Debugging

### System Status
```python
status = coordinator.get_system_status()
print(f"Active agents: {status['active_agents']}")
print(f"Completed tasks: {status['completed_tasks']}")
```

### Agent Performance
```python
for agent_id, agent_status in status['agent_statuses'].items():
    print(f"Agent: {agent_status['name']}")
    print(f"Tasks completed: {agent_status['tasks_completed']}")
    print(f"Success rate: {agent_status['tasks_completed'] / max(agent_status['tasks_completed'] + agent_status['tasks_failed'], 1) * 100:.1f}%")
```

### Logging
The system uses Python's logging module. Configure as needed:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 🔒 Security & Best Practices

1. **API Key Management**: Store API keys in environment variables, never in code
2. **Rate Limiting**: Respect API rate limits for external services
3. **Error Handling**: Implement proper error handling for network failures
4. **Resource Management**: Use async context managers for cleanup
5. **Monitoring**: Monitor agent performance and costs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with inspiration from multi-agent systems research
- Thanks to the teams at OpenAI, Anthropic, and Ollama for their APIs
- Community feedback and contributions

## 📞 Support

- Create an issue for bugs or feature requests
- Check the examples for common usage patterns
- Review the code documentation for detailed API reference

---

**Happy multi-agent coordinating! 🤖✨**
