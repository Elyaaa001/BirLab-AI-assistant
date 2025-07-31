"""
BirLab AI

A Python framework for coordinating multiple AI agents to work together on complex tasks.
"""

from .core.agent import Agent
from .core.coordinator import CoordinatorAgent
from .core.task import Task, TaskResult, TaskPriority
from .core.message import Message
from .connectors.openai_connector import OpenAIConnector
from .connectors.anthropic_connector import AnthropicConnector
from .connectors.ollama_connector import OllamaConnector
from .connectors.google_connector import GoogleAIConnector
from .connectors.cohere_connector import CohereConnector
from .connectors.huggingface_connector import HuggingFaceConnector
from .connectors.mistral_connector import MistralConnector
from .connectors.together_connector import TogetherConnector
from .connectors.replicate_connector import ReplicateConnector
from .connectors.grok_connector import GrokConnector
from .connectors.perplexity_connector import PerplexityConnector  
from .connectors.ai21_connector import AI21Connector
from .connectors.groq_connector import GroqConnector
from .connectors.fireworks_connector import FireworksConnector

__version__ = "1.0.0"
__all__ = [
    # Core classes
    "Agent",
    "CoordinatorAgent", 
    "Task",
    "TaskResult",
    "TaskPriority",
    "Message",
    
    # AI Connectors - THE ULTIMATE COLLECTION! 🚀
    "OpenAIConnector",           # 🔥 GPT Models
    "AnthropicConnector",        # 🎨 Claude Models
    "OllamaConnector",           # 🏠 Local/Private Models
    "GoogleAIConnector",         # 🌟 Gemini/PaLM
    "CohereConnector",           # 💬 Command Models
    "HuggingFaceConnector",      # 🤗 Thousands of Models
    "MistralConnector",          # ⚡ European AI
    "TogetherConnector",         # 🤝 Open Source Models
    "ReplicateConnector",        # 🔄 Community Models
    "GrokConnector",             # 🤖 Rebellious AI (xAI)
    "PerplexityConnector",       # 🔍 Search-Enhanced AI
    "AI21Connector",             # 🧠 Jamba & Jurassic
    "GroqConnector",             # ⚡ Ultra-Fast LPU Inference
    "FireworksConnector"         # 🔥 Fast Open Source Serving
] 