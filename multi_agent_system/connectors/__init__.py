"""
AI Model Connectors

This module provides connectors for various AI services and local models.
"""

from .base_connector import AIModelConnector, AIAgent
from .openai_connector import OpenAIConnector, create_openai_agent
from .anthropic_connector import AnthropicConnector, create_anthropic_agent
from .ollama_connector import OllamaConnector, create_ollama_agent
from .google_connector import (
    GoogleAIConnector, create_google_agent,
    create_gemini_pro_agent, create_gemini_flash_agent,
    create_gemini_vision_agent, create_gemini_coder_agent, create_gemini_researcher_agent
)
from .cohere_connector import CohereConnector, create_cohere_agent
from .huggingface_connector import (
    HuggingFaceConnector, create_huggingface_agent,
    create_codegen_agent, create_flan_agent, create_dialogpt_agent
)
from .mistral_connector import MistralConnector, create_mistral_agent
from .together_connector import (
    TogetherConnector, create_together_agent,
    create_llama2_agent, create_codellama_together_agent, create_falcon_agent
)
from .replicate_connector import (
    ReplicateConnector, create_replicate_agent,
    create_llama2_replicate_agent, create_stable_diffusion_agent
)
from .grok_connector import GrokConnector, create_grok_agent
from .perplexity_connector import (
    PerplexityConnector, create_perplexity_agent,
    create_perplexity_research_agent, create_perplexity_news_agent
)
from .ai21_connector import (
    AI21Connector, create_ai21_agent,
    create_jamba_agent, create_jurassic_agent
)
from .groq_connector import (
    GroqConnector, create_groq_agent,
    create_groq_speed_demon, create_groq_powerhouse, create_groq_multilingual
)
from .fireworks_connector import (
    FireworksConnector, create_fireworks_agent,
    create_fireworks_llama_agent, create_fireworks_mixtral_agent,
    create_fireworks_code_agent, create_fireworks_math_agent
)

__all__ = [
    # Base classes
    "AIModelConnector",
    "AIAgent",
    
    # OpenAI
    "OpenAIConnector",
    "create_openai_agent",
    
    # Anthropic
    "AnthropicConnector", 
    "create_anthropic_agent",
    
    # Ollama (Local)
    "OllamaConnector",
    "create_ollama_agent",
    
    # Google AI - ENHANCED GEMINI! 🌟
    "GoogleAIConnector",
    "create_google_agent",
    "create_gemini_pro_agent",      # 🧠 2M context window!
    "create_gemini_flash_agent",    # ⚡ Lightning fast
    "create_gemini_vision_agent",   # 👁️ Multimodal vision expert
    "create_gemini_coder_agent",    # 💻 Programming specialist
    "create_gemini_researcher_agent", # 📚 Research powerhouse
    
    # Cohere
    "CohereConnector",
    "create_cohere_agent",
    
    # Hugging Face
    "HuggingFaceConnector",
    "create_huggingface_agent",
    "create_codegen_agent",
    "create_flan_agent", 
    "create_dialogpt_agent",
    
    # Mistral AI
    "MistralConnector",
    "create_mistral_agent",
    
    # Together AI
    "TogetherConnector",
    "create_together_agent",
    "create_llama2_agent",
    "create_codellama_together_agent",
    "create_falcon_agent",
    
    # Replicate
    "ReplicateConnector",
    "create_replicate_agent", 
    "create_llama2_replicate_agent",
    "create_stable_diffusion_agent",
    
    # Grok (xAI)
    "GrokConnector",
    "create_grok_agent",
    
    # Perplexity AI
    "PerplexityConnector",
    "create_perplexity_agent",
    "create_perplexity_research_agent",
    "create_perplexity_news_agent",
    
    # AI21 Labs
    "AI21Connector",
    "create_ai21_agent",
    "create_jamba_agent",
    "create_jurassic_agent",
    
    # Groq (Super Fast!)
    "GroqConnector",
    "create_groq_agent",
    "create_groq_speed_demon",
    "create_groq_powerhouse",
    "create_groq_multilingual",
    
    # Fireworks AI
    "FireworksConnector",
    "create_fireworks_agent",
    "create_fireworks_llama_agent",
    "create_fireworks_mixtral_agent",
    "create_fireworks_code_agent",
    "create_fireworks_math_agent"
] 