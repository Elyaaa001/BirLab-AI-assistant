#!/usr/bin/env python3
"""
🌟 BirLab AI - Ultra Coordinator
Manages 100+ AI Models with Intelligent Auto-Registration

This coordinator automatically detects available API keys and registers
all compatible AI models, creating a massive multi-agent army.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

from .coordinator import CoordinatorAgent
from ..connectors.expanded_models import (
    BIRLAB_AI_MODELS, 
    create_birlab_ai_agent,
    get_models_by_provider,
    get_provider_count,
    print_birlab_ai_registry
)

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class BirLabUltraCoordinator(CoordinatorAgent):
    """
    🚀 Ultra Coordinator for 100+ AI Models
    
    Automatically registers all available AI models based on API keys.
    Provides intelligent agent selection and load balancing.
    """
    
    def __init__(self):
        super().__init__()
        self.registered_models: Dict[str, Any] = {}
        self.provider_stats: Dict[str, int] = {}
        self.total_agents = 0
        
    def auto_register_all_agents(self) -> Dict[str, List[str]]:
        """
        🤖 AUTO-REGISTER ALL AVAILABLE AI MODELS
        
        Detects API keys and registers all compatible models automatically.
        Returns a report of registered agents by provider.
        """
        logger.info("🚀 Starting BirLab AI Ultra Registration...")
        
        # API key detection
        api_keys = self._detect_api_keys()
        registration_report = {}
        total_registered = 0
        
        # Register models by provider
        for provider in ["openai", "anthropic", "google", "cohere", "mistral", 
                        "huggingface", "together", "ollama"]:
            
            registered_models = self._register_provider_models(provider, api_keys)
            if registered_models:
                registration_report[provider] = registered_models
                total_registered += len(registered_models)
        
        self.total_agents = total_registered
        self.provider_stats = get_provider_count()
        
        logger.info(f"✅ BirLab AI Registration Complete!")
        logger.info(f"📊 Total Agents Registered: {total_registered}")
        logger.info(f"🏢 Active Providers: {len(registration_report)}")
        
        return registration_report
    
    def _detect_api_keys(self) -> Dict[str, str]:
        """Detect available API keys from environment"""
        api_keys = {}
        
        # Common API key patterns
        key_mappings = {
            "openai": ["OPENAI_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
            "google": ["GOOGLE_AI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"],
            "cohere": ["COHERE_API_KEY"],
            "mistral": ["MISTRAL_API_KEY"],
            "huggingface": ["HUGGINGFACE_API_KEY", "HF_TOKEN", "HUGGINGFACE_TOKEN"],
            "together": ["TOGETHER_API_KEY"],
            "replicate": ["REPLICATE_API_TOKEN"],
            "groq": ["GROQ_API_KEY"],
            "fireworks": ["FIREWORKS_API_KEY"],
            "perplexity": ["PERPLEXITY_API_KEY"],
            "ai21": ["AI21_API_KEY"],
            "grok": ["GROK_API_KEY", "XAI_API_KEY"],
            "ollama": ["OLLAMA_HOST"]  # Ollama doesn't need API key
        }
        
        for provider, env_vars in key_mappings.items():
            for env_var in env_vars:
                key = os.getenv(env_var)
                if key:
                    api_keys[provider] = key
                    logger.info(f"🔑 Found API key for {provider}")
                    break
        
        # Ollama is always available (local)
        if "ollama" not in api_keys:
            api_keys["ollama"] = "local"  # Placeholder for local models
        
        return api_keys
    
    def _register_provider_models(self, provider: str, api_keys: Dict[str, str]) -> List[str]:
        """Register all models for a specific provider"""
        if provider not in api_keys and provider != "ollama":
            logger.warning(f"⚠️ No API key found for {provider}")
            return []
        
        provider_models = get_models_by_provider(provider)
        registered = []
        
        for model_id in provider_models:
            try:
                # Create agent
                api_key = api_keys.get(provider) if provider != "ollama" else None
                agent = create_birlab_ai_agent(model_id, api_key)
                
                # Register with coordinator
                model_info = BIRLAB_AI_MODELS[model_id]
                agent_id = f"birlab_{model_id}"
                
                # Set agent attributes to match expected interface
                agent.agent_id = agent_id
                agent.name = model_info['name']
                if not hasattr(agent, 'capabilities'):
                    agent.capabilities = model_info.get("capabilities", [])
                
                # Register with parent coordinator (simplified call)
                self.register_agent(agent)
                
                registered.append(model_id)
                self.registered_models[model_id] = {
                    "agent_id": agent_id,
                    "agent": agent,
                    "model_info": model_info
                }
                
                logger.debug(f"✅ Registered {model_info['name']}")
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to register {model_id}: {e}")
        
        logger.info(f"🎯 {provider.title()}: {len(registered)}/{len(provider_models)} models registered")
        return registered
    
    def _calculate_priority_boost(self, model_info: Dict[str, Any]) -> float:
        """Calculate priority boost based on model capabilities"""
        base_priority = 1.0
        
        # Boost for specific capabilities
        capabilities = model_info.get("capabilities", [])
        context_length = model_info.get("context_length", 0)
        
        # Context length boost
        if context_length > 100000:
            base_priority += 0.3  # Ultra long context
        elif context_length > 30000:
            base_priority += 0.2  # Long context
        elif context_length > 8000:
            base_priority += 0.1  # Medium context
        
        # Capability boosts
        capability_boosts = {
            "reasoning": 0.2,
            "multimodal": 0.2,
            "code_generation": 0.15,
            "fast_responses": 0.1,
            "advanced_reasoning": 0.25,
            "expert_tasks": 0.25,
            "local_inference": 0.05  # Slightly lower for privacy models
        }
        
        for capability in capabilities:
            if capability in capability_boosts:
                base_priority += capability_boosts[capability]
        
        return min(base_priority, 2.0)  # Cap at 2.0
    
    def get_best_agent_for_task(self, task_type: str, **constraints) -> Optional[str]:
        """
        🎯 INTELLIGENT AGENT SELECTION
        
        Selects the best agent based on task requirements and constraints.
        """
        # Define task type to capability mapping
        task_mappings = {
            "coding": ["code_generation", "programming"],
            "analysis": ["reasoning", "analysis", "advanced_reasoning"],
            "multimodal": ["multimodal", "vision", "image_analysis"],
            "creative": ["creative_writing", "art", "creativity"],
            "fast": ["fast_responses", "efficient"],
            "research": ["reasoning", "analysis", "retrieval"],
            "local": ["local_inference", "privacy"],
            "long_context": ["long_context", "ultra_long_context"]
        }
        
        required_capabilities = task_mappings.get(task_type, [])
        
        # Filter agents by capabilities
        compatible_agents = []
        for model_id, model_data in self.registered_models.items():
            agent_capabilities = model_data["model_info"].get("capabilities", [])
            
            # Check if agent has required capabilities
            if not required_capabilities or any(cap in agent_capabilities for cap in required_capabilities):
                compatible_agents.append((model_id, model_data))
        
        if not compatible_agents:
            logger.warning(f"No compatible agents found for task type: {task_type}")
            return None
        
        # Sort by priority and context length
        def agent_score(item):
            model_id, model_data = item
            model_info = model_data["model_info"]
            
            score = 0
            
            # Capability match bonus
            agent_capabilities = model_info.get("capabilities", [])
            matches = sum(1 for cap in required_capabilities if cap in agent_capabilities)
            score += matches * 10
            
            # Context length bonus
            context_length = model_info.get("context_length", 0)
            score += min(context_length / 1000, 50)  # Up to 50 points for context
            
            # Apply constraints
            if constraints.get("prefer_local") and "local_inference" in agent_capabilities:
                score += 20
            
            if constraints.get("prefer_fast") and "fast_responses" in agent_capabilities:
                score += 15
            
            return score
        
        # Get best agent
        best_agent = max(compatible_agents, key=agent_score)
        return best_agent[1]["agent_id"]
    
    def get_agent_army_stats(self) -> Dict[str, Any]:
        """Get comprehensive stats about the registered agent army"""
        provider_counts = {}
        capability_counts = {}
        context_distribution = {"small": 0, "medium": 0, "large": 0, "ultra": 0}
        
        for model_data in self.registered_models.values():
            model_info = model_data["model_info"]
            provider = model_info["provider"]
            
            # Provider stats
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
            
            # Capability stats
            for capability in model_info.get("capabilities", []):
                capability_counts[capability] = capability_counts.get(capability, 0) + 1
            
            # Context length distribution
            context_length = model_info.get("context_length", 0)
            if context_length > 100000:
                context_distribution["ultra"] += 1
            elif context_length > 30000:
                context_distribution["large"] += 1
            elif context_length > 4000:
                context_distribution["medium"] += 1
            else:
                context_distribution["small"] += 1
        
        return {
            "total_agents": len(self.registered_models),
            "providers": provider_counts,
            "capabilities": capability_counts,
            "context_distribution": context_distribution,
            "top_capabilities": sorted(capability_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    def print_agent_army_status(self):
        """Print a comprehensive status of the BirLab AI agent army"""
        stats = self.get_agent_army_stats()
        
        print("\n" + "=" * 60)
        print("🌟 BIRLAB AI - AGENT ARMY STATUS 🌟")
        print("=" * 60)
        
        print(f"🤖 Total AI Agents: {stats['total_agents']}")
        print(f"🏢 Active Providers: {len(stats['providers'])}")
        
        print(f"\n📊 PROVIDERS:")
        for provider, count in sorted(stats['providers'].items()):
            print(f"  • {provider.title()}: {count} models")
        
        print(f"\n🎯 TOP CAPABILITIES:")
        for capability, count in stats['top_capabilities'][:5]:
            print(f"  • {capability}: {count} agents")
        
        print(f"\n📖 CONTEXT LENGTH DISTRIBUTION:")
        for size, count in stats['context_distribution'].items():
            print(f"  • {size.title()}: {count} agents")
        
        print(f"\n🚀 READY FOR MAXIMUM AI COORDINATION! 🚀")
        print("=" * 60)


def create_birlab_ultra_coordinator() -> BirLabUltraCoordinator:
    """
    🌟 CREATE BIRLAB AI ULTRA COORDINATOR
    
    Automatically sets up and registers 100+ AI models.
    """
    coordinator = BirLabUltraCoordinator()
    
    print("🚀 Initializing BirLab AI Ultra Coordinator...")
    print_birlab_ai_registry()
    
    # Auto-register all available agents
    registration_report = coordinator.auto_register_all_agents()
    
    # Print status
    coordinator.print_agent_army_status()
    
    return coordinator


if __name__ == "__main__":
    # Demo the ultra coordinator
    coordinator = create_birlab_ultra_coordinator()
    
    # Test intelligent agent selection
    print("\n🎯 TESTING INTELLIGENT AGENT SELECTION:")
    
    test_tasks = ["coding", "analysis", "multimodal", "fast", "local"]
    for task in test_tasks:
        best_agent = coordinator.get_best_agent_for_task(task)
        if best_agent:
            print(f"  • Best for {task}: {best_agent}")
        else:
            print(f"  • No agent found for {task}") 