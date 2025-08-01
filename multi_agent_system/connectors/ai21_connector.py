import asyncio
import json
from typing import Dict, Any, Optional
import aiohttp
from .base_connector import AIModelConnector


class AI21Connector(AIModelConnector):
    """
    Connector for AI21 Labs models (Jurassic series).
    """
    
    def __init__(self, model_name: str = "jamba-1.5-large", api_key: Optional[str] = None,
                 base_url: str = "https://api.ai21.com/studio/v1", **config):
        super().__init__(model_name, config)
        self.api_key = api_key or config.get("api_key")
        self.base_url = base_url.rstrip('/')
        
        if not self.api_key:
            raise ValueError("AI21 API key is required")
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using AI21's API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response string
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Use chat format for newer models, completion for older ones
        if self._is_chat_model():
            return await self._generate_chat_response(prompt, headers, **kwargs)
        else:
            return await self._generate_completion_response(prompt, headers, **kwargs)
    
    async def _generate_chat_response(self, prompt: str, headers: dict, **kwargs) -> str:
        """Generate response using chat completions endpoint"""
        messages = [{"role": "user", "content": prompt}]
        
        # Add system message if provided
        if "system_prompt" in kwargs:
            messages.insert(0, {"role": "system", "content": kwargs["system_prompt"]})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 1.0),
            "stream": False
        }
        
        # Add stop sequences if provided
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"AI21 API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    if "choices" not in result or not result["choices"]:
                        raise Exception("No response generated by AI21")
                    
                    return result["choices"][0]["message"]["content"]
                    
            except aiohttp.ClientError as e:
                raise Exception(f"Network error communicating with AI21: {str(e)}")
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse AI21 response: {str(e)}")
    
    async def _generate_completion_response(self, prompt: str, headers: dict, **kwargs) -> str:
        """Generate response using completions endpoint for older models"""
        payload = {
            "prompt": prompt,
            "maxTokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "topP": kwargs.get("top_p", 1.0),
            "topKReturn": kwargs.get("top_k_return", 0),
            "frequencyPenalty": {
                "scale": kwargs.get("frequency_penalty", 0),
                "applyToWhitespaces": kwargs.get("penalty_whitespace", True),
                "applyToPunctuations": kwargs.get("penalty_punctuation", True),
                "applyToNumbers": kwargs.get("penalty_numbers", True),
                "applyToStopwords": kwargs.get("penalty_stopwords", False),
                "applyToEmojis": kwargs.get("penalty_emojis", True)
            },
            "presencePenalty": {
                "scale": kwargs.get("presence_penalty", 0),
                "applyToWhitespaces": kwargs.get("presence_whitespace", True),
                "applyToPunctuations": kwargs.get("presence_punctuation", True),
                "applyToNumbers": kwargs.get("presence_numbers", True),
                "applyToStopwords": kwargs.get("presence_stopwords", False),
                "applyToEmojis": kwargs.get("presence_emojis", True)
            },
            "countPenalty": {
                "scale": kwargs.get("count_penalty", 0),
                "applyToWhitespaces": kwargs.get("count_whitespace", True),
                "applyToPunctuations": kwargs.get("count_punctuation", True),
                "applyToNumbers": kwargs.get("count_numbers", True),
                "applyToStopwords": kwargs.get("count_stopwords", False),
                "applyToEmojis": kwargs.get("count_emojis", True)
            }
        }
        
        # Add stop sequences if provided
        if "stop" in kwargs:
            payload["stopSequences"] = kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
        
        # Determine endpoint based on model
        endpoint = f"/{self.model_name}/complete"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}{endpoint}",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"AI21 API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    if "completions" not in result or not result["completions"]:
                        raise Exception("No response generated by AI21")
                    
                    return result["completions"][0]["data"]["text"].strip()
                    
            except aiohttp.ClientError as e:
                raise Exception(f"Network error communicating with AI21: {str(e)}")
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse AI21 response: {str(e)}")
    
    def _is_chat_model(self) -> bool:
        """Check if this is a chat-capable model"""
        chat_models = ["jamba", "jamba-1.5"]
        return any(model.lower() in self.model_name.lower() for model in chat_models)
    
    async def validate_connection(self) -> bool:
        """Validate connection to AI21 API"""
        try:
            response = await self.generate_response(
                "Hello",
                max_tokens=5,
                temperature=0
            )
            return len(response.strip()) > 0
        except Exception as e:
            self.logger.error(f"Connection validation failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the AI21 model"""
        model_info = {
            "provider": "AI21 Labs",
            "model_name": self.model_name,
            "base_url": self.base_url
        }
        
        # Add model-specific information
        if "jamba" in self.model_name.lower():
            model_info.update({
                "model_family": "Jamba",
                "architecture": "Mamba-Transformer hybrid",
                "capabilities": [
                    "text_generation",
                    "long_context",
                    "reasoning",
                    "multilingual",
                    "conversation",
                    "structured_output"
                ],
                "special_features": [
                    "Hybrid Mamba-Transformer architecture",
                    "Extremely long context window",
                    "Memory efficient",
                    "Fast inference"
                ]
            })
            
            if "1.5" in self.model_name:
                model_info.update({
                    "context_length": 256000,
                    "version": "1.5",
                    "improvements": ["Better reasoning", "Enhanced multilingual", "Improved structured output"]
                })
            else:
                model_info.update({
                    "context_length": 256000,
                    "version": "1.0"
                })
                
        elif "jurassic" in self.model_name.lower():
            model_info.update({
                "model_family": "Jurassic",
                "architecture": "Transformer",
                "capabilities": [
                    "text_generation",
                    "completion",
                    "creative_writing",
                    "summarization"
                ]
            })
            
            if "ultra" in self.model_name.lower():
                model_info.update({
                    "size": "178B parameters",
                    "context_length": 8192,
                    "tier": "Ultra (most capable)"
                })
            elif "mid" in self.model_name.lower():
                model_info.update({
                    "size": "7.5B parameters", 
                    "context_length": 8192,
                    "tier": "Mid (balanced)"
                })
            elif "light" in self.model_name.lower():
                model_info.update({
                    "size": "7.5B parameters",
                    "context_length": 8192,
                    "tier": "Light (fast)"
                })
        
        return model_info
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models from AI21"""
        # AI21 doesn't provide a models endpoint, so return known models
        known_models = [
            {
                "name": "jamba-1.5-large",
                "family": "Jamba",
                "version": "1.5",
                "context_length": 256000,
                "capabilities": ["chat", "long_context", "reasoning"]
            },
            {
                "name": "jamba-1.5-mini", 
                "family": "Jamba",
                "version": "1.5",
                "context_length": 256000,
                "capabilities": ["chat", "long_context", "fast_inference"]
            },
            {
                "name": "j2-ultra",
                "family": "Jurassic-2",
                "size": "178B",
                "capabilities": ["completion", "creative_writing"]
            },
            {
                "name": "j2-mid",
                "family": "Jurassic-2", 
                "size": "7.5B",
                "capabilities": ["completion", "balanced_performance"]
            },
            {
                "name": "j2-light",
                "family": "Jurassic-2",
                "size": "7.5B", 
                "capabilities": ["completion", "fast_inference"]
            }
        ]
        
        return {
            "available_models": known_models,
            "total_count": len(known_models),
            "note": "AI21 Labs models - contact AI21 for latest model availability"
        }


# Convenience functions
def create_ai21_agent(model_name: str = "jamba-1.5-large",
                     api_key: Optional[str] = None,
                     capabilities: Optional[list] = None,
                     system_prompt: str = "",
                     **config) -> 'AIAgent':
    """Create an AI agent using AI21 connector"""
    from .base_connector import AIAgent
    
    if capabilities is None:
        if "jamba" in model_name.lower():
            capabilities = [
                "long_context_processing",
                "reasoning",
                "multilingual",
                "structured_output",
                "conversation",
                "text_generation"
            ]
        else:  # Jurassic models
            capabilities = [
                "text_generation",
                "completion",
                "creative_writing", 
                "summarization"
            ]
    
    connector = AI21Connector(model_name, api_key, **config)
    agent = AIAgent(connector, capabilities, name=f"AI21_{model_name}")
    
    if not system_prompt:
        if "jamba" in model_name.lower():
            system_prompt = """You are powered by AI21's Jamba model with hybrid Mamba-Transformer architecture. 
            You excel at processing very long contexts efficiently and providing structured, reasoned responses."""
        else:
            system_prompt = """You are powered by AI21's Jurassic model. You provide helpful, creative, and 
            well-structured responses with a focus on clarity and helpfulness."""
    
    agent.set_system_prompt(system_prompt)
    return agent


def create_jamba_agent(api_key: Optional[str] = None, **config):
    """Create a Jamba agent optimized for long context tasks"""
    return create_ai21_agent(
        model_name="jamba-1.5-large",
        api_key=api_key,
        capabilities=[
            "ultra_long_context",
            "document_analysis",
            "reasoning",
            "structured_thinking",
            "multilingual_processing"
        ],
        system_prompt="""You are Jamba, an advanced AI with hybrid Mamba-Transformer architecture optimized for 
        extremely long context processing. You can handle documents, conversations, and tasks requiring memory 
        of extensive context with high efficiency.""",
        **config
    )


def create_jurassic_agent(model_size: str = "ultra", api_key: Optional[str] = None, **config):
    """Create a Jurassic agent for creative and completion tasks"""
    model_name = f"j2-{model_size}"
    
    return create_ai21_agent(
        model_name=model_name,
        api_key=api_key,
        capabilities=[
            "creative_writing",
            "text_completion",
            "storytelling",
            "content_generation"
        ],
        system_prompt=f"""You are powered by AI21's Jurassic-2 {model_size.title()} model. 
        You excel at creative writing, text completion, and generating engaging content with 
        sophisticated language understanding.""",
        **config
    ) 