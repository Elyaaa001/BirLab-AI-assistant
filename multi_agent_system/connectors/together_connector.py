import asyncio
import json
from typing import Dict, Any, Optional
import aiohttp
from .base_connector import AIModelConnector


class TogetherConnector(AIModelConnector):
    """
    Connector for Together AI models (various open-source models via Together API)
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None,
                 base_url: str = "https://api.together.xyz/v1", **config):
        super().__init__(model_name, config)
        self.api_key = api_key or config.get("api_key")
        self.base_url = base_url.rstrip('/')
        
        if not self.api_key:
            raise ValueError("Together API key is required")
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using Together's API.
        
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
        
        # Use chat completions if model supports it, otherwise use completions
        if self._is_chat_model():
            return await self._generate_chat_response(prompt, headers, **kwargs)
        else:
            return await self._generate_completion_response(prompt, headers, **kwargs)
    
    async def _generate_chat_response(self, prompt: str, headers: dict, **kwargs) -> str:
        """Generate response using chat completions endpoint"""
        messages = [{"role": "user", "content": prompt}]
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "stream": False
        }
        
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
                        raise Exception(f"Together API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    if "choices" not in result or not result["choices"]:
                        raise Exception("No response generated by Together")
                    
                    return result["choices"][0]["message"]["content"]
                    
            except aiohttp.ClientError as e:
                raise Exception(f"Network error communicating with Together: {str(e)}")
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse Together response: {str(e)}")
    
    async def _generate_completion_response(self, prompt: str, headers: dict, **kwargs) -> str:
        """Generate response using completions endpoint"""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
            "stream": False
        }
        
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Together API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    if "choices" not in result or not result["choices"]:
                        raise Exception("No response generated by Together")
                    
                    return result["choices"][0]["text"].strip()
                    
            except aiohttp.ClientError as e:
                raise Exception(f"Network error communicating with Together: {str(e)}")
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse Together response: {str(e)}")
    
    def _is_chat_model(self) -> bool:
        """Check if this model supports chat format"""
        chat_indicators = [
            "chat", "instruct", "assistant", "vicuna", 
            "alpaca", "wizard", "orcamini", "guanaco"
        ]
        return any(indicator.lower() in self.model_name.lower() 
                  for indicator in chat_indicators)
    
    async def validate_connection(self) -> bool:
        """Validate connection to Together API"""
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
        """Get information about the Together model"""
        model_info = {
            "provider": "Together AI",
            "model_name": self.model_name,
            "base_url": self.base_url
        }
        
        # Add capabilities based on model name patterns
        if any(x in self.model_name.lower() for x in ["code", "codellama", "starcoder"]):
            model_info["capabilities"] = ["code_generation", "code_completion", "debugging"]
        elif any(x in self.model_name.lower() for x in ["llama", "alpaca", "vicuna"]):
            model_info["capabilities"] = ["text_generation", "conversation", "reasoning", "instruction_following"]
        elif any(x in self.model_name.lower() for x in ["falcon", "mpt"]):
            model_info["capabilities"] = ["text_generation", "creative_writing", "conversation"]
        elif any(x in self.model_name.lower() for x in ["flan", "t5"]):
            model_info["capabilities"] = ["instruction_following", "reasoning", "analysis"]
        else:
            model_info["capabilities"] = ["text_generation", "conversation"]
        
        # Add model size information if detectable
        if "7b" in self.model_name.lower():
            model_info["parameters"] = "7B"
        elif "13b" in self.model_name.lower():
            model_info["parameters"] = "13B"
        elif "30b" in self.model_name.lower():
            model_info["parameters"] = "30B"
        elif "65b" in self.model_name.lower():
            model_info["parameters"] = "65B"
        elif "70b" in self.model_name.lower():
            model_info["parameters"] = "70B"
        
        return model_info
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models from Together"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status != 200:
                        return {"error": f"API error {response.status}"}
                    
                    result = await response.json()
                    
                    models = []
                    for model in result:
                        models.append({
                            "id": model.get("id"),
                            "display_name": model.get("display_name", model.get("id")),
                            "description": model.get("description", ""),
                            "context_length": model.get("context_length", 0),
                            "pricing": {
                                "input": model.get("pricing", {}).get("input", 0),
                                "output": model.get("pricing", {}).get("output", 0)
                            }
                        })
                    
                    return {
                        "available_models": models,
                        "total_count": len(models)
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            return {"error": str(e)}


# Convenience functions for popular Together models
def create_together_agent(model_name: str,
                         api_key: Optional[str] = None,
                         capabilities: Optional[list] = None,
                         system_prompt: str = "",
                         **config) -> 'AIAgent':
    """Create an AI agent using Together connector"""
    from .base_connector import AIAgent
    
    # Auto-detect capabilities if not provided
    if capabilities is None:
        if any(x in model_name.lower() for x in ["code", "codellama", "starcoder"]):
            capabilities = ["code_generation", "code_completion", "debugging"]
        elif any(x in model_name.lower() for x in ["llama", "alpaca", "vicuna"]):
            capabilities = ["text_generation", "conversation", "reasoning", "instruction_following"]
        elif any(x in model_name.lower() for x in ["falcon", "mpt"]):
            capabilities = ["text_generation", "creative_writing", "conversation"]
        else:
            capabilities = ["text_generation", "conversation"]
    
    connector = TogetherConnector(model_name, api_key, **config)
    agent = AIAgent(connector, capabilities, name=f"Together_{model_name.split('/')[-1]}")
    
    if system_prompt:
        agent.set_system_prompt(system_prompt)
    
    return agent


# Pre-configured popular models
def create_llama2_agent(size: str = "7b", api_key: Optional[str] = None, **config):
    """Create a Llama 2 agent via Together"""
    model_name = f"meta-llama/Llama-2-{size}-chat-hf"
    return create_together_agent(
        model_name,
        api_key=api_key,
        capabilities=["conversation", "reasoning", "instruction_following", "analysis"],
        system_prompt="You are a helpful AI assistant based on Llama 2. Provide accurate and helpful responses.",
        **config
    )


def create_codellama_together_agent(size: str = "7b", api_key: Optional[str] = None, **config):
    """Create a Code Llama agent via Together"""
    model_name = f"codellama/CodeLlama-{size}-Instruct-hf"
    return create_together_agent(
        model_name,
        api_key=api_key,
        capabilities=["code_generation", "code_explanation", "debugging", "code_completion"],
        system_prompt="You are a coding assistant specialized in generating high-quality code and explanations.",
        **config
    )


def create_falcon_agent(size: str = "7b", api_key: Optional[str] = None, **config):
    """Create a Falcon agent via Together"""
    model_name = f"tiiuae/falcon-{size}-instruct"
    return create_together_agent(
        model_name,
        api_key=api_key,
        capabilities=["text_generation", "creative_writing", "conversation", "reasoning"],
        system_prompt="You are an intelligent AI assistant based on Falcon. Provide creative and thoughtful responses.",
        **config
    ) 