import asyncio
import json
from typing import Dict, Any, Optional
import aiohttp
from .base_connector import AIModelConnector


class HuggingFaceConnector(AIModelConnector):
    """
    Connector for Hugging Face models via Inference API.
    
    Supports thousands of models including:
    - Text generation models
    - Conversational models
    - Code generation models
    - Specialized domain models
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None,
                 base_url: str = "https://api-inference.huggingface.co/models", **config):
        super().__init__(model_name, config)
        self.api_key = api_key or config.get("api_key")
        self.base_url = base_url.rstrip('/')
        self.wait_for_model = config.get("wait_for_model", True)
        
        if not self.api_key:
            raise ValueError("Hugging Face API key is required")
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using Hugging Face Inference API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response string
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        # Determine task type based on model name
        if self._is_conversational_model():
            return await self._generate_conversational_response(prompt, headers, **kwargs)
        else:
            return await self._generate_text_response(prompt, headers, **kwargs)
    
    async def _generate_text_response(self, prompt: str, headers: dict, **kwargs) -> str:
        """Generate response for text generation models"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": kwargs.get("max_tokens", 250),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.95),
                "top_k": kwargs.get("top_k", 50),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
                "return_full_text": False
            },
            "options": {
                "wait_for_model": self.wait_for_model,
                "use_cache": kwargs.get("use_cache", True)
            }
        }
        
        # Add stop sequences if provided
        if "stop" in kwargs:
            stop_sequences = kwargs["stop"] if isinstance(kwargs["stop"], list) else [kwargs["stop"]]
            payload["parameters"]["stop"] = stop_sequences
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/{self.model_name}",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)  # HF models can be slow
                ) as response:
                    
                    if response.status == 503:
                        # Model is loading, wait and retry
                        await asyncio.sleep(10)
                        return await self._generate_text_response(prompt, headers, **kwargs)
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Hugging Face API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    if isinstance(result, list) and len(result) > 0:
                        generated_text = result[0].get("generated_text", "").strip()
                        return generated_text
                    elif isinstance(result, dict) and "generated_text" in result:
                        return result["generated_text"].strip()
                    else:
                        raise Exception("Invalid response format from Hugging Face")
                    
            except aiohttp.ClientError as e:
                raise Exception(f"Network error communicating with Hugging Face: {str(e)}")
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse Hugging Face response: {str(e)}")
    
    async def _generate_conversational_response(self, prompt: str, headers: dict, **kwargs) -> str:
        """Generate response for conversational models"""
        payload = {
            "inputs": {
                "text": prompt
            },
            "parameters": {
                "max_length": kwargs.get("max_tokens", 1000),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.95),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.1)
            },
            "options": {
                "wait_for_model": self.wait_for_model,
                "use_cache": kwargs.get("use_cache", True)
            }
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/{self.model_name}",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as response:
                    
                    if response.status == 503:
                        # Model is loading, wait and retry
                        await asyncio.sleep(10)
                        return await self._generate_conversational_response(prompt, headers, **kwargs)
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Hugging Face API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    if isinstance(result, dict) and "generated_text" in result:
                        return result["generated_text"].strip()
                    elif isinstance(result, list) and len(result) > 0:
                        return result[0].get("generated_text", "").strip()
                    else:
                        raise Exception("Invalid conversational response from Hugging Face")
                    
            except aiohttp.ClientError as e:
                raise Exception(f"Network error communicating with Hugging Face: {str(e)}")
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse Hugging Face response: {str(e)}")
    
    def _is_conversational_model(self) -> bool:
        """Check if this is a conversational model"""
        conversational_indicators = [
            "DialoGPT", "BlenderBot", "conversational", 
            "chat", "dialog", "conversation"
        ]
        return any(indicator.lower() in self.model_name.lower() 
                  for indicator in conversational_indicators)
    
    async def validate_connection(self) -> bool:
        """Validate connection to Hugging Face API"""
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
        """Get information about the Hugging Face model"""
        model_info = {
            "provider": "Hugging Face",
            "model_name": self.model_name,
            "base_url": self.base_url,
            "type": "inference_api"
        }
        
        # Add capabilities based on model name patterns
        if any(x in self.model_name.lower() for x in ["code", "codegen", "starcoder", "incoder"]):
            model_info["capabilities"] = ["code_generation", "code_completion", "code_explanation"]
        elif any(x in self.model_name.lower() for x in ["chat", "dialog", "conversational"]):
            model_info["capabilities"] = ["conversation", "text_generation", "reasoning"]
        elif any(x in self.model_name.lower() for x in ["flan", "t5", "ul2"]):
            model_info["capabilities"] = ["text_generation", "instruction_following", "reasoning"]
        elif any(x in self.model_name.lower() for x in ["gpt", "opt", "bloom", "llama"]):
            model_info["capabilities"] = ["text_generation", "creative_writing", "reasoning"]
        else:
            model_info["capabilities"] = ["text_generation"]
        
        return model_info
    
    async def get_model_details(self) -> Dict[str, Any]:
        """Get detailed information about the model from HF API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"https://huggingface.co/api/models/{self.model_name}",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status != 200:
                        return {"error": f"API error {response.status}"}
                    
                    result = await response.json()
                    
                    return {
                        "model_id": result.get("modelId"),
                        "pipeline_tag": result.get("pipeline_tag"),
                        "tags": result.get("tags", []),
                        "downloads": result.get("downloads", 0),
                        "likes": result.get("likes", 0),
                        "library_name": result.get("library_name"),
                        "created_at": result.get("createdAt"),
                        "last_modified": result.get("lastModified")
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to get model details: {e}")
            return {"error": str(e)}


# Convenience functions for popular HF models
def create_huggingface_agent(model_name: str,
                            api_key: Optional[str] = None,
                            capabilities: Optional[list] = None,
                            system_prompt: str = "",
                            **config) -> 'AIAgent':
    """Create an AI agent using Hugging Face connector"""
    from .base_connector import AIAgent
    
    # Auto-detect capabilities if not provided
    if capabilities is None:
        if any(x in model_name.lower() for x in ["code", "codegen", "starcoder"]):
            capabilities = ["code_generation", "code_completion", "debugging"]
        elif any(x in model_name.lower() for x in ["chat", "dialog", "conversational"]):
            capabilities = ["conversation", "text_generation", "reasoning"]
        elif any(x in model_name.lower() for x in ["flan", "t5"]):
            capabilities = ["instruction_following", "reasoning", "analysis"]
        else:
            capabilities = ["text_generation", "creative_writing"]
    
    connector = HuggingFaceConnector(model_name, api_key, **config)
    agent = AIAgent(connector, capabilities, name=f"HF_{model_name.split('/')[-1]}")
    
    if system_prompt:
        agent.set_system_prompt(system_prompt)
    
    return agent


# Pre-configured popular models
def create_codegen_agent(api_key: Optional[str] = None, **config):
    """Create a code generation agent using CodeGen model"""
    return create_huggingface_agent(
        "microsoft/CodeGPT-small-py",
        api_key=api_key,
        capabilities=["code_generation", "code_completion", "python_coding"],
        system_prompt="You are a Python coding assistant. Generate clean, efficient, and well-commented code.",
        **config
    )


def create_flan_agent(api_key: Optional[str] = None, **config):
    """Create an instruction-following agent using FLAN-T5"""
    return create_huggingface_agent(
        "google/flan-t5-large",
        api_key=api_key,
        capabilities=["instruction_following", "reasoning", "analysis", "summarization"],
        system_prompt="You are an intelligent assistant that follows instructions precisely and provides detailed explanations.",
        **config
    )


def create_dialogpt_agent(api_key: Optional[str] = None, **config):
    """Create a conversational agent using DialoGPT"""
    return create_huggingface_agent(
        "microsoft/DialoGPT-large",
        api_key=api_key,
        capabilities=["conversation", "chat", "dialogue"],
        system_prompt="You are a friendly conversational AI assistant.",
        **config
    ) 