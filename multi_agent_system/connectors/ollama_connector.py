import asyncio
import json
from typing import Dict, Any, Optional, List
import aiohttp
from .base_connector import AIModelConnector


class OllamaConnector(AIModelConnector):
    """
    Connector for Ollama local AI models.
    """
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434", **config):
        super().__init__(model_name, config)
        self.base_url = base_url.rstrip('/')
        self.timeout = config.get("timeout", 120)
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using Ollama's generate API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters (temperature, top_p, etc.)
            
        Returns:
            Generated response string
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }
        
        # Add optional parameters
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["top_k"] = kwargs["top_k"]
        if "num_predict" in kwargs:
            payload["num_predict"] = kwargs["num_predict"]
        elif "max_tokens" in kwargs:
            payload["num_predict"] = kwargs["max_tokens"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        if "system" in kwargs:
            payload["system"] = kwargs["system"]
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    if "response" not in result:
                        raise Exception("No response generated by Ollama")
                    
                    return result["response"]
                    
            except aiohttp.ClientError as e:
                raise Exception(f"Network error communicating with Ollama: {str(e)}")
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse Ollama response: {str(e)}")
    
    async def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response using Ollama's chat API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters
            
        Returns:
            Generated response string
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }
        
        # Add optional parameters
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            payload["top_k"] = kwargs["top_k"]
        if "num_predict" in kwargs:
            payload["num_predict"] = kwargs["num_predict"]
        elif "max_tokens" in kwargs:
            payload["num_predict"] = kwargs["max_tokens"]
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama chat API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    if "message" not in result or "content" not in result["message"]:
                        raise Exception("No response generated by Ollama chat")
                    
                    return result["message"]["content"]
                    
            except aiohttp.ClientError as e:
                raise Exception(f"Network error communicating with Ollama chat: {str(e)}")
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse Ollama chat response: {str(e)}")
    
    async def validate_connection(self) -> bool:
        """
        Validate the connection to Ollama by checking if it's running and the model exists.
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            # First check if Ollama is running by getting version
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/version",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        return False
            
            # Then check if the model exists
            model_exists = await self.check_model_exists()
            if not model_exists:
                self.logger.warning(f"Model {self.model_name} not found locally. You may need to run 'ollama pull {self.model_name}'")
                return False
            
            # Finally, make a simple request to validate the model works
            response = await self.generate_response(
                "Hello", 
                num_predict=5,
                temperature=0
            )
            return len(response.strip()) > 0
            
        except Exception as e:
            self.logger.error(f"Connection validation failed: {e}")
            return False
    
    async def check_model_exists(self) -> bool:
        """
        Check if the specified model exists locally in Ollama.
        
        Returns:
            True if model exists, False otherwise
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status != 200:
                        return False
                    
                    result = await response.json()
                    models = result.get("models", [])
                    
                    for model in models:
                        if model.get("name", "").startswith(self.model_name):
                            return True
                    
                    return False
                    
        except Exception as e:
            self.logger.error(f"Failed to check if model exists: {e}")
            return False
    
    async def pull_model(self) -> bool:
        """
        Pull/download the model to Ollama.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            payload = {"name": self.model_name}
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/pull",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=600)  # 10 minutes for model download
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        self.logger.error(f"Failed to pull model: {error_text}")
                        return False
                    
                    # Stream the download progress
                    async for line in response.content:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if "status" in data:
                                self.logger.info(f"Pull progress: {data['status']}")
                        except:
                            continue
                    
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to pull model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the Ollama model.
        
        Returns:
            Dictionary containing model information
        """
        model_info = {
            "provider": "Ollama",
            "model_name": self.model_name,
            "type": "local_model",
            "base_url": self.base_url
        }
        
        # Add known information for popular models
        if "llama" in self.model_name.lower():
            model_info.update({
                "family": "LLaMA",
                "capabilities": ["text_generation", "conversation", "reasoning"]
            })
        elif "mistral" in self.model_name.lower():
            model_info.update({
                "family": "Mistral",
                "capabilities": ["text_generation", "conversation", "code_generation"]
            })
        elif "codellama" in self.model_name.lower():
            model_info.update({
                "family": "Code Llama",
                "capabilities": ["code_generation", "code_completion", "code_explanation"]
            })
        elif "phi" in self.model_name.lower():
            model_info.update({
                "family": "Phi",
                "capabilities": ["text_generation", "reasoning", "conversation"]
            })
        
        return model_info
    
    async def get_available_models(self) -> Dict[str, Any]:
        """
        Get list of available models from Ollama.
        
        Returns:
            Dictionary containing available models
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    
                    if response.status != 200:
                        return {"error": f"API error {response.status}"}
                    
                    result = await response.json()
                    models = result.get("models", [])
                    
                    formatted_models = []
                    for model in models:
                        formatted_models.append({
                            "name": model.get("name"),
                            "size": model.get("size"),
                            "modified_at": model.get("modified_at"),
                            "digest": model.get("digest", "")[:12]  # Short digest
                        })
                    
                    return {
                        "available_models": formatted_models,
                        "total_count": len(formatted_models)
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            return {"error": str(e)}
    
    async def stream_response(self, prompt: str, **kwargs):
        """
        Generate a streaming response from Ollama.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Yields:
            Response chunks as they arrive
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True
        }
        
        # Add optional parameters
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "num_predict" in kwargs:
            payload["num_predict"] = kwargs["num_predict"]
        elif "max_tokens" in kwargs:
            payload["num_predict"] = kwargs["max_tokens"]
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Ollama streaming error {response.status}: {error_text}")
                    
                    async for line in response.content:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                    
            except aiohttp.ClientError as e:
                raise Exception(f"Network error during streaming: {str(e)}")


# Convenience function to create Ollama agents
def create_ollama_agent(model_name: str, 
                       base_url: str = "http://localhost:11434",
                       capabilities: Optional[list] = None,
                       system_prompt: str = "",
                       **config) -> 'AIAgent':
    """
    Create an AI agent using Ollama connector.
    
    Args:
        model_name: Ollama model to use
        base_url: Ollama server URL
        capabilities: List of capabilities for the agent
        system_prompt: System prompt for the agent
        **config: Additional configuration
        
    Returns:
        Configured AIAgent instance
    """
    from .base_connector import AIAgent
    
    # Default capabilities based on model type
    if capabilities is None:
        if "codellama" in model_name.lower() or "code" in model_name.lower():
            capabilities = [
                "code_generation",
                "code_completion", 
                "code_explanation",
                "debugging",
                "text_generation"
            ]
        elif "llama" in model_name.lower():
            capabilities = [
                "text_generation",
                "conversation",
                "reasoning",
                "analysis",
                "creative_writing"
            ]
        else:
            capabilities = [
                "text_generation",
                "conversation",
                "analysis"
            ]
    
    connector = OllamaConnector(model_name, base_url, **config)
    agent = AIAgent(connector, capabilities, name=f"Ollama_{model_name}")
    
    if system_prompt:
        agent.set_system_prompt(system_prompt)
    
    return agent 