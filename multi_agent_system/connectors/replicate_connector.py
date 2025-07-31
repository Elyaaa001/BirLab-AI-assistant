import asyncio
import json
from typing import Dict, Any, Optional
import aiohttp
from .base_connector import AIModelConnector


class ReplicateConnector(AIModelConnector):
    """
    Connector for Replicate models (various models via Replicate API)
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None,
                 base_url: str = "https://api.replicate.com/v1", **config):
        super().__init__(model_name, config)
        self.api_key = api_key or config.get("api_key")
        self.base_url = base_url.rstrip('/')
        
        if not self.api_key:
            raise ValueError("Replicate API key is required")
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using Replicate's API.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters
            
        Returns:
            Generated response string
        """
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Create prediction
        prediction_id = await self._create_prediction(prompt, headers, **kwargs)
        
        # Wait for completion and get result
        return await self._wait_for_completion(prediction_id, headers)
    
    async def _create_prediction(self, prompt: str, headers: dict, **kwargs) -> str:
        """Create a prediction and return its ID"""
        # Build input based on model type
        input_data = self._build_input(prompt, **kwargs)
        
        payload = {
            "version": self._get_model_version(),
            "input": input_data
        }
        
        # Use model-specific endpoint if available
        if "/" in self.model_name:
            url = f"{self.base_url}/predictions"
        else:
            url = f"{self.base_url}/models/{self.model_name}/predictions"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status not in [200, 201]:
                        error_text = await response.text()
                        raise Exception(f"Replicate API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    return result["id"]
                    
            except aiohttp.ClientError as e:
                raise Exception(f"Network error communicating with Replicate: {str(e)}")
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse Replicate response: {str(e)}")
    
    async def _wait_for_completion(self, prediction_id: str, headers: dict, max_wait: int = 300) -> str:
        """Wait for prediction to complete and return result"""
        start_time = asyncio.get_event_loop().time()
        
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get(
                        f"{self.base_url}/predictions/{prediction_id}",
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        
                        if response.status != 200:
                            error_text = await response.text()
                            raise Exception(f"Replicate status check error {response.status}: {error_text}")
                        
                        result = await response.json()
                        status = result.get("status")
                        
                        if status == "succeeded":
                            output = result.get("output")
                            if isinstance(output, list):
                                return "".join(str(item) for item in output)
                            else:
                                return str(output) if output else ""
                        
                        elif status == "failed":
                            error_msg = result.get("error", "Unknown error")
                            raise Exception(f"Prediction failed: {error_msg}")
                        
                        elif status in ["starting", "processing"]:
                            # Still running, wait and check again
                            await asyncio.sleep(2)
                            
                            # Check timeout
                            if asyncio.get_event_loop().time() - start_time > max_wait:
                                raise Exception("Prediction timeout")
                        
                        else:
                            raise Exception(f"Unknown prediction status: {status}")
                            
                except aiohttp.ClientError as e:
                    raise Exception(f"Network error checking prediction status: {str(e)}")
                except json.JSONDecodeError as e:
                    raise Exception(f"Failed to parse status response: {str(e)}")
    
    def _build_input(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Build input dict based on model type"""
        # Default input structure
        input_data = {
            "prompt": prompt,
            "max_new_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.1)
        }
        
        # Model-specific adjustments
        if "llama" in self.model_name.lower():
            input_data.update({
                "system_prompt": kwargs.get("system_prompt", ""),
                "min_new_tokens": kwargs.get("min_tokens", 1)
            })
        
        elif "stable-diffusion" in self.model_name.lower():
            # For image generation models
            input_data = {
                "prompt": prompt,
                "width": kwargs.get("width", 512),
                "height": kwargs.get("height", 512),
                "num_inference_steps": kwargs.get("steps", 50),
                "guidance_scale": kwargs.get("guidance_scale", 7.5)
            }
        
        elif "whisper" in self.model_name.lower():
            # For audio transcription models
            input_data = {
                "audio": prompt,  # Assume prompt is audio file path/URL
                "model": kwargs.get("whisper_model", "large-v2"),
                "transcription": kwargs.get("transcription", "plain text")
            }
        
        return input_data
    
    def _get_model_version(self) -> Optional[str]:
        """Get the specific version hash for the model"""
        # This would typically be stored in a config or fetched from API
        # For now, return None to use latest version
        version_mapping = {
            "meta/llama-2-70b-chat": "02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            "meta/llama-2-13b-chat": "f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d",
            "meta/llama-2-7b-chat": "13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0"
        }
        
        return version_mapping.get(self.model_name)
    
    async def validate_connection(self) -> bool:
        """Validate connection to Replicate API"""
        try:
            # Just check if we can access the API
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.error(f"Connection validation failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the Replicate model"""
        model_info = {
            "provider": "Replicate",
            "model_name": self.model_name,
            "base_url": self.base_url
        }
        
        # Add capabilities based on model name patterns
        if any(x in self.model_name.lower() for x in ["llama", "alpaca", "vicuna"]):
            model_info["capabilities"] = ["text_generation", "conversation", "reasoning"]
        elif "stable-diffusion" in self.model_name.lower():
            model_info["capabilities"] = ["image_generation", "text_to_image"]
        elif "whisper" in self.model_name.lower():
            model_info["capabilities"] = ["audio_transcription", "speech_to_text"]
        elif "code" in self.model_name.lower():
            model_info["capabilities"] = ["code_generation", "code_completion"]
        else:
            model_info["capabilities"] = ["text_generation"]
        
        return model_info
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models from Replicate"""
        headers = {
            "Authorization": f"Token {self.api_key}",
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
                    for model in result.get("results", []):
                        models.append({
                            "name": model.get("name"),
                            "owner": model.get("owner"),
                            "description": model.get("description", ""),
                            "visibility": model.get("visibility"),
                            "github_url": model.get("github_url"),
                            "paper_url": model.get("paper_url"),
                            "license_url": model.get("license_url"),
                            "run_count": model.get("run_count", 0)
                        })
                    
                    return {
                        "available_models": models,
                        "total_count": len(models)
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            return {"error": str(e)}


# Convenience functions for popular Replicate models
def create_replicate_agent(model_name: str,
                          api_key: Optional[str] = None,
                          capabilities: Optional[list] = None,
                          system_prompt: str = "",
                          **config) -> 'AIAgent':
    """Create an AI agent using Replicate connector"""
    from .base_connector import AIAgent
    
    # Auto-detect capabilities if not provided
    if capabilities is None:
        if any(x in model_name.lower() for x in ["llama", "alpaca", "vicuna"]):
            capabilities = ["text_generation", "conversation", "reasoning"]
        elif "stable-diffusion" in model_name.lower():
            capabilities = ["image_generation", "text_to_image"]
        elif "code" in model_name.lower():
            capabilities = ["code_generation", "code_completion"]
        else:
            capabilities = ["text_generation"]
    
    connector = ReplicateConnector(model_name, api_key, **config)
    agent = AIAgent(connector, capabilities, name=f"Replicate_{model_name.split('/')[-1]}")
    
    if system_prompt:
        agent.set_system_prompt(system_prompt)
    
    return agent


# Pre-configured popular models
def create_llama2_replicate_agent(size: str = "7b", api_key: Optional[str] = None, **config):
    """Create a Llama 2 agent via Replicate"""
    size_map = {"7b": "7b-chat", "13b": "13b-chat", "70b": "70b-chat"}
    model_name = f"meta/llama-2-{size_map.get(size, '7b-chat')}"
    
    return create_replicate_agent(
        model_name,
        api_key=api_key,
        capabilities=["conversation", "reasoning", "text_generation", "analysis"],
        system_prompt="You are a helpful AI assistant based on Llama 2. Provide accurate and thoughtful responses.",
        **config
    )


def create_stable_diffusion_agent(version: str = "xl", api_key: Optional[str] = None, **config):
    """Create a Stable Diffusion image generation agent via Replicate"""
    model_name = f"stability-ai/stable-diffusion-{version}"
    
    return create_replicate_agent(
        model_name,
        api_key=api_key,
        capabilities=["image_generation", "text_to_image", "creative_visual"],
        system_prompt="You are an image generation AI. Create detailed and creative visual content from text descriptions.",
        **config
    ) 