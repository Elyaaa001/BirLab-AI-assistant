import asyncio
import json
import base64
import os
from typing import Dict, Any, Optional, List, Union
import aiohttp
from .base_connector import AIModelConnector

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed - continue without it
    pass


class GoogleAIConnector(AIModelConnector):
    """
    Enhanced connector for Google AI models (Gemini Pro, Gemini Ultra, PaLM, etc.)
    
    Supports:
    - Multimodal inputs (text + images)
    - Various Gemini model variants
    - Advanced safety settings
    - Function calling capabilities
    """
    
    def __init__(self, model_name: str = "gemini-1.5-pro", api_key: Optional[str] = None,
                 base_url: str = "https://generativelanguage.googleapis.com/v1beta", **config):
        super().__init__(model_name, config)
        self.api_key = api_key or config.get("api_key") or os.getenv("GOOGLE_AI_API_KEY")
        self.base_url = base_url.rstrip('/')
        
        if not self.api_key:
            raise ValueError(
                "Google AI API key is required. Please provide it via:\n"
                "1. api_key parameter\n"
                "2. GOOGLE_AI_API_KEY environment variable\n"
                "3. .env file with GOOGLE_AI_API_KEY=your-key\n"
                "Get your key from: https://aistudio.google.com/app/apikey"
            )
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response using Google AI models with enhanced capabilities.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional parameters including images, safety settings, etc.
            
        Returns:
            Generated response string
        """
        # Handle multimodal inputs
        if "images" in kwargs:
            return await self._generate_multimodal_response(prompt, kwargs["images"], **kwargs)
        else:
            return await self._generate_text_response(prompt, **kwargs)
    
    async def _generate_text_response(self, prompt: str, **kwargs) -> str:
        """Generate text-only response"""
        if "gemini" in self.model_name.lower():
            return await self._generate_gemini_response(prompt, **kwargs)
        else:
            return await self._generate_palm_response(prompt, **kwargs)
    
    async def _generate_multimodal_response(self, prompt: str, images: List[Union[str, bytes]], **kwargs) -> str:
        """Generate response with text and image inputs"""
        if "gemini" not in self.model_name.lower():
            raise Exception("Multimodal input is only supported by Gemini models")
        
        url = f"{self.base_url}/models/{self.model_name}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Build multimodal content
        parts = [{"text": prompt}]
        
        # Add images
        for image in images:
            if isinstance(image, str):
                # Assume it's a base64 encoded image or file path
                if image.startswith('data:image'):
                    # Extract base64 data
                    base64_data = image.split(',')[1]
                    mime_type = image.split(';')[0].split(':')[1]
                elif image.startswith('/') or '.' in image:
                    # File path - read and encode
                    import os
                    if os.path.exists(image):
                        with open(image, 'rb') as f:
                            image_data = f.read()
                        base64_data = base64.b64encode(image_data).decode()
                        # Determine MIME type from extension
                        ext = image.lower().split('.')[-1]
                        mime_type = f"image/{ext}" if ext in ['png', 'jpg', 'jpeg', 'gif', 'webp'] else "image/jpeg"
                    else:
                        continue
                else:
                    # Direct base64 string
                    base64_data = image
                    mime_type = "image/jpeg"  # Default
            else:
                # Bytes data
                base64_data = base64.b64encode(image).decode()
                mime_type = "image/jpeg"  # Default
            
            parts.append({
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64_data
                }
            })
        
        payload = {
            "contents": [{
                "parts": parts
            }],
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 2048),
                "topP": kwargs.get("top_p", 0.95),
                "topK": kwargs.get("top_k", 40)
            }
        }
        
        # Add safety settings if provided
        if "safety_settings" in kwargs:
            payload["safetySettings"] = kwargs["safety_settings"]
        else:
            # Default safety settings
            payload["safetySettings"] = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
        
        params = {"key": self.api_key}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=120)  # Longer timeout for multimodal
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Google AI API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    if "candidates" not in result or not result["candidates"]:
                        raise Exception("No response generated by Google AI")
                    
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        return candidate["content"]["parts"][0]["text"]
                    else:
                        raise Exception("Invalid response format from Google AI")
                    
            except aiohttp.ClientError as e:
                raise Exception(f"Network error communicating with Google AI: {str(e)}")
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse Google AI response: {str(e)}")
    
    async def _generate_gemini_response(self, prompt: str, **kwargs) -> str:
        """Enhanced Gemini response generation with better parameter support"""
        url = f"{self.base_url}/models/{self.model_name}:generateContent"
        
        headers = {
            "Content-Type": "application/json",
        }
        
        # Build content with system instructions if provided
        contents = []
        if "system_prompt" in kwargs and kwargs["system_prompt"]:
            contents.append({
                "role": "user",
                "parts": [{"text": f"System: {kwargs['system_prompt']}\n\nUser: {prompt}"}]
            })
        else:
            contents.append({
                "parts": [{"text": prompt}]
            })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 2048),
                "topP": kwargs.get("top_p", 0.95),
                "topK": kwargs.get("top_k", 40),
                "candidateCount": 1,
                "stopSequences": kwargs.get("stop", [])
            }
        }
        
        # Add safety settings
        if "safety_settings" in kwargs:
            payload["safetySettings"] = kwargs["safety_settings"]
        elif kwargs.get("safe_mode", True):
            payload["safetySettings"] = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
            ]
        
        params = {"key": self.api_key}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Google AI API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    if "candidates" not in result or not result["candidates"]:
                        raise Exception("No response generated by Google AI")
                    
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        return candidate["content"]["parts"][0]["text"]
                    else:
                        raise Exception("Invalid response format from Google AI")
                    
            except aiohttp.ClientError as e:
                raise Exception(f"Network error communicating with Google AI: {str(e)}")
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse Google AI response: {str(e)}")
    
    async def _generate_palm_response(self, prompt: str, **kwargs) -> str:
        """Generate response using PaLM API (legacy support)"""
        url = f"{self.base_url}/models/{self.model_name}:generateText"
        
        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "prompt": {"text": prompt},
            "temperature": kwargs.get("temperature", 0.7),
            "candidateCount": 1,
            "maxOutputTokens": kwargs.get("max_tokens", 1000),
            "topP": kwargs.get("top_p", 0.95),
            "topK": kwargs.get("top_k", 40)
        }
        
        params = {"key": self.api_key}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    headers=headers,
                    json=payload,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Google AI API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    if "candidates" not in result or not result["candidates"]:
                        raise Exception("No response generated by Google AI")
                    
                    return result["candidates"][0]["output"]
                    
            except aiohttp.ClientError as e:
                raise Exception(f"Network error communicating with Google AI: {str(e)}")
            except json.JSONDecodeError as e:
                raise Exception(f"Failed to parse Google AI response: {str(e)}")
    
    async def analyze_image(self, image_path_or_data: Union[str, bytes], 
                           question: str = "Describe this image in detail", **kwargs) -> str:
        """
        Analyze an image using Gemini's vision capabilities.
        
        Args:
            image_path_or_data: Path to image file or raw image bytes
            question: Question to ask about the image
            **kwargs: Additional parameters
            
        Returns:
            Analysis of the image
        """
        return await self._generate_multimodal_response(question, [image_path_or_data], **kwargs)
    
    async def chat_with_images(self, messages: List[Dict], **kwargs) -> str:
        """
        Have a conversation that includes images.
        
        Args:
            messages: List of message dicts with 'role', 'text', and optional 'images'
            **kwargs: Additional parameters
            
        Returns:
            Chat response
        """
        # Build conversation context
        conversation_parts = []
        for msg in messages:
            if msg.get("images"):
                # Add multimodal message
                parts = [{"text": msg["text"]}]
                for image in msg["images"]:
                    if isinstance(image, str) and image.startswith('data:image'):
                        base64_data = image.split(',')[1]
                        parts.append({
                            "inline_data": {
                                "mime_type": image.split(';')[0].split(':')[1],
                                "data": base64_data
                            }
                        })
                conversation_parts.append(f"{msg['role']}: {msg['text']} [with image]")
            else:
                conversation_parts.append(f"{msg['role']}: {msg['text']}")
        
        # Get the last message for response
        last_message = messages[-1]
        if last_message.get("images"):
            return await self._generate_multimodal_response(
                "\n".join(conversation_parts), 
                last_message["images"], 
                **kwargs
            )
        else:
            return await self._generate_text_response("\n".join(conversation_parts), **kwargs)
    
    async def validate_connection(self) -> bool:
        """Validate connection to Google AI"""
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
        """Get enhanced information about the Google AI model"""
        model_info = {
            "provider": "Google AI",
            "model_name": self.model_name,
            "base_url": self.base_url
        }
        
        if "gemini-1.5-pro" in self.model_name.lower():
            model_info.update({
                "context_length": 2000000,  # 2M tokens!
                "capabilities": [
                    "text_generation", "reasoning", "analysis", "multimodal", 
                    "vision", "document_processing", "code_generation", "math"
                ],
                "type": "gemini",
                "version": "1.5 Pro",
                "special_features": [
                    "2M token context window",
                    "Multimodal understanding (text + images)",
                    "Advanced reasoning capabilities",
                    "Native code generation",
                    "Document and video analysis"
                ]
            })
        elif "gemini-1.5-flash" in self.model_name.lower():
            model_info.update({
                "context_length": 1000000,  # 1M tokens
                "capabilities": [
                    "text_generation", "fast_responses", "multimodal", 
                    "vision", "reasoning", "analysis"
                ],
                "type": "gemini",
                "version": "1.5 Flash",
                "special_features": [
                    "1M token context window",
                    "Lightning-fast responses",
                    "Cost-optimized",
                    "Multimodal capabilities"
                ]
            })
        elif "gemini-pro" in self.model_name.lower():
            model_info.update({
                "context_length": 32768,
                "capabilities": ["text_generation", "reasoning", "analysis", "multimodal"],
                "type": "gemini",
                "version": "1.0 Pro"
            })
        elif "gemini-ultra" in self.model_name.lower():
            model_info.update({
                "context_length": 32768,
                "capabilities": [
                    "text_generation", "reasoning", "analysis", "complex_tasks", 
                    "multimodal", "advanced_math", "expert_level_tasks"
                ],
                "type": "gemini",
                "version": "1.0 Ultra",
                "performance": "Highest capability model"
            })
        elif "palm" in self.model_name.lower():
            model_info.update({
                "context_length": 8192,
                "capabilities": ["text_generation", "conversation", "analysis"],
                "type": "palm",
                "status": "Legacy model"
            })
        
        return model_info
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models from Google AI"""
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/models",
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status != 200:
                        return {"error": f"API error {response.status}"}
                    
                    result = await response.json()
                    
                    models = []
                    for model in result.get("models", []):
                        model_id = model.get("name", "").replace("models/", "")
                        models.append({
                            "name": model_id,
                            "display_name": model.get("displayName", model_id),
                            "description": model.get("description", ""),
                            "supported_methods": model.get("supportedGenerationMethods", []),
                            "input_token_limit": model.get("inputTokenLimit", 0),
                            "output_token_limit": model.get("outputTokenLimit", 0)
                        })
                    
                    return {
                        "available_models": models,
                        "total_count": len(models)
                    }
                    
        except Exception as e:
            self.logger.error(f"Failed to get available models: {e}")
            return {"error": str(e)}


# Enhanced convenience functions
def create_google_agent(model_name: str = "gemini-1.5-pro",
                       api_key: Optional[str] = None,
                       capabilities: Optional[list] = None,
                       system_prompt: str = "",
                       **config) -> 'AIAgent':
    """Create an enhanced AI agent using Google AI connector"""
    from .base_connector import AIAgent
    
    if capabilities is None:
        if "gemini-1.5-pro" in model_name.lower():
            capabilities = [
                "ultra_long_context",
                "multimodal_understanding",
                "advanced_reasoning",
                "document_analysis",
                "code_generation",
                "mathematical_reasoning",
                "vision_analysis"
            ]
        elif "gemini-1.5-flash" in model_name.lower():
            capabilities = [
                "fast_responses",
                "long_context",
                "multimodal_understanding",
                "vision_analysis",
                "cost_efficient"
            ]
        elif "gemini" in model_name.lower():
            capabilities = [
                "text_generation",
                "reasoning", 
                "analysis",
                "multimodal",
                "creative_writing",
                "code_generation"
            ]
        else:  # PaLM
            capabilities = [
                "text_generation",
                "conversation",
                "analysis",
                "reasoning"
            ]
    
    connector = GoogleAIConnector(model_name, api_key, **config)
    agent = AIAgent(connector, capabilities, name=f"Google_{model_name.replace('gemini-', 'Gemini_').replace('-', '_')}")
    
    if system_prompt:
        agent.set_system_prompt(system_prompt)
    
    return agent


def create_gemini_pro_agent(api_key: Optional[str] = None, **config):
    """Create a Gemini 1.5 Pro agent with maximum capabilities"""
    return create_google_agent(
        model_name="gemini-1.5-pro",
        api_key=api_key,
        capabilities=[
            "ultra_long_context",
            "multimodal_mastery",
            "advanced_reasoning",
            "document_processing",
            "vision_analysis",
            "code_generation",
            "mathematical_reasoning",
            "research_synthesis"
        ],
        system_prompt="""You are Gemini 1.5 Pro, Google's most advanced AI model with a 2 million token context window. 
        You excel at multimodal understanding, advanced reasoning, and processing extremely long documents. 
        You can analyze images, generate code, solve complex mathematical problems, and synthesize information 
        from massive amounts of text.""",
        **config
    )


def create_gemini_flash_agent(api_key: Optional[str] = None, **config):
    """Create a Gemini 1.5 Flash agent optimized for speed and efficiency"""
    return create_google_agent(
        model_name="gemini-1.5-flash",
        api_key=api_key,
        capabilities=[
            "lightning_fast_responses",
            "long_context_processing",
            "multimodal_understanding",
            "cost_efficient_processing",
            "rapid_analysis"
        ],
        system_prompt="""You are Gemini 1.5 Flash, optimized for lightning-fast responses while maintaining 
        high quality and multimodal capabilities. You provide quick, accurate answers with a 1 million token 
        context window, perfect for rapid analysis and efficient processing.""",
        **config
    )


def create_gemini_vision_agent(api_key: Optional[str] = None, **config):
    """Create a Gemini agent specialized for vision and multimodal tasks"""
    return create_google_agent(
        model_name="gemini-1.5-pro",
        api_key=api_key,
        capabilities=[
            "advanced_vision_analysis",
            "multimodal_reasoning",
            "image_understanding",
            "document_ocr",
            "visual_question_answering",
            "scene_analysis"
        ],
        system_prompt="""You are Gemini Vision, specialized in analyzing and understanding images, documents, 
        and visual content. You can describe images in detail, answer questions about visual content, 
        extract text from documents, analyze charts and graphs, and provide insights from visual data.""",
        **config
    )


def create_gemini_coder_agent(api_key: Optional[str] = None, **config):
    """Create a Gemini agent specialized for programming and code analysis"""
    return create_google_agent(
        model_name="gemini-1.5-pro",
        api_key=api_key,
        capabilities=[
            "advanced_code_generation",
            "code_analysis",
            "debugging",
            "architecture_design",
            "code_review",
            "multiple_languages"
        ],
        system_prompt="""You are Gemini Coder, specialized in programming and software development. 
        You excel at generating clean, efficient code across multiple programming languages, 
        analyzing existing code, debugging issues, and providing architectural guidance. 
        Your 2M token context allows you to work with entire codebases.""",
        **config
    )


def create_gemini_researcher_agent(api_key: Optional[str] = None, **config):
    """Create a Gemini agent specialized for research and analysis"""
    return create_google_agent(
        model_name="gemini-1.5-pro",
        api_key=api_key,
        capabilities=[
            "deep_research_analysis",
            "document_synthesis",
            "academic_writing",
            "data_analysis",
            "literature_review",
            "citation_management"
        ],
        system_prompt="""You are Gemini Researcher, specialized in academic and professional research. 
        With your 2 million token context window, you can process entire research papers, books, 
        and document collections to provide comprehensive analysis, synthesis, and insights. 
        You excel at literature reviews, data analysis, and academic writing.""",
        **config
    ) 