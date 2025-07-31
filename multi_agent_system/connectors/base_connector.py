from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import asyncio
import logging

from ..core.agent import Agent
from ..core.task import Task, TaskResult


class AIModelConnector(ABC):
    """
    Abstract base class for AI model connectors.
    
    This class defines the interface that all AI model connectors must implement
    to integrate with different AI services (OpenAI, Anthropic, local models, etc.)
    """
    
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{model_name}]")
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> str:
        """
        Generate a response from the AI model.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional model-specific parameters
            
        Returns:
            Generated response string
        """
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """
        Validate that the connection to the AI service is working.
        
        Returns:
            True if connection is valid, False otherwise
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the AI model.
        
        Returns:
            Dictionary containing model information
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the connector.
        
        Returns:
            Health status information
        """
        try:
            is_healthy = await self.validate_connection()
            return {
                "model_name": self.model_name,
                "status": "healthy" if is_healthy else "unhealthy",
                "connector_type": self.__class__.__name__,
                "model_info": self.get_model_info() if is_healthy else None
            }
        except Exception as e:
            return {
                "model_name": self.model_name,
                "status": "error",
                "connector_type": self.__class__.__name__,
                "error": str(e)
            }


class AIAgent(Agent):
    """
    An agent that uses an AI model connector to execute tasks.
    
    This agent wraps an AI model connector and provides the standard
    agent interface for task execution.
    """
    
    def __init__(self, connector: AIModelConnector, capabilities: List[str],
                 agent_id: Optional[str] = None, name: Optional[str] = None):
        super().__init__(agent_id, name or f"AI_{connector.model_name}")
        self.connector = connector
        self.capabilities_list = capabilities
        
        # AI-specific configuration
        self.system_prompt = ""
        self.temperature = 0.7
        self.max_tokens = 1000
        self.context_window = []
        self.max_context_length = 10
    
    def get_capabilities(self) -> List[str]:
        """Return the capabilities this AI agent can handle"""
        return self.capabilities_list
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for this AI agent"""
        self.system_prompt = prompt
        self.logger.info("System prompt updated")
    
    def set_parameters(self, temperature: float = None, max_tokens: int = None):
        """Set AI model parameters"""
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens
        self.logger.debug(f"Parameters updated: temp={self.temperature}, max_tokens={self.max_tokens}")
    
    def add_to_context(self, role: str, content: str):
        """Add a message to the conversation context"""
        self.context_window.append({"role": role, "content": content})
        
        # Keep context window within limits
        if len(self.context_window) > self.max_context_length:
            # Remove oldest non-system messages
            system_messages = [msg for msg in self.context_window if msg.get("role") == "system"]
            other_messages = [msg for msg in self.context_window if msg.get("role") != "system"]
            
            # Keep system messages and most recent other messages
            keep_count = self.max_context_length - len(system_messages)
            self.context_window = system_messages + other_messages[-keep_count:]
    
    def clear_context(self):
        """Clear the conversation context"""
        self.context_window = []
        self.logger.debug("Context cleared")
    
    async def execute_task(self, task: Task) -> TaskResult:
        """
        Execute a task using the AI model.
        
        Args:
            task: The task to execute
            
        Returns:
            TaskResult: The result of task execution
        """
        try:
            self.logger.info(f"Executing task: {task.description}")
            
            # Build the prompt
            prompt = self.build_prompt(task)
            
            # Add task to context
            self.add_to_context("user", prompt)
            
            # Generate response using the connector
            response = await self.connector.generate_response(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **task.context
            )
            
            # Add response to context
            self.add_to_context("assistant", response)
            
            # Parse and validate the response
            parsed_result = self.parse_response(response, task)
            
            return TaskResult(
                task_id=task.task_id,
                success=True,
                result=parsed_result,
                metadata={
                    "model_name": self.connector.model_name,
                    "prompt_length": len(prompt),
                    "response_length": len(response),
                    "context_size": len(self.context_window)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return TaskResult(
                task_id=task.task_id,
                success=False,
                result=None,
                error_message=str(e),
                metadata={"model_name": self.connector.model_name}
            )
    
    def build_prompt(self, task: Task) -> str:
        """
        Build a prompt for the AI model based on the task.
        
        Args:
            task: The task to build a prompt for
            
        Returns:
            The constructed prompt string
        """
        prompt_parts = []
        
        # Add system prompt if set
        if self.system_prompt:
            prompt_parts.append(f"System: {self.system_prompt}")
        
        # Add task description
        prompt_parts.append(f"Task: {task.description}")
        
        # Add task type context
        if task.task_type in self.capabilities_list:
            prompt_parts.append(f"Task Type: {task.task_type}")
        
        # Add any additional context from the task
        if task.context:
            context_str = "\n".join([f"{k}: {v}" for k, v in task.context.items()])
            prompt_parts.append(f"Context:\n{context_str}")
        
        return "\n\n".join(prompt_parts)
    
    def parse_response(self, response: str, task: Task) -> Any:
        """
        Parse the AI model response based on the task type.
        
        Args:
            response: The raw response from the AI model
            task: The original task
            
        Returns:
            Parsed result (default is just the response string)
        """
        # Default implementation returns the response as-is
        # Subclasses can override this for specific parsing logic
        return response.strip()
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get status of the underlying AI model"""
        base_status = self.get_status()
        connector_health = await self.connector.health_check()
        
        return {
            **base_status,
            "connector_health": connector_health,
            "context_size": len(self.context_window),
            "system_prompt_set": bool(self.system_prompt),
            "model_parameters": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
        } 