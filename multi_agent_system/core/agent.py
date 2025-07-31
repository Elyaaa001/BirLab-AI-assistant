from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import asyncio
import uuid
import logging
from datetime import datetime
from enum import Enum

from .message import Message, MessageType
from .task import Task, TaskResult, TaskStatus


class AgentStatus(Enum):
    """Status of an agent"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


class Agent(ABC):
    """
    Abstract base class for all agents in the multi-agent system.
    
    This class defines the common interface that all agents must implement,
    including task execution, message handling, and status management.
    """
    
    def __init__(self, agent_id: Optional[str] = None, name: Optional[str] = None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or f"Agent_{self.agent_id[:8]}"
        self.status = AgentStatus.IDLE
        self.capabilities: List[str] = []
        self.current_tasks: Dict[str, Task] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.logger = logging.getLogger(f"{self.__class__.__name__}[{self.agent_id[:8]}]")
        
        # Performance metrics
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0
        self.created_at = datetime.now()
    
    @abstractmethod
    async def execute_task(self, task: Task) -> TaskResult:
        """
        Execute a given task and return the result.
        
        Args:
            task: The task to execute
            
        Returns:
            TaskResult: The result of task execution
        """
        pass
    
    @abstractmethod 
    def get_capabilities(self) -> List[str]:
        """
        Return a list of capabilities this agent can handle.
        
        Returns:
            List of capability strings
        """
        pass
    
    async def can_handle_task(self, task: Task) -> bool:
        """
        Check if this agent can handle the given task type.
        
        Args:
            task: The task to check
            
        Returns:
            True if agent can handle the task, False otherwise
        """
        return task.task_type in self.get_capabilities()
    
    async def receive_message(self, message: Message):
        """
        Receive and queue a message for processing.
        
        Args:
            message: The message to receive
        """
        await self.message_queue.put(message)
        self.logger.debug(f"Received message from {message.sender_id}: {message.message_type}")
    
    async def send_message(self, recipient_id: str, message_type: MessageType, 
                          content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Create and send a message to another agent.
        
        Args:
            recipient_id: ID of the recipient agent
            message_type: Type of message to send
            content: Message content
            metadata: Optional metadata
            
        Returns:
            The created message
        """
        message = Message(
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            content=content,
            metadata=metadata or {}
        )
        self.logger.debug(f"Sending message to {recipient_id}: {message_type}")
        return message
    
    async def process_messages(self):
        """
        Process messages from the message queue.
        This should be called regularly to handle incoming messages.
        """
        while not self.message_queue.empty():
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
                await self.handle_message(message)
                self.message_queue.task_done()
            except asyncio.TimeoutError:
                break
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
    
    async def handle_message(self, message: Message):
        """
        Handle a received message. Can be overridden by subclasses.
        
        Args:
            message: The message to handle
        """
        if message.message_type == MessageType.TASK_REQUEST:
            # Default task request handling
            self.logger.info(f"Received task request: {message.content}")
        elif message.message_type == MessageType.COORDINATION:
            # Default coordination message handling
            self.logger.info(f"Received coordination message: {message.content}")
        else:
            self.logger.debug(f"Received {message.message_type}: {message.content}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status and metrics.
        
        Returns:
            Dictionary containing agent status information
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status.value,
            "capabilities": self.get_capabilities(),
            "current_tasks": len(self.current_tasks),
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "average_execution_time": (
                self.total_execution_time / max(self.tasks_completed, 1)
            ),
            "uptime_seconds": (datetime.now() - self.created_at).total_seconds(),
            "message_queue_size": self.message_queue.qsize()
        }
    
    async def start_task(self, task: Task) -> TaskResult:
        """
        Start executing a task with proper status management.
        
        Args:
            task: The task to execute
            
        Returns:
            TaskResult: The result of task execution
        """
        if self.status == AgentStatus.BUSY:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                result=None,
                error_message="Agent is currently busy"
            )
        
        self.status = AgentStatus.BUSY
        task.start()
        self.current_tasks[task.task_id] = task
        
        try:
            start_time = datetime.now()
            result = await self.execute_task(task)
            end_time = datetime.now()
            
            execution_time = (end_time - start_time).total_seconds()
            result.execution_time_seconds = execution_time
            
            if result.success:
                task.complete()
                self.tasks_completed += 1
            else:
                task.fail()
                self.tasks_failed += 1
            
            self.total_execution_time += execution_time
            
        except Exception as e:
            task.fail()
            self.tasks_failed += 1
            result = TaskResult(
                task_id=task.task_id,
                success=False,
                result=None,
                error_message=str(e)
            )
            self.logger.error(f"Task execution failed: {e}")
        
        finally:
            self.current_tasks.pop(task.task_id, None)
            self.status = AgentStatus.IDLE
        
        return result
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id[:8]}, name={self.name})"
    
    def __repr__(self) -> str:
        return self.__str__() 