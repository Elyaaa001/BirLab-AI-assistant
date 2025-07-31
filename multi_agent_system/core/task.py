from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import uuid


class TaskStatus(Enum):
    """Status of a task in the system"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """Priority levels for tasks"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Task:
    """
    Represents a task that can be assigned to agents in the system.
    """
    description: str
    task_type: str
    priority: TaskPriority = TaskPriority.MEDIUM
    assigned_agent_id: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    
    # Auto-generated fields
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def start(self):
        """Mark task as started"""
        self.status = TaskStatus.IN_PROGRESS
        self.started_at = datetime.now()
    
    def complete(self):
        """Mark task as completed"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
    
    def fail(self):
        """Mark task as failed"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
    
    def cancel(self):
        """Mark task as cancelled"""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary format"""
        return {
            "task_id": self.task_id,
            "description": self.description,
            "task_type": self.task_type,
            "priority": self.priority.value,
            "assigned_agent_id": self.assigned_agent_id,
            "dependencies": self.dependencies,
            "context": self.context,
            "timeout_seconds": self.timeout_seconds,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass 
class TaskResult:
    """
    Represents the result of a completed task.
    """
    task_id: str
    success: bool
    result: Any
    error_message: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format"""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "result": self.result,
            "error_message": self.error_message,
            "execution_time_seconds": self.execution_time_seconds,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        } 