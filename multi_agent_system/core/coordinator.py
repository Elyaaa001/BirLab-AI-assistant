import asyncio
from typing import Dict, List, Optional, Any, Callable
import logging
from datetime import datetime

from .agent import Agent, AgentStatus
from .task import Task, TaskResult, TaskPriority, TaskStatus
from .message import Message, MessageType


class CoordinatorAgent(Agent):
    """
    The main coordinator agent that manages and delegates tasks to other agents.
    
    This agent serves as the central controller, receiving high-level tasks,
    breaking them down, and distributing work to appropriate specialist agents.
    """
    
    def __init__(self, agent_id: Optional[str] = None, name: Optional[str] = None):
        super().__init__(agent_id, name or "Coordinator")
        self.registered_agents: Dict[str, Agent] = {}
        self.task_queue: List[Task] = []
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.task_dependencies: Dict[str, List[str]] = {}
        
        # Strategy for task decomposition and assignment
        self.task_decomposition_strategies: Dict[str, Callable] = {}
        
    def get_capabilities(self) -> List[str]:
        """Coordinator can handle coordination and delegation tasks"""
        return ["coordination", "task_delegation", "workflow_management"]
    
    def register_agent(self, agent: Agent):
        """
        Register an agent with the coordinator.
        
        Args:
            agent: The agent to register
        """
        self.registered_agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.name} ({agent.agent_id[:8]})")
    
    def unregister_agent(self, agent_id: str):
        """
        Unregister an agent from the coordinator.
        
        Args:
            agent_id: ID of the agent to unregister
        """
        if agent_id in self.registered_agents:
            agent = self.registered_agents.pop(agent_id)
            self.logger.info(f"Unregistered agent: {agent.name}")
    
    async def find_capable_agents(self, task: Task) -> List[Agent]:
        """
        Find all agents capable of handling a given task.
        
        Args:
            task: The task to find agents for
            
        Returns:
            List of capable agents
        """
        capable_agents = []
        for agent in self.registered_agents.values():
            if await agent.can_handle_task(task):
                capable_agents.append(agent)
        return capable_agents
    
    async def select_best_agent(self, task: Task, capable_agents: List[Agent]) -> Optional[Agent]:
        """
        Select the best agent for a task based on availability and performance.
        
        Args:
            task: The task to assign
            capable_agents: List of capable agents
            
        Returns:
            The selected agent or None if no suitable agent found
        """
        if not capable_agents:
            return None
        
        # Filter out busy agents
        available_agents = [
            agent for agent in capable_agents 
            if agent.status == AgentStatus.IDLE
        ]
        
        if not available_agents:
            return None
        
        # Select agent with best performance (highest success rate and lowest avg execution time)
        def agent_score(agent: Agent) -> float:
            total_tasks = agent.tasks_completed + agent.tasks_failed
            if total_tasks == 0:
                success_rate = 1.0  # New agents get benefit of doubt
                avg_time = 1.0
            else:
                success_rate = agent.tasks_completed / total_tasks
                avg_time = agent.total_execution_time / max(agent.tasks_completed, 1)
            
            # Higher success rate and lower execution time = higher score
            return success_rate / max(avg_time, 0.1)
        
        return max(available_agents, key=agent_score)
    
    async def decompose_task(self, task: Task) -> List[Task]:
        """
        Decompose a complex task into smaller subtasks.
        
        Args:
            task: The task to decompose
            
        Returns:
            List of subtasks
        """
        # Check if we have a custom decomposition strategy for this task type
        if task.task_type in self.task_decomposition_strategies:
            return await self.task_decomposition_strategies[task.task_type](task)
        
        # Default decomposition: return the task as-is
        return [task]
    
    def register_decomposition_strategy(self, task_type: str, strategy: Callable):
        """
        Register a custom task decomposition strategy.
        
        Args:
            task_type: The task type this strategy handles
            strategy: Async function that takes a Task and returns List[Task]
        """
        self.task_decomposition_strategies[task_type] = strategy
        self.logger.info(f"Registered decomposition strategy for task type: {task_type}")
    
    async def execute_task(self, task: Task, preferred_agent: Optional[str] = None) -> TaskResult:
        """
        Execute a coordination task by delegating to appropriate agents.
        
        Args:
            task: The coordination task to execute
            preferred_agent: Optional agent ID to use for execution (bypasses normal selection)
            
        Returns:
            TaskResult: The aggregated result
        """
        try:
            self.logger.info(f"Coordinating task: {task.description}")
            
            # If preferred agent is specified, use it directly
            if preferred_agent and preferred_agent in self.registered_agents:
                agent = self.registered_agents[preferred_agent]
                self.logger.info(f"Using preferred agent: {agent.name}")
                
                # Send task request message
                message = await self.send_message(
                    recipient_id=agent.agent_id,
                    message_type=MessageType.TASK_REQUEST,
                    content=f"Task assignment: {task.description}",
                    metadata={"task": task.to_dict()}
                )
                
                # Execute the task through the preferred agent
                return await agent.start_task(task)
            
            # Decompose the task into subtasks
            subtasks = await self.decompose_task(task)
            
            if len(subtasks) == 1 and subtasks[0].task_id == task.task_id:
                # Task wasn't decomposed, delegate directly
                return await self.delegate_single_task(task)
            else:
                # Execute subtasks and aggregate results
                return await self.execute_workflow(task, subtasks)
                
        except Exception as e:
            self.logger.error(f"Coordination failed: {e}")
            return TaskResult(
                task_id=task.task_id,
                success=False,
                result=None,
                error_message=f"Coordination failed: {str(e)}"
            )
    
    async def delegate_single_task(self, task: Task) -> TaskResult:
        """
        Delegate a single task to an appropriate agent.
        
        Args:
            task: The task to delegate
            
        Returns:
            TaskResult: The result from the executing agent
        """
        capable_agents = await self.find_capable_agents(task)
        selected_agent = await self.select_best_agent(task, capable_agents)
        
        if not selected_agent:
            return TaskResult(
                task_id=task.task_id,
                success=False,
                result=None,
                error_message="No capable agent available for this task"
            )
        
        self.logger.info(f"Delegating task to {selected_agent.name}")
        
        # Send task request message
        message = await self.send_message(
            recipient_id=selected_agent.agent_id,
            message_type=MessageType.TASK_REQUEST,
            content=f"Task assignment: {task.description}",
            metadata={"task": task.to_dict()}
        )
        
        # Execute the task through the selected agent
        return await selected_agent.start_task(task)
    
    async def execute_workflow(self, main_task: Task, subtasks: List[Task]) -> TaskResult:
        """
        Execute a workflow of subtasks and aggregate results.
        
        Args:
            main_task: The original task
            subtasks: List of subtasks to execute
            
        Returns:
            TaskResult: Aggregated result
        """
        results = []
        failed_tasks = []
        
        # Group tasks by dependencies
        ready_tasks = [t for t in subtasks if not t.dependencies]
        pending_tasks = [t for t in subtasks if t.dependencies]
        
        while ready_tasks or pending_tasks:
            # Execute ready tasks
            if ready_tasks:
                batch_results = await asyncio.gather(
                    *[self.delegate_single_task(task) for task in ready_tasks],
                    return_exceptions=True
                )
                
                for task, result in zip(ready_tasks, batch_results):
                    if isinstance(result, Exception):
                        failed_tasks.append(task)
                        self.logger.error(f"Task {task.task_id} failed: {result}")
                    else:
                        results.append(result)
                        if result.success:
                            # Mark task as completed for dependency resolution
                            completed_task_ids = [r.task_id for r in results if r.success]
                            
                            # Move tasks whose dependencies are now satisfied
                            still_pending = []
                            for pending_task in pending_tasks:
                                if all(dep in completed_task_ids for dep in pending_task.dependencies):
                                    ready_tasks.append(pending_task)
                                else:
                                    still_pending.append(pending_task)
                            pending_tasks = still_pending
                
                ready_tasks = [t for t in ready_tasks if t not in [task for task, _ in zip(ready_tasks, batch_results)]]
            
            # If no progress can be made, break
            if not ready_tasks and pending_tasks:
                failed_tasks.extend(pending_tasks)
                break
        
        # Aggregate results
        successful_results = [r for r in results if r.success]
        total_execution_time = sum(r.execution_time_seconds or 0 for r in results)
        
        success = len(failed_tasks) == 0 and len(successful_results) > 0
        aggregated_result = {
            "subtask_results": [r.to_dict() for r in results],
            "successful_tasks": len(successful_results),
            "failed_tasks": len(failed_tasks),
            "total_subtasks": len(subtasks)
        }
        
        return TaskResult(
            task_id=main_task.task_id,
            success=success,
            result=aggregated_result,
            error_message=f"Failed tasks: {len(failed_tasks)}" if failed_tasks else None,
            execution_time_seconds=total_execution_time
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get status of the entire multi-agent system.
        
        Returns:
            System status dictionary
        """
        agent_statuses = {}
        for agent_id, agent in self.registered_agents.items():
            agent_statuses[agent_id] = agent.get_status()
        
        return {
            "coordinator": self.get_status(),
            "registered_agents": len(self.registered_agents),
            "agent_statuses": agent_statuses,
            "queued_tasks": len(self.task_queue),
            "completed_tasks": len(self.completed_tasks),
            "active_agents": len([a for a in self.registered_agents.values() 
                                 if a.status == AgentStatus.BUSY])
        }
    
    async def shutdown(self):
        """
        Gracefully shutdown the coordinator and all registered agents.
        """
        self.logger.info("Shutting down coordinator and all agents...")
        
        # Send shutdown messages to all agents
        shutdown_tasks = []
        for agent in self.registered_agents.values():
            message = await self.send_message(
                recipient_id=agent.agent_id,
                message_type=MessageType.COORDINATION,
                content="shutdown",
                metadata={"action": "shutdown"}
            )
            shutdown_tasks.append(agent.receive_message(message))
        
        # Wait for all agents to receive shutdown messages
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        self.logger.info("Coordinator shutdown complete") 