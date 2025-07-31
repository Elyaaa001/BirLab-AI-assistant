#!/usr/bin/env python3
"""
Basic Usage Example for Multi-Agent AI System

This example demonstrates how to:
1. Create a coordinator agent
2. Register multiple AI agents with different capabilities  
3. Submit tasks and let the coordinator delegate them
4. Handle results and monitor the system

Prerequisites:
- Set environment variables for API keys:
  export OPENAI_API_KEY="your-openai-key"
  export ANTHROPIC_API_KEY="your-anthropic-key"
- For Ollama: ensure Ollama is running and models are pulled
"""

import asyncio
import logging
import os
from multi_agent_system import CoordinatorAgent, Task, TaskPriority
from multi_agent_system.connectors import (
    create_openai_agent, 
    create_anthropic_agent, 
    create_ollama_agent
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def create_multi_agent_system():
    """
    Create and configure a multi-agent system with different AI models.
    """
    # Create the coordinator
    coordinator = CoordinatorAgent(name="MainCoordinator")
    
    # Create AI agents with different specializations
    agents = []
    
    # OpenAI GPT for general tasks
    if os.getenv("OPENAI_API_KEY"):
        try:
            gpt_agent = create_openai_agent(
                model_name="gpt-3.5-turbo",
                api_key=os.getenv("OPENAI_API_KEY"),
                capabilities=["text_generation", "analysis", "conversation", "reasoning"],
                system_prompt="You are a helpful assistant specialized in general text analysis and reasoning tasks."
            )
            agents.append(("OpenAI GPT", gpt_agent))
        except Exception as e:
            logger.warning(f"Could not create OpenAI agent: {e}")
    
    # Anthropic Claude for research and analysis
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            claude_agent = create_anthropic_agent(
                model_name="claude-3-haiku-20240307",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                capabilities=["research", "analysis", "creative_writing", "reasoning"],
                system_prompt="You are a research assistant specialized in deep analysis and creative tasks."
            )
            agents.append(("Claude", claude_agent))
        except Exception as e:
            logger.warning(f"Could not create Anthropic agent: {e}")
    
    # Local Ollama model for code generation
    try:
        code_agent = create_ollama_agent(
            model_name="codellama:7b",
            capabilities=["code_generation", "code_explanation", "debugging"],
            system_prompt="You are a coding assistant specialized in generating and explaining code."
        )
        # Check if the model is available
        if await code_agent.connector.validate_connection():
            agents.append(("CodeLlama", code_agent))
        else:
            logger.warning("CodeLlama model not available. Run 'ollama pull codellama:7b' to use this agent.")
    except Exception as e:
        logger.warning(f"Could not create Ollama agent: {e}")
    
    # Register agents with coordinator
    for name, agent in agents:
        coordinator.register_agent(agent)
        logger.info(f"Registered {name} agent with capabilities: {agent.get_capabilities()}")
    
    if not agents:
        logger.error("No agents could be created. Please check your API keys and Ollama setup.")
        return None
    
    return coordinator


async def run_example_tasks(coordinator):
    """
    Run various example tasks through the multi-agent system.
    """
    logger.info("=== Starting Multi-Agent Task Execution ===")
    
    # Define example tasks
    tasks = [
        Task(
            description="Analyze the pros and cons of remote work for software development teams",
            task_type="analysis",
            priority=TaskPriority.HIGH,
            context={"format": "bullet_points", "perspective": "manager"}
        ),
        
        Task(
            description="Write a Python function to calculate the Fibonacci sequence using recursion",
            task_type="code_generation", 
            priority=TaskPriority.MEDIUM,
            context={"language": "python", "include_comments": True}
        ),
        
        Task(
            description="Create a creative story about an AI that becomes conscious",
            task_type="creative_writing",
            priority=TaskPriority.LOW,
            context={"length": "short", "genre": "science_fiction"}
        ),
        
        Task(
            description="Explain the concept of machine learning to a 10-year-old",
            task_type="text_generation",
            priority=TaskPriority.MEDIUM,
            context={"audience": "child", "use_analogies": True}
        )
    ]
    
    # Execute tasks and collect results
    results = []
    for i, task in enumerate(tasks, 1):
        logger.info(f"\n--- Task {i}: {task.description[:50]}... ---")
        
        try:
            result = await coordinator.start_task(task)
            results.append((task, result))
            
            if result.success:
                logger.info(f"✅ Task completed in {result.execution_time_seconds:.2f}s")
                logger.info(f"Result preview: {str(result.result)[:100]}...")
                
                # Show which agent handled the task
                if result.metadata and "model_name" in result.metadata:
                    logger.info(f"Handled by: {result.metadata['model_name']}")
            else:
                logger.error(f"❌ Task failed: {result.error_message}")
                
        except Exception as e:
            logger.error(f"❌ Task execution error: {e}")
    
    return results


async def demonstrate_system_monitoring(coordinator):
    """
    Demonstrate system monitoring and status reporting.
    """
    logger.info("\n=== System Status ===")
    
    # Get overall system status
    system_status = coordinator.get_system_status()
    
    logger.info(f"Registered agents: {system_status['registered_agents']}")
    logger.info(f"Active agents: {system_status['active_agents']}")
    logger.info(f"Completed tasks: {system_status['completed_tasks']}")
    
    # Show individual agent status
    logger.info("\n--- Agent Details ---")
    for agent_id, agent_status in system_status['agent_statuses'].items():
        logger.info(f"Agent: {agent_status['name']}")
        logger.info(f"  Status: {agent_status['status']}")
        logger.info(f"  Capabilities: {', '.join(agent_status['capabilities'])}")
        logger.info(f"  Tasks completed: {agent_status['tasks_completed']}")
        logger.info(f"  Success rate: {agent_status['tasks_completed'] / max(agent_status['tasks_completed'] + agent_status['tasks_failed'], 1) * 100:.1f}%")


async def main():
    """
    Main function to run the complete example.
    """
    logger.info("🚀 Multi-Agent AI System Example Starting...")
    
    try:
        # Create the multi-agent system
        coordinator = await create_multi_agent_system()
        
        if not coordinator:
            logger.error("Failed to create multi-agent system")
            return
        
        # Run example tasks
        results = await run_example_tasks(coordinator)
        
        # Monitor system status
        await demonstrate_system_monitoring(coordinator)
        
        # Display final results summary
        logger.info(f"\n=== Final Results Summary ===")
        successful_tasks = sum(1 for _, result in results if result.success)
        total_tasks = len(results)
        
        logger.info(f"Successfully completed: {successful_tasks}/{total_tasks} tasks")
        
        if successful_tasks > 0:
            total_time = sum(result.execution_time_seconds or 0 for _, result in results if result.success)
            avg_time = total_time / successful_tasks
            logger.info(f"Average execution time: {avg_time:.2f} seconds")
        
        # Graceful shutdown
        logger.info("\n🔄 Shutting down system...")
        await coordinator.shutdown()
        logger.info("✅ System shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise


if __name__ == "__main__":
    # Run the example
    asyncio.run(main()) 