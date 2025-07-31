#!/usr/bin/env python3
"""
Advanced Workflow Example for Multi-Agent AI System

This example demonstrates:
1. Custom task decomposition strategies
2. Complex workflow management with dependencies
3. Agent specialization and coordination
4. Error handling and recovery

This shows how the coordinator can break down complex tasks into subtasks
and coordinate multiple agents to work together on a larger project.
"""

import asyncio
import logging
import os
from typing import List
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


async def software_project_decomposition(main_task: Task) -> List[Task]:
    """
    Custom decomposition strategy for software development projects.
    
    This function breaks down a software project into smaller, manageable tasks
    that can be handled by different specialized agents.
    """
    project_description = main_task.description
    
    # Create subtasks for a typical software project
    subtasks = [
        Task(
            description=f"Create a detailed project specification and requirements document for: {project_description}",
            task_type="analysis",
            priority=TaskPriority.HIGH,
            context={
                **main_task.context,
                "deliverable": "requirements_document",
                "format": "detailed_specification"
            }
        ),
        
        Task(
            description=f"Design the system architecture and data models for: {project_description}",
            task_type="analysis", 
            priority=TaskPriority.HIGH,
            dependencies=[],  # Will be updated with actual task IDs
            context={
                **main_task.context,
                "deliverable": "system_design",
                "include_diagrams": True
            }
        ),
        
        Task(
            description=f"Generate the main application code for: {project_description}",
            task_type="code_generation",
            priority=TaskPriority.MEDIUM,
            dependencies=[],  # Will depend on design task
            context={
                **main_task.context,
                "deliverable": "application_code",
                "include_comments": True,
                "include_error_handling": True
            }
        ),
        
        Task(
            description=f"Create comprehensive tests for: {project_description}",
            task_type="code_generation",
            priority=TaskPriority.MEDIUM,
            dependencies=[],  # Will depend on code generation
            context={
                **main_task.context,
                "deliverable": "test_code",
                "test_types": ["unit", "integration"],
                "include_edge_cases": True
            }
        ),
        
        Task(
            description=f"Write user documentation and README for: {project_description}",
            task_type="creative_writing",
            priority=TaskPriority.LOW,
            dependencies=[],  # Will depend on code generation
            context={
                **main_task.context,
                "deliverable": "documentation",
                "audience": "end_users",
                "include_examples": True
            }
        )
    ]
    
    # Set up dependencies between tasks
    if len(subtasks) >= 2:
        subtasks[1].dependencies = [subtasks[0].task_id]  # Design depends on requirements
    if len(subtasks) >= 3:
        subtasks[2].dependencies = [subtasks[1].task_id]  # Code depends on design
    if len(subtasks) >= 4:
        subtasks[3].dependencies = [subtasks[2].task_id]  # Tests depend on code
    if len(subtasks) >= 5:
        subtasks[4].dependencies = [subtasks[2].task_id]  # Docs depend on code
    
    logger.info(f"Decomposed software project into {len(subtasks)} subtasks")
    return subtasks


async def research_report_decomposition(main_task: Task) -> List[Task]:
    """
    Custom decomposition strategy for research reports.
    """
    topic = main_task.description
    
    subtasks = [
        Task(
            description=f"Conduct background research and gather sources on: {topic}",
            task_type="research",
            priority=TaskPriority.HIGH,
            context={
                **main_task.context,
                "deliverable": "research_sources",
                "source_count": 10
            }
        ),
        
        Task(
            description=f"Analyze and synthesize the research findings on: {topic}",
            task_type="analysis",
            priority=TaskPriority.HIGH,
            dependencies=[],  # Will depend on research task
            context={
                **main_task.context,
                "deliverable": "analysis",
                "include_trends": True
            }
        ),
        
        Task(
            description=f"Write a comprehensive report on: {topic}",
            task_type="creative_writing",
            priority=TaskPriority.MEDIUM,
            dependencies=[],  # Will depend on analysis
            context={
                **main_task.context,
                "deliverable": "final_report",
                "length": "comprehensive",
                "include_citations": True
            }
        )
    ]
    
    # Set dependencies
    if len(subtasks) >= 2:
        subtasks[1].dependencies = [subtasks[0].task_id]
    if len(subtasks) >= 3:
        subtasks[2].dependencies = [subtasks[1].task_id]
    
    logger.info(f"Decomposed research project into {len(subtasks)} subtasks")
    return subtasks


async def create_specialized_multi_agent_system():
    """
    Create a multi-agent system with specialized agents for different domains.
    """
    coordinator = CoordinatorAgent(name="ProjectCoordinator")
    
    # Register custom decomposition strategies
    coordinator.register_decomposition_strategy("software_project", software_project_decomposition)
    coordinator.register_decomposition_strategy("research_report", research_report_decomposition)
    
    agents = []
    
    # Research specialist (Claude)
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            research_agent = create_anthropic_agent(
                model_name="claude-3-sonnet-20240229",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                capabilities=["research", "analysis", "creative_writing"],
                system_prompt="""You are a research specialist with expertise in gathering information, 
                analyzing complex topics, and writing comprehensive reports. Focus on accuracy, 
                depth of analysis, and clear communication."""
            )
            agents.append(("Research Specialist", research_agent))
        except Exception as e:
            logger.warning(f"Could not create research agent: {e}")
    
    # Code specialist (GPT or local model)
    if os.getenv("OPENAI_API_KEY"):
        try:
            code_agent = create_openai_agent(
                model_name="gpt-4",
                api_key=os.getenv("OPENAI_API_KEY"),
                capabilities=["code_generation", "debugging", "analysis"],
                system_prompt="""You are a senior software engineer specializing in code generation, 
                architecture design, and best practices. Write clean, well-documented, and maintainable code."""
            )
            agents.append(("Code Specialist", code_agent))
        except Exception as e:
            logger.warning(f"Could not create code agent: {e}")
    
    # General purpose agent (GPT-3.5)
    if os.getenv("OPENAI_API_KEY"):
        try:
            general_agent = create_openai_agent(
                model_name="gpt-3.5-turbo",
                api_key=os.getenv("OPENAI_API_KEY"),
                capabilities=["text_generation", "analysis", "conversation", "creative_writing"],
                system_prompt="""You are a versatile assistant capable of handling various tasks 
                including analysis, writing, and general problem-solving."""
            )
            agents.append(("General Assistant", general_agent))
        except Exception as e:
            logger.warning(f"Could not create general agent: {e}")
    
    # Register all agents
    for name, agent in agents:
        coordinator.register_agent(agent)
        logger.info(f"Registered {name} with capabilities: {agent.get_capabilities()}")
    
    if not agents:
        logger.error("No agents could be created. Please check your API keys.")
        return None
    
    return coordinator


async def run_complex_project(coordinator, project_type: str, description: str):
    """
    Run a complex project that requires multiple agents working together.
    """
    logger.info(f"\n=== Starting Complex Project: {project_type.upper()} ===")
    logger.info(f"Description: {description}")
    
    # Create the main project task
    main_task = Task(
        description=description,
        task_type=project_type,
        priority=TaskPriority.HIGH,
        context={
            "project_scope": "comprehensive",
            "quality_level": "professional"
        }
    )
    
    # Execute the project through the coordinator
    # The coordinator will automatically decompose it using the registered strategy
    try:
        result = await coordinator.start_task(main_task)
        
        if result.success:
            logger.info(f"✅ Project completed successfully!")
            
            # Display the results from all subtasks
            aggregated_result = result.result
            if isinstance(aggregated_result, dict) and "subtask_results" in aggregated_result:
                logger.info(f"\n--- Project Results Summary ---")
                logger.info(f"Total subtasks: {aggregated_result['total_subtasks']}")
                logger.info(f"Successful: {aggregated_result['successful_tasks']}")
                logger.info(f"Failed: {aggregated_result['failed_tasks']}")
                
                # Show each subtask result
                for i, subtask_result in enumerate(aggregated_result["subtask_results"], 1):
                    if subtask_result["success"]:
                        logger.info(f"\n✅ Subtask {i}: Success")
                        logger.info(f"   Result preview: {str(subtask_result['result'])[:150]}...")
                        if subtask_result.get("metadata", {}).get("model_name"):
                            logger.info(f"   Handled by: {subtask_result['metadata']['model_name']}")
                    else:
                        logger.error(f"\n❌ Subtask {i}: Failed - {subtask_result.get('error_message', 'Unknown error')}")
            
        else:
            logger.error(f"❌ Project failed: {result.error_message}")
            
    except Exception as e:
        logger.error(f"❌ Project execution error: {e}")
    
    return result


async def main():
    """
    Main function demonstrating advanced workflow management.
    """
    logger.info("🚀 Advanced Multi-Agent Workflow Example Starting...")
    
    try:
        # Create the specialized multi-agent system
        coordinator = await create_specialized_multi_agent_system()
        
        if not coordinator:
            logger.error("Failed to create multi-agent system")
            return
        
        # Example 1: Software development project
        await run_complex_project(
            coordinator,
            "software_project",
            "Build a Python web API for a task management system with user authentication, CRUD operations for tasks, and SQLite database storage"
        )
        
        # Example 2: Research report
        await run_complex_project(
            coordinator,
            "research_report", 
            "The impact of artificial intelligence on remote work productivity and employee satisfaction in the post-pandemic era"
        )
        
        # Show final system status
        logger.info("\n=== Final System Status ===")
        system_status = coordinator.get_system_status()
        
        logger.info(f"Total tasks completed: {sum(agent['tasks_completed'] for agent in system_status['agent_statuses'].values())}")
        logger.info(f"Total tasks failed: {sum(agent['tasks_failed'] for agent in system_status['agent_statuses'].values())}")
        
        # Agent performance summary
        logger.info("\n--- Agent Performance ---")
        for agent_id, agent_status in system_status['agent_statuses'].items():
            total_tasks = agent_status['tasks_completed'] + agent_status['tasks_failed']
            if total_tasks > 0:
                success_rate = agent_status['tasks_completed'] / total_tasks * 100
                logger.info(f"{agent_status['name']}: {agent_status['tasks_completed']}/{total_tasks} tasks ({success_rate:.1f}% success)")
        
        # Graceful shutdown
        logger.info("\n🔄 Shutting down system...")
        await coordinator.shutdown()
        logger.info("✅ Advanced workflow example complete")
        
    except Exception as e:
        logger.error(f"❌ Advanced example failed: {e}")
        raise


if __name__ == "__main__":
    # Run the advanced example
    asyncio.run(main()) 