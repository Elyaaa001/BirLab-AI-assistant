#!/usr/bin/env python3
"""
🌟 BIRLAB AI - 100+ AI MODELS ARMY DEMO 🌟

This demo showcases the massive BirLab AI system with automatic registration
of 100+ AI models from multiple providers, intelligent agent selection,
and specialized task execution.

Features demonstrated:
- 🤖 100+ AI models from 8+ providers
- 🧠 Intelligent agent selection
- 🎯 Specialized task routing
- 📊 Comprehensive analytics
- ⚡ Performance comparison
- 🌟 Provider diversity

Usage:
    python examples/birlab_ai_army_demo.py
"""

import asyncio
import time
import sys
import os
from typing import Dict, List, Any
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from multi_agent_system.core.birlab_coordinator import create_birlab_ultra_coordinator
from multi_agent_system.core.task import Task, TaskPriority
from multi_agent_system.connectors.expanded_models import (
    BIRLAB_AI_MODELS, 
    get_all_birlab_ai_models,
    get_models_by_provider,
    get_models_by_capability,
    print_birlab_ai_registry
)

# Demo configuration
DEMO_TASKS = {
    "coding": [
        "Write a Python function to calculate fibonacci numbers recursively",
        "Create a REST API endpoint using FastAPI for user management",
        "Implement a binary search tree in JavaScript",
        "Write a SQL query to find top 5 customers by purchase amount"
    ],
    "analysis": [
        "Analyze the pros and cons of renewable energy adoption",
        "Compare the economic impact of remote work vs office work",
        "Evaluate the security implications of cloud computing",
        "Assess the market potential for electric vehicles in 2024"
    ],
    "creative": [
        "Write a short story about AI discovering emotions",
        "Create a marketing slogan for a sustainable tech company",
        "Design a lesson plan for teaching programming to kids",
        "Brainstorm 10 innovative app ideas for environmental conservation"
    ],
    "research": [
        "Summarize the latest developments in quantum computing",
        "Research the benefits of Mediterranean diet on health",
        "Investigate the impact of social media on mental health",
        "Explore the potential of blockchain in supply chain management"
    ],
    "fast": [
        "What is the capital of Australia?",
        "Convert 100 USD to EUR",
        "Name 5 programming languages",
        "What year was Python created?"
    ]
}


class BirLabArmyDemo:
    """Demo class for showcasing BirLab AI's 100+ model army"""
    
    def __init__(self):
        self.coordinator = None
        self.demo_results = {}
        self.performance_stats = {}
        
    async def initialize(self):
        """Initialize the BirLab AI Ultra Coordinator"""
        print("🚀 Initializing BirLab AI Army Demo...")
        print("=" * 60)
        
        # Create ultra coordinator
        self.coordinator = create_birlab_ultra_coordinator()
        
        if not self.coordinator.registered_models:
            print("⚠️ No AI models were registered!")
            print("💡 Make sure you have API keys set in your .env file")
            print("🔧 Run: python setup_env.py")
            return False
        
        return True
    
    async def demo_intelligent_selection(self):
        """Demonstrate intelligent agent selection for different task types"""
        print("\n🎯 INTELLIGENT AGENT SELECTION DEMO")
        print("=" * 50)
        
        selection_results = {}
        
        for task_type in ["coding", "analysis", "multimodal", "fast", "local"]:
            print(f"\n📋 Task Type: {task_type.upper()}")
            
            # Get best agent
            best_agent = self.coordinator.get_best_agent_for_task(task_type)
            
            if best_agent:
                model_id = None
                for mid, data in self.coordinator.registered_models.items():
                    if data["agent_id"] == best_agent:
                        model_id = mid
                        break
                
                if model_id:
                    model_info = self.coordinator.registered_models[model_id]["model_info"]
                    print(f"✅ Selected: {model_info['name']}")
                    print(f"   Provider: {model_info['provider']}")
                    print(f"   Capabilities: {', '.join(model_info.get('capabilities', []))}")
                    print(f"   Context: {model_info.get('context_length', 0):,} tokens")
                    
                    selection_results[task_type] = {
                        "agent_id": best_agent,
                        "model_name": model_info['name'],
                        "provider": model_info['provider']
                    }
            else:
                print(f"❌ No suitable agent found for {task_type}")
                selection_results[task_type] = None
        
        return selection_results
    
    async def demo_task_execution(self):
        """Execute sample tasks with different agents"""
        print("\n🏃‍♂️ TASK EXECUTION DEMO")
        print("=" * 40)
        
        execution_results = {}
        
        for task_type, tasks in DEMO_TASKS.items():
            print(f"\n📋 {task_type.upper()} TASKS:")
            
            # Get best agent for this task type
            best_agent = self.coordinator.get_best_agent_for_task(task_type)
            
            if not best_agent:
                print(f"❌ No agent available for {task_type}")
                continue
            
            # Execute first task as demo
            task_description = tasks[0]
            print(f"🎯 Task: {task_description}")
            
            try:
                # Create and execute task
                task = Task(
                    description=task_description,
                    priority=TaskPriority.MEDIUM
                )
                
                start_time = time.time()
                result = await self.coordinator.execute_task(task, preferred_agent=best_agent)
                execution_time = time.time() - start_time
                
                # Get model info
                model_name = "Unknown"
                provider = "Unknown"
                for model_id, data in self.coordinator.registered_models.items():
                    if data["agent_id"] == best_agent:
                        model_info = data["model_info"]
                        model_name = model_info["name"]
                        provider = model_info["provider"]
                        break
                
                print(f"✅ Agent: {model_name}")
                print(f"⏱️ Time: {execution_time:.2f}s")
                print(f"📝 Result: {result.content[:200]}...")
                
                execution_results[task_type] = {
                    "task": task_description,
                    "agent": model_name,
                    "provider": provider,
                    "time": execution_time,
                    "success": True
                }
                
            except Exception as e:
                print(f"❌ Error: {e}")
                execution_results[task_type] = {
                    "task": task_description,
                    "error": str(e),
                    "success": False
                }
        
        return execution_results
    
    async def demo_provider_comparison(self):
        """Compare performance across different providers"""
        print("\n🏆 PROVIDER COMPARISON DEMO")
        print("=" * 45)
        
        # Get agent army stats
        stats = self.coordinator.get_agent_army_stats()
        
        print(f"📊 PROVIDER BREAKDOWN:")
        for provider, count in sorted(stats["providers"].items()):
            print(f"  • {provider.title()}: {count} models")
        
        print(f"\n🎯 TOP CAPABILITIES:")
        for capability, count in stats["top_capabilities"][:10]:
            print(f"  • {capability}: {count} agents")
        
        print(f"\n📖 CONTEXT LENGTH DISTRIBUTION:")
        for size, count in stats["context_distribution"].items():
            print(f"  • {size.title()}: {count} agents")
        
        return stats
    
    async def demo_specialized_agents(self):
        """Demonstrate specialized agent capabilities"""
        print("\n🔧 SPECIALIZED AGENTS DEMO")
        print("=" * 42)
        
        specializations = {
            "Coding Agents": get_models_by_capability("code_generation"),
            "Multimodal Agents": get_models_by_capability("multimodal"),
            "Fast Response Agents": get_models_by_capability("fast_responses"),
            "Long Context Agents": get_models_by_capability("ultra_long_context"),
            "Local Privacy Agents": get_models_by_capability("local_inference")
        }
        
        for category, model_ids in specializations.items():
            print(f"\n🎯 {category}: {len(model_ids)} agents")
            
            # Show top 3 examples
            for model_id in model_ids[:3]:
                if model_id in BIRLAB_AI_MODELS:
                    model_info = BIRLAB_AI_MODELS[model_id]
                    print(f"  • {model_info['name']} ({model_info['provider']})")
            
            if len(model_ids) > 3:
                print(f"  ... and {len(model_ids) - 3} more")
    
    async def demo_load_balancing(self):
        """Demonstrate load balancing across multiple agents"""
        print("\n⚖️ LOAD BALANCING DEMO")
        print("=" * 38)
        
        # Simulate multiple concurrent tasks
        tasks = [
            "What is artificial intelligence?",
            "Explain quantum computing",
            "Write a hello world program",
            "Describe machine learning",
            "What is blockchain technology?"
        ]
        
        print(f"🚀 Executing {len(tasks)} concurrent tasks...")
        
        concurrent_tasks = []
        for i, task_desc in enumerate(tasks):
            task = Task(
                description=f"Task {i+1}: {task_desc}",
                priority=TaskPriority.MEDIUM
            )
            
            # Let coordinator choose best agent
            task_coroutine = self.coordinator.execute_task(task)
            concurrent_tasks.append(task_coroutine)
        
        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        total_time = time.time() - start_time
        
        print(f"✅ Completed {len(tasks)} tasks in {total_time:.2f}s")
        print(f"⚡ Average: {total_time/len(tasks):.2f}s per task")
        
        # Show which agents were used
        agents_used = {}
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                agent_id = result.agent_id or "unknown"
                agents_used[agent_id] = agents_used.get(agent_id, 0) + 1
        
        print(f"\n📊 Agent Distribution:")
        for agent_id, count in agents_used.items():
            print(f"  • {agent_id}: {count} tasks")
    
    def print_final_summary(self):
        """Print comprehensive demo summary"""
        print("\n" + "=" * 60)
        print("🌟 BIRLAB AI ARMY DEMO - FINAL SUMMARY 🌟")
        print("=" * 60)
        
        if not self.coordinator:
            print("❌ Demo failed to initialize")
            return
        
        stats = self.coordinator.get_agent_army_stats()
        
        print(f"🤖 Total AI Agents: {stats['total_agents']}")
        print(f"🏢 Active Providers: {len(stats['providers'])}")
        print(f"🎯 Unique Capabilities: {len(stats['capabilities'])}")
        
        print(f"\n🏆 TOP PROVIDERS:")
        for provider, count in sorted(stats['providers'].items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  🥇 {provider.title()}: {count} models")
        
        print(f"\n⚡ PERFORMANCE HIGHLIGHTS:")
        print(f"  • Intelligent agent selection ✅")
        print(f"  • Multi-provider load balancing ✅")
        print(f"  • Specialized task routing ✅")
        print(f"  • Concurrent task execution ✅")
        
        print(f"\n🚀 BIRLAB AI IS READY FOR MAXIMUM COORDINATION!")
        print("=" * 60)


async def main():
    """Main demo function"""
    print("🌟 WELCOME TO BIRLAB AI - 100+ MODELS ARMY DEMO! 🌟")
    print()
    
    demo = BirLabArmyDemo()
    
    # Initialize
    if not await demo.initialize():
        print("❌ Demo initialization failed!")
        return
    
    # Run all demos
    try:
        print("\n🎬 Starting comprehensive demo...")
        
        # 1. Show intelligent selection
        await demo.demo_intelligent_selection()
        
        # 2. Execute sample tasks
        await demo.demo_task_execution()
        
        # 3. Compare providers
        await demo.demo_provider_comparison()
        
        # 4. Show specialized agents
        await demo.demo_specialized_agents()
        
        # 5. Demonstrate load balancing
        await demo.demo_load_balancing()
        
        # Final summary
        demo.print_final_summary()
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check for API keys
    if not any(os.getenv(key) for key in [
        "GOOGLE_AI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", 
        "COHERE_API_KEY", "MISTRAL_API_KEY", "HUGGINGFACE_API_KEY"
    ]):
        print("⚠️ WARNING: No API keys detected!")
        print("🔧 Please run: python setup_env.py")
        print("💡 Or set API keys in your .env file")
        print()
    
    # Run demo
    asyncio.run(main()) 