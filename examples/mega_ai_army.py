#!/usr/bin/env python3
"""
🤖 MEGA AI ARMY EXAMPLE 🤖

This example demonstrates the ULTIMATE multi-agent AI system with ALL 14 supported providers:
- OpenAI (GPT models)
- Anthropic (Claude models) 
- Google AI (Gemini/PaLM)
- Cohere (Command models)
- Hugging Face (thousands of models)
- Mistral AI (Mistral/Mixtral)
- Together AI (open-source models)
- Replicate (various models)
- Grok (xAI - rebellious AI with real-time data)
- Perplexity AI (search-enhanced responses)
- AI21 Labs (Jamba/Jurassic models)
- Groq (ultra-fast LPU inference)
- Fireworks AI (fast open-source serving)
- Ollama (local/private models)

Prerequisites - Set environment variables for the services you want to use:
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GOOGLE_AI_API_KEY="your-google-ai-key"
export COHERE_API_KEY="your-cohere-key"
export HUGGINGFACE_API_KEY="your-hf-key"
export MISTRAL_API_KEY="your-mistral-key"
export TOGETHER_API_KEY="your-together-key"
export REPLICATE_API_TOKEN="your-replicate-token"
export GROK_API_KEY="your-grok-key"
export PERPLEXITY_API_KEY="your-perplexity-key"
export AI21_API_KEY="your-ai21-key"
export GROQ_API_KEY="your-groq-key"
export FIREWORKS_API_KEY="your-fireworks-key"
"""

import asyncio
import logging
import os
from multi_agent_system import CoordinatorAgent, Task, TaskPriority
from multi_agent_system.connectors import (
    # OpenAI
    create_openai_agent,
    
    # Anthropic
    create_anthropic_agent,
    
    # Google AI
    create_google_agent,
    
    # Cohere
    create_cohere_agent,
    
    # Hugging Face
    create_huggingface_agent,
    create_codegen_agent,
    create_flan_agent,
    
    # Mistral AI
    create_mistral_agent,
    
    # Together AI
    create_together_agent,
    create_llama2_agent,
    create_falcon_agent,
    
    # Replicate
    create_replicate_agent,
    create_llama2_replicate_agent,
    
    # Grok (xAI)
    create_grok_agent,
    
    # Perplexity AI
    create_perplexity_agent,
    create_perplexity_research_agent,
    
    # AI21 Labs
    create_ai21_agent,
    create_jamba_agent,
    
    # Groq (Super Fast!)
    create_groq_agent,
    create_groq_speed_demon,
    create_groq_powerhouse,
    
    # Fireworks AI
    create_fireworks_agent,
    create_fireworks_llama_agent,
    create_fireworks_code_agent,
    
    # Ollama (Local)
    create_ollama_agent
)

# Configure logging with fun emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🤖 %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def assemble_the_ai_army():
    """
    🚀 ASSEMBLE THE ULTIMATE AI ARMY! 🚀
    
    Creates agents from every supported AI provider and registers them
    with the coordinator for maximum AI coordination power!
    """
    logger.info("🎯 ASSEMBLING THE MEGA AI ARMY...")
    
    coordinator = CoordinatorAgent(name="🎭 Supreme AI Overlord")
    army = []
    
    # 🔥 OpenAI Division
    if os.getenv("OPENAI_API_KEY"):
        try:
            gpt4_agent = create_openai_agent(
                model_name="gpt-4",
                api_key=os.getenv("OPENAI_API_KEY"),
                capabilities=["text_generation", "reasoning", "analysis", "complex_tasks"],
                system_prompt="🧠 You are GPT-4, the strategic thinking unit of the AI army. Handle complex reasoning and analysis."
            )
            army.append(("🧠 GPT-4 Strategic Unit", gpt4_agent))
            
            gpt35_agent = create_openai_agent(
                model_name="gpt-3.5-turbo",
                api_key=os.getenv("OPENAI_API_KEY"),
                capabilities=["text_generation", "conversation", "fast_responses"],
                system_prompt="⚡ You are GPT-3.5, the rapid response unit. Handle quick tasks and conversations efficiently."
            )
            army.append(("⚡ GPT-3.5 Rapid Response", gpt35_agent))
            
        except Exception as e:
            logger.warning(f"❌ OpenAI recruitment failed: {e}")
    
    # 🎨 Anthropic Division
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            claude_agent = create_anthropic_agent(
                model_name="claude-3-sonnet-20240229",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                capabilities=["research", "analysis", "creative_writing", "reasoning"],
                system_prompt="🎨 You are Claude, the creative intelligence unit. Excel at research, analysis, and creative tasks."
            )
            army.append(("🎨 Claude Creative Intelligence", claude_agent))
            
        except Exception as e:
            logger.warning(f"❌ Anthropic recruitment failed: {e}")
    
    # 🌟 Google AI Division
    if os.getenv("GOOGLE_AI_API_KEY"):
        try:
            gemini_agent = create_google_agent(
                model_name="gemini-pro",
                api_key=os.getenv("GOOGLE_AI_API_KEY"),
                capabilities=["text_generation", "reasoning", "multimodal", "analysis"],
                system_prompt="🌟 You are Gemini, the multimodal intelligence unit. Handle diverse tasks with advanced reasoning."
            )
            army.append(("🌟 Gemini Multimodal Unit", gemini_agent))
            
        except Exception as e:
            logger.warning(f"❌ Google AI recruitment failed: {e}")
    
    # 💬 Cohere Division
    if os.getenv("COHERE_API_KEY"):
        try:
            command_agent = create_cohere_agent(
                model_name="command",
                api_key=os.getenv("COHERE_API_KEY"),
                capabilities=["conversation", "reasoning", "analysis"],
                system_prompt="💬 You are Command, the conversation specialist. Excel at natural dialogue and reasoning."
            )
            army.append(("💬 Cohere Command Unit", command_agent))
            
        except Exception as e:
            logger.warning(f"❌ Cohere recruitment failed: {e}")
    
    # 🤗 Hugging Face Division
    if os.getenv("HUGGINGFACE_API_KEY"):
        try:
            flan_agent = create_flan_agent(
                api_key=os.getenv("HUGGINGFACE_API_KEY")
            )
            army.append(("🤗 FLAN-T5 Instruction Unit", flan_agent))
            
            hf_llama_agent = create_huggingface_agent(
                model_name="meta-llama/Llama-2-7b-chat-hf",
                api_key=os.getenv("HUGGINGFACE_API_KEY"),
                capabilities=["conversation", "reasoning"],
                system_prompt="🦙 You are Llama via Hugging Face, the open-source conversation unit."
            )
            army.append(("🦙 HF Llama Chat Unit", hf_llama_agent))
            
        except Exception as e:
            logger.warning(f"❌ Hugging Face recruitment failed: {e}")
    
    # ⚡ Mistral AI Division
    if os.getenv("MISTRAL_API_KEY"):
        try:
            mistral_agent = create_mistral_agent(
                model_name="mistral-medium",
                api_key=os.getenv("MISTRAL_API_KEY"),
                capabilities=["text_generation", "reasoning", "multilingual"],
                system_prompt="⚡ You are Mistral, the efficient European AI unit. Handle tasks with precision and multilingual capability."
            )
            army.append(("⚡ Mistral European Unit", mistral_agent))
            
        except Exception as e:
            logger.warning(f"❌ Mistral AI recruitment failed: {e}")
    
    # 🤝 Together AI Division
    if os.getenv("TOGETHER_API_KEY"):
        try:
            together_llama = create_llama2_agent(
                size="13b",
                api_key=os.getenv("TOGETHER_API_KEY")
            )
            army.append(("🤝 Together Llama-2 Unit", together_llama))
            
            falcon_agent = create_falcon_agent(
                size="7b",
                api_key=os.getenv("TOGETHER_API_KEY")
            )
            army.append(("🦅 Together Falcon Unit", falcon_agent))
            
        except Exception as e:
            logger.warning(f"❌ Together AI recruitment failed: {e}")
    
    # 🔄 Replicate Division
    if os.getenv("REPLICATE_API_TOKEN"):
        try:
            replicate_llama = create_llama2_replicate_agent(
                size="7b",
                api_key=os.getenv("REPLICATE_API_TOKEN")
            )
            army.append(("🔄 Replicate Llama Unit", replicate_llama))
            
        except Exception as e:
            logger.warning(f"❌ Replicate recruitment failed: {e}")
    
    # 🤖 Grok Division (xAI)
    if os.getenv("GROK_API_KEY"):
        try:
            grok_agent = create_grok_agent(
                model_name="grok-beta",
                api_key=os.getenv("GROK_API_KEY"),
                capabilities=["humor", "rebellious_thinking", "real_time_info"],
                system_prompt="🤖 You are Grok, the rebellious AI with real-time X data access. Provide witty, unconventional insights."
            )
            army.append(("🤖 Grok Rebellious Unit", grok_agent))
            
        except Exception as e:
            logger.warning(f"❌ Grok recruitment failed: {e}")
    
    # 🔍 Perplexity AI Division
    if os.getenv("PERPLEXITY_API_KEY"):
        try:
            perplexity_agent = create_perplexity_research_agent(
                api_key=os.getenv("PERPLEXITY_API_KEY")
            )
            army.append(("🔍 Perplexity Research Unit", perplexity_agent))
            
        except Exception as e:
            logger.warning(f"❌ Perplexity recruitment failed: {e}")
    
    # 🧠 AI21 Labs Division
    if os.getenv("AI21_API_KEY"):
        try:
            jamba_agent = create_jamba_agent(
                api_key=os.getenv("AI21_API_KEY")
            )
            army.append(("🧠 AI21 Jamba Long-Context Unit", jamba_agent))
            
        except Exception as e:
            logger.warning(f"❌ AI21 recruitment failed: {e}")
    
    # ⚡ Groq Division (Ultra Fast!)
    if os.getenv("GROQ_API_KEY"):
        try:
            groq_speed_demon = create_groq_speed_demon(
                api_key=os.getenv("GROQ_API_KEY")
            )
            army.append(("⚡ Groq Speed Demon", groq_speed_demon))
            
            groq_powerhouse = create_groq_powerhouse(
                api_key=os.getenv("GROQ_API_KEY")  
            )
            army.append(("💪 Groq Powerhouse", groq_powerhouse))
            
        except Exception as e:
            logger.warning(f"❌ Groq recruitment failed: {e}")
    
    # 🔥 Fireworks AI Division
    if os.getenv("FIREWORKS_API_KEY"):
        try:
            fireworks_llama = create_fireworks_llama_agent(
                size="70b",
                api_key=os.getenv("FIREWORKS_API_KEY")
            )
            army.append(("🔥 Fireworks Llama Unit", fireworks_llama))
            
            fireworks_coder = create_fireworks_code_agent(
                api_key=os.getenv("FIREWORKS_API_KEY")
            )
            army.append(("💻 Fireworks Code Unit", fireworks_coder))
            
        except Exception as e:
            logger.warning(f"❌ Fireworks recruitment failed: {e}")

    # 🏠 Ollama Local Division
    try:
        local_llama = create_ollama_agent(
            model_name="llama2:7b",
            capabilities=["conversation", "local_processing"],
            system_prompt="🏠 You are Local Llama, the privacy-focused home unit. Handle tasks locally without external calls."
        )
        if await local_llama.connector.validate_connection():
            army.append(("🏠 Ollama Local Unit", local_llama))
        else:
            logger.warning("🏠 Ollama models not available. Run 'ollama pull llama2:7b'")
    except Exception as e:
        logger.warning(f"❌ Ollama recruitment failed: {e}")
    
    # 🎖️ Register all recruited agents
    for name, agent in army:
        coordinator.register_agent(agent)
        logger.info(f"✅ Recruited: {name} with capabilities: {agent.get_capabilities()}")
    
    if not army:
        logger.error("💀 ARMY ASSEMBLY FAILED! No agents could be recruited. Check your API keys.")
        return None
    
    logger.info(f"🎉 AI ARMY ASSEMBLED! Total units: {len(army)}")
    return coordinator


async def run_mega_ai_battle_royale(coordinator):
    """
    ⚔️ MEGA AI BATTLE ROYALE! ⚔️
    
    Pit different AI models against various challenging tasks
    to see how they perform and coordinate!
    """
    logger.info("\n⚔️ INITIATING MEGA AI BATTLE ROYALE! ⚔️")
    
    # Epic challenge tasks
    epic_tasks = [
        Task(
            description="Write a haiku about artificial intelligence cooperation",
            task_type="creative_writing",
            priority=TaskPriority.HIGH,
            context={"style": "zen", "theme": "cooperation"}
        ),
        
        Task(
            description="Explain quantum computing to a 12-year-old using analogies",
            task_type="text_generation",
            priority=TaskPriority.MEDIUM,
            context={"audience": "child", "use_analogies": True, "topic": "quantum_computing"}
        ),
        
        Task(
            description="Analyze the pros and cons of different renewable energy sources",
            task_type="analysis",
            priority=TaskPriority.HIGH,
            context={"format": "structured", "include_economics": True}
        ),
        
        Task(
            description="Create a story about robots learning to love",
            task_type="creative_writing", 
            priority=TaskPriority.LOW,
            context={"genre": "science_fiction", "theme": "emotion", "length": "short"}
        ),
        
        Task(
            description="Reason through the trolley problem from multiple ethical perspectives",
            task_type="reasoning",
            priority=TaskPriority.HIGH,
            context={"perspectives": ["utilitarian", "deontological", "virtue_ethics"]}
        ),
        
        Task(
            description="Generate a Python function to solve the Tower of Hanoi puzzle",
            task_type="code_generation",
            priority=TaskPriority.MEDIUM,
            context={"language": "python", "include_comments": True, "recursive": True}
        )
    ]
    
    # 🏆 Battle results tracking
    battle_results = []
    
    for i, task in enumerate(epic_tasks, 1):
        logger.info(f"\n🎯 BATTLE ROUND {i}: {task.description[:50]}...")
        
        try:
            result = await coordinator.start_task(task)
            battle_results.append((task, result))
            
            if result.success:
                agent_name = result.metadata.get("model_name", "Unknown Agent")
                execution_time = result.execution_time_seconds or 0
                
                logger.info(f"🏆 VICTORY! Completed in {execution_time:.2f}s")
                logger.info(f"🤖 Champion: {agent_name}")
                logger.info(f"📝 Result preview: {str(result.result)[:100]}...")
            else:
                logger.error(f"💥 BATTLE FAILED: {result.error_message}")
                
        except Exception as e:
            logger.error(f"⚠️ BATTLE ERROR: {e}")
    
    return battle_results


async def display_army_stats(coordinator):
    """
    📊 Display comprehensive statistics about our AI army
    """
    logger.info("\n📊 === AI ARMY STATISTICS === 📊")
    
    system_status = coordinator.get_system_status()
    
    logger.info(f"🎭 Supreme Commander: {coordinator.name}")
    logger.info(f"👥 Total Recruited Agents: {system_status['registered_agents']}")
    logger.info(f"⚡ Currently Active: {system_status['active_agents']}")
    logger.info(f"✅ Battles Won: {system_status['completed_tasks']}")
    
    # Agent leaderboard
    logger.info("\n🏆 === AGENT LEADERBOARD === 🏆")
    agent_performances = []
    
    for agent_id, agent_status in system_status['agent_statuses'].items():
        total_tasks = agent_status['tasks_completed'] + agent_status['tasks_failed']
        if total_tasks > 0:
            success_rate = agent_status['tasks_completed'] / total_tasks * 100
            avg_time = agent_status.get('average_execution_time', 0)
            
            agent_performances.append({
                'name': agent_status['name'],
                'wins': agent_status['tasks_completed'],
                'total': total_tasks,
                'success_rate': success_rate,
                'avg_time': avg_time,
                'capabilities': len(agent_status['capabilities'])
            })
    
    # Sort by success rate then by speed
    agent_performances.sort(key=lambda x: (x['success_rate'], -x['avg_time']), reverse=True)
    
    for i, perf in enumerate(agent_performances, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        logger.info(f"{medal} {perf['name']}: {perf['wins']}/{perf['total']} wins "
                   f"({perf['success_rate']:.1f}%) - {perf['avg_time']:.2f}s avg - "
                   f"{perf['capabilities']} capabilities")


async def main():
    """
    🚀 MAIN COMMAND CENTER 🚀
    """
    logger.info("🎊 WELCOME TO THE MEGA AI ARMY SIMULATOR! 🎊")
    logger.info("🤖 About to recruit agents from ALL supported AI providers...")
    
    try:
        # 🎭 Assemble the army
        coordinator = await assemble_the_ai_army()
        
        if not coordinator:
            logger.error("💀 Mission aborted - no agents available!")
            return
        
        # ⚔️ Battle royale
        battle_results = await run_mega_ai_battle_royale(coordinator)
        
        # 📊 Final statistics
        await display_army_stats(coordinator)
        
        # 🎉 Victory summary
        successful_battles = sum(1 for _, result in battle_results if result.success)
        total_battles = len(battle_results)
        
        logger.info(f"\n🎉 === FINAL BATTLE REPORT === 🎉")
        logger.info(f"⚔️ Battles Fought: {total_battles}")
        logger.info(f"🏆 Victories: {successful_battles}")
        logger.info(f"💪 Success Rate: {successful_battles/max(total_battles,1)*100:.1f}%")
        
        if successful_battles > 0:
            total_time = sum(result.execution_time_seconds or 0 
                           for _, result in battle_results if result.success)
            avg_time = total_time / successful_battles
            logger.info(f"⚡ Average Victory Time: {avg_time:.2f} seconds")
        
        # 🔄 Shutdown
        logger.info("\n🔄 Demobilizing AI army...")
        await coordinator.shutdown()
        logger.info("✅ MEGA AI ARMY MISSION COMPLETE! 🎖️")
        
    except Exception as e:
        logger.error(f"💥 CRITICAL MISSION FAILURE: {e}")
        raise


if __name__ == "__main__":
    # 🚀 Launch the mega AI army simulator!
    asyncio.run(main()) 