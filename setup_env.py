#!/usr/bin/env python3
"""
🔑 BirLab AI - Environment Setup Helper

This script helps you set up your API keys and configuration quickly.
"""

import os
import sys
from pathlib import Path


def create_env_file():
    """Create a .env file with API key placeholders"""
    env_content = """# 🌟 BIRLAB AI - API KEYS CONFIGURATION
# Add your actual API keys below (remove the # comments)

# ===== GOOGLE AI (GEMINI) - START HERE! ===== 🌟
# Get your key from: https://aistudio.google.com/app/apikey
#GOOGLE_AI_API_KEY=AIzaSyD-GFjjs0gZ1gpZKX0Wa0c8MWLUrb95sXE

# ===== OTHER AI PROVIDERS (Optional) =====

# 🤖 OpenAI - https://platform.openai.com/api-keys
#OPENAI_API_KEY=your-openai-api-key-here

# 🧠 Anthropic - https://console.anthropic.com/
#ANTHROPIC_API_KEY=your-anthropic-api-key-here

# 💬 Cohere - https://dashboard.cohere.ai/api-keys
#COHERE_API_KEY=your-cohere-api-key-here

# 🤗 Hugging Face - https://huggingface.co/settings/tokens
#HUGGINGFACE_API_KEY=your-huggingface-api-key-here

# 🌪️ Mistral AI - https://console.mistral.ai/
#MISTRAL_API_KEY=your-mistral-api-key-here

# 🤝 Together AI - https://api.together.xyz/settings/api-keys
#TOGETHER_API_KEY=your-together-api-key-here

# 🔄 Replicate - https://replicate.com/account/api-tokens
#REPLICATE_API_TOKEN=your-replicate-api-token-here

# 🚀 Grok (xAI) - https://console.x.ai/
#GROK_API_KEY=your-grok-api-key-here

# 🔍 Perplexity AI - https://www.perplexity.ai/settings/api
#PERPLEXITY_API_KEY=your-perplexity-api-key-here

# 🔬 AI21 Labs - https://studio.ai21.com/account/api-key
#AI21_API_KEY=your-ai21-api-key-here

# ⚡ Groq - https://console.groq.com/keys
#GROQ_API_KEY=your-groq-api-key-here

# 🎆 Fireworks AI - https://fireworks.ai/account/api-keys
#FIREWORKS_API_KEY=your-fireworks-api-key-here

# 🏠 Ollama (Local) - No API key needed
#OLLAMA_BASE_URL=http://localhost:11434
"""
    
    env_path = Path(".env")
    
    if env_path.exists():
        print("⚠️  .env file already exists!")
        response = input("Do you want to overwrite it? (y/N): ").lower().strip()
        if response != 'y':
            print("✅ Keeping existing .env file")
            return
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file with API key placeholders")
    print("📝 Edit .env and uncomment the keys you want to use")


def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import aiohttp
        import asyncio
        print("✅ Core dependencies installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("📦 Run: pip install -r requirements.txt")
        return False


def test_gemini_connection():
    """Test Gemini API connection"""
    try:
        # Try to load from .env first
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("💡 Install python-dotenv for .env file support: pip install python-dotenv")
    
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    
    if not api_key:
        print("\n🔑 No GOOGLE_AI_API_KEY found in environment")
        print("Options:")
        print("1. Set environment variable: export GOOGLE_AI_API_KEY='your-key'")
        print("2. Add to .env file: GOOGLE_AI_API_KEY=your-key")
        print("3. Get key from: https://aistudio.google.com/app/apikey")
        return False
    
    print(f"✅ Found GOOGLE_AI_API_KEY: {api_key[:10]}...{api_key[-4:]}")
    
    # Test basic import
    try:
        from multi_agent_system.connectors import create_gemini_pro_agent
        print("✅ Gemini agent import successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    """Main setup function"""
    print("🌟 Welcome to BirLab AI Setup! 🌟")
    print("This will help you configure your environment\n")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Create .env file
    create_env = input("Create .env file template? (Y/n): ").lower().strip()
    if create_env != 'n':
        create_env_file()
    
    print("\n" + "="*50)
    print("🚀 QUICK START GUIDE")
    print("="*50)
    
    # Test connection
    print("\n1. 🔑 Setting up Google AI API Key:")
    if test_gemini_connection():
        print("   🎉 Ready to go!")
    else:
        print("   📋 Setup needed - see instructions above")
    
    print("\n2. 🧪 Test your setup:")
    print("   export GOOGLE_AI_API_KEY='your-key'")
    print("   python examples/gemini_showcase.py")
    
    print("\n3. 📚 Read the full guide:")
    print("   CONFIG_GUIDE.md")
    
    print("\n🌟 You're ready to unleash Gemini's power! 🚀")


if __name__ == "__main__":
    main() 