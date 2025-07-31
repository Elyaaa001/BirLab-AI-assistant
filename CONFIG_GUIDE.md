# 🔑 **API KEYS & CONFIGURATION GUIDE** 🔑

This guide shows you **exactly** where to put your API keys and how to configure BirLab AI.

## 🚀 **QUICK START - GEMINI SETUP**

### **Method 1: Environment Variables (Recommended)**
```bash
# Set in your terminal/shell
export GOOGLE_AI_API_KEY="your-actual-google-ai-key-here"

# Then run any example
python examples/gemini_showcase.py
```

### **Method 2: .env File (Persistent)**
Create a `.env` file in the project root:
```bash
# Create .env file
touch .env

# Add your keys
echo "GOOGLE_AI_API_KEY=your-actual-google-ai-key-here" >> .env
```

### **Method 3: Direct in Code (Quick Testing)**
```python
# In your Python script
from multi_agent_system.connectors import create_gemini_pro_agent

gemini_pro = create_gemini_pro_agent(api_key="your-actual-google-ai-key-here")
```

## 🌟 **GET YOUR GOOGLE AI API KEY**

1. **Go to**: https://aistudio.google.com/app/apikey
2. **Sign in** with your Google account
3. **Click "Create API Key"**
4. **Copy the key** and use it in your configuration

## 📋 **COMPLETE .env FILE TEMPLATE**

Create a `.env` file in your project root with these contents:

```bash
# 🌟 GOOGLE AI (GEMINI) - Primary focus
GOOGLE_AI_API_KEY=your-google-ai-api-key-here

# 🤖 OTHER AI PROVIDERS (Optional)
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
COHERE_API_KEY=your-cohere-api-key-here
HUGGINGFACE_API_KEY=your-huggingface-api-key-here
MISTRAL_API_KEY=your-mistral-api-key-here
TOGETHER_API_KEY=your-together-api-key-here
REPLICATE_API_TOKEN=your-replicate-api-token-here
GROK_API_KEY=your-grok-api-key-here
PERPLEXITY_API_KEY=your-perplexity-api-key-here
AI21_API_KEY=your-ai21-api-key-here
GROQ_API_KEY=your-groq-api-key-here
FIREWORKS_API_KEY=your-fireworks-api-key-here

# 🏠 LOCAL MODEL (No API key needed)
OLLAMA_BASE_URL=http://localhost:11434
```

## 🎯 **USAGE EXAMPLES BY METHOD**

### **Environment Variables**
```bash
# Terminal/Shell setup
export GOOGLE_AI_API_KEY="your-key"
python examples/gemini_showcase.py
```

### **Python Code with Direct API Key**
```python
from multi_agent_system.connectors import (
    create_gemini_pro_agent,
    create_gemini_flash_agent,
    create_gemini_vision_agent
)

# Method 1: Direct API key
gemini_pro = create_gemini_pro_agent(api_key="your-key-here")

# Method 2: Using environment variable
import os
gemini_flash = create_gemini_flash_agent(api_key=os.getenv("GOOGLE_AI_API_KEY"))

# Method 3: Auto-detection (if GOOGLE_AI_API_KEY is set)
gemini_vision = create_gemini_vision_agent()  # Will find the key automatically
```

### **Configuration Object**
```python
config = {
    "api_key": "your-key-here",
    "temperature": 0.7,
    "max_tokens": 2048,
    "safe_mode": True
}

gemini_agent = create_gemini_pro_agent(**config)
```

## 🔗 **WHERE TO GET API KEYS**

| Provider | URL | Notes |
|----------|-----|--------|
| **🌟 Google AI** | https://aistudio.google.com/app/apikey | **START HERE!** |
| 🤖 OpenAI | https://platform.openai.com/api-keys | GPT models |
| 🧠 Anthropic | https://console.anthropic.com/ | Claude models |
| 💬 Cohere | https://dashboard.cohere.ai/api-keys | Command models |
| 🤗 Hugging Face | https://huggingface.co/settings/tokens | Open models |
| 🌪️ Mistral AI | https://console.mistral.ai/ | Mistral models |
| 🤝 Together AI | https://api.together.xyz/settings/api-keys | Various models |
| 🔄 Replicate | https://replicate.com/account/api-tokens | Model hosting |
| 🚀 Grok (xAI) | https://console.x.ai/ | Grok models |
| 🔍 Perplexity | https://www.perplexity.ai/settings/api | Search-enhanced |
| 🔬 AI21 Labs | https://studio.ai21.com/account/api-key | Jurassic models |
| ⚡ Groq | https://console.groq.com/keys | Super fast inference |
| 🎆 Fireworks | https://fireworks.ai/account/api-keys | Fast model hosting |

## 🛡️ **SECURITY BEST PRACTICES**

### **✅ DO:**
- Use `.env` files for persistent storage
- Add `.env` to your `.gitignore`
- Use environment variables in production
- Rotate API keys regularly
- Use separate keys for development/production

### **❌ DON'T:**
- Commit API keys to version control
- Share keys in public code
- Use production keys for testing
- Hardcode keys in source code

## 🚨 **TROUBLESHOOTING**

### **"API key not found" Error**
```python
# Check if your key is set
import os
print("Gemini key:", os.getenv("GOOGLE_AI_API_KEY"))

# If None, set it:
os.environ["GOOGLE_AI_API_KEY"] = "your-key-here"
```

### **"Invalid API key" Error**
- Verify key at: https://aistudio.google.com/app/apikey
- Check for extra spaces or characters
- Ensure key has proper permissions

### **Connection Error**
```python
# Test connection
from multi_agent_system.connectors import create_gemini_pro_agent

agent = create_gemini_pro_agent(api_key="your-key")
is_working = await agent.connector.validate_connection()
print(f"Connection working: {is_working}")
```

## 🎮 **READY TO GO COMMANDS**

### **1. Quick Gemini Test**
```bash
export GOOGLE_AI_API_KEY="your-key"
python -c "
import asyncio
from multi_agent_system.connectors import create_gemini_pro_agent

async def test():
    agent = create_gemini_pro_agent()
    result = await agent.connector.generate_response('Hello, Gemini!')
    print(f'Gemini says: {result}')

asyncio.run(test())
"
```

### **2. Full Gemini Showcase**
```bash
export GOOGLE_AI_API_KEY="your-key"
python examples/gemini_showcase.py
```

### **3. Mega AI Army (All Providers)**
```bash
# Set all your keys in .env file first
python examples/mega_ai_army.py
```

## 🌟 **PRO TIPS**

1. **Start with Gemini**: Only need `GOOGLE_AI_API_KEY` to get started
2. **Use .env files**: More convenient for development
3. **Test connection**: Always validate keys before running big tasks
4. **Monitor usage**: Check your API usage on provider dashboards
5. **Set limits**: Use rate limiting to avoid exceeding quotas

**You're now ready to unleash the full power of Gemini!** 🚀🌟 