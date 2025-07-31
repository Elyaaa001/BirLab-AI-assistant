#!/usr/bin/env python3
"""
🌟 BirLab AI - FastAPI Backend Server
Ultimate API for 100+ AI Models

This server exposes the BirLab AI Ultra Coordinator with automatic
registration of all available AI models.
"""

import os
import sys
import logging
import asyncio
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import json
from datetime import datetime

# Import BirLab AI components
from multi_agent_system.core.birlab_coordinator import create_birlab_ultra_coordinator, BirLabUltraCoordinator
from multi_agent_system.core.task import Task, TaskPriority
from multi_agent_system.connectors.expanded_models import BIRLAB_AI_MODELS, get_all_birlab_ai_models

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global coordinator instance
coordinator: Optional[BirLabUltraCoordinator] = None
available_agents: Dict[str, Any] = {}
registration_report: Dict[str, List[str]] = {}


# Pydantic models for API
class ChatMessage(BaseModel):
    message: str
    agent_id: Optional[str] = None
    task_type: Optional[str] = None  # For intelligent agent selection


class ChatResponse(BaseModel):
    response: str
    agent_used: str
    timestamp: str
    model_info: Optional[Dict[str, Any]] = None


class SplitViewRequest(BaseModel):
    message: str
    model_ids: List[str]  # List of specific model IDs to compare
    max_models: Optional[int] = 4  # Maximum number of models to use
    task_type: Optional[str] = None  # For intelligent model selection


class ModelComparisonResult(BaseModel):
    agent_id: str
    response: str
    model_info: Dict[str, Any]
    response_time: float
    response_length_words: int = 0
    response_length_chars: int = 0
    quality_score: float = 0.0
    timestamp: str
    error: Optional[str] = None


class SplitViewResponse(BaseModel):
    results: List[ModelComparisonResult]
    original_message: str
    timestamp: str
    comparison_stats: Dict[str, Any]


class TaskRequest(BaseModel):
    description: str
    priority: str = "medium"
    agent_id: Optional[str] = None
    task_type: Optional[str] = None


class TaskResponse(BaseModel):
    task_id: str
    result: str
    agent_used: str
    execution_time: float
    timestamp: str


class MultimodalRequest(BaseModel):
    prompt: str
    agent_id: Optional[str] = None


class SystemStatus(BaseModel):
    status: str
    total_agents: int
    active_providers: int
    timestamp: str
    provider_stats: Dict[str, int]
    capabilities_available: List[str]


class AgentInfo(BaseModel):
    agent_id: str
    name: str
    provider: str
    model: str
    capabilities: List[str]
    context_length: int
    specialty: Optional[str] = None


# Lifespan manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup_event()
    yield
    # Shutdown (if needed)
    pass


# FastAPI app
app = FastAPI(
    title="🌟 BirLab AI API",
    description="Production-ready API for 100+ AI models with intelligent coordination",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def startup_event():
    """Initialize the BirLab AI Ultra Coordinator with 100+ models"""
    global coordinator, available_agents, registration_report
    
    logger.info("🚀 Starting BirLab AI Backend...")
    
    try:
        # Create ultra coordinator
        coordinator = create_birlab_ultra_coordinator()
        
        # Get registration report
        registration_report = {
            provider: models for provider, models in coordinator.registered_models.items()
        }
        
        # Get all available agents
        available_agents = {}
        for model_id, model_data in coordinator.registered_models.items():
            agent_id = model_data["agent_id"]
            model_info = model_data["model_info"]
            
            available_agents[agent_id] = {
                "model_id": model_id,
                "name": model_info["name"],
                "provider": model_info["provider"],
                "model": model_info["model"],
                "capabilities": model_info.get("capabilities", []),
                "context_length": model_info.get("context_length", 0),
                "specialty": model_info.get("specialty")
            }
        
        logger.info(f"✅ BirLab AI initialized with {len(available_agents)} agents!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize BirLab AI: {e}")
        # Continue with empty coordinator for basic functionality
        coordinator = None
        available_agents = {}


@app.get("/")
async def root():
    """API root endpoint with system information"""
    return {
        "message": "🌟 Welcome to BirLab AI API!",
        "version": "2.0.0",
        "docs": "/docs",
        "total_agents": len(available_agents),
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "agents_available": len(available_agents)
    }


@app.get("/agents")
async def list_agents() -> List[AgentInfo]:
    """List all available AI agents"""
    if not available_agents:
        raise HTTPException(status_code=503, detail="No agents available")
    
    agents = []
    for agent_id, info in available_agents.items():
        agents.append(AgentInfo(
            agent_id=agent_id,
            name=info["name"],
            provider=info["provider"],
            model=info["model"],
            capabilities=info["capabilities"],
            context_length=info["context_length"],
            specialty=info.get("specialty")
        ))
    
    return agents


@app.get("/system/status")
async def system_status() -> SystemStatus:
    """Get comprehensive system status"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="BirLab AI coordinator not available")
    
    stats = coordinator.get_agent_army_stats()
    
    return SystemStatus(
        status="operational",
        total_agents=stats["total_agents"],
        active_providers=len(stats["providers"]),
        timestamp=datetime.now().isoformat(),
        provider_stats=stats["providers"],
        capabilities_available=list(stats["capabilities"].keys())
    )


@app.post("/chat")
async def chat(request: ChatMessage) -> ChatResponse:
    """Chat with an AI agent (with intelligent selection)"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="BirLab AI coordinator not available")
    
    try:
        # Intelligent agent selection if not specified
        agent_id = request.agent_id
        if not agent_id and request.task_type:
            agent_id = coordinator.get_best_agent_for_task(request.task_type)
        
        if not agent_id:
            # Fallback to any available agent
            agent_id = next(iter(available_agents.keys())) if available_agents else None
        
        if not agent_id:
            raise HTTPException(status_code=404, detail="No suitable agent found")
        
        # Create task
        task = Task(
            description=f"Chat: {request.message}",
            priority=TaskPriority.MEDIUM,
            task_type=request.task_type or "chat"
        )
        
        # Execute task
        result = await coordinator.execute_task(task, preferred_agent=agent_id)
        
        # Get model info
        model_info = None
        if agent_id in available_agents:
            model_info = available_agents[agent_id]
        
        return ChatResponse(
            response=result.result,
            agent_used=agent_id,
            timestamp=datetime.now().isoformat(),
            model_info=model_info
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


@app.post("/split-view")
async def split_view_comparison(request: SplitViewRequest) -> SplitViewResponse:
    """🔀 Split View: Compare responses from multiple AI models simultaneously"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="BirLab AI coordinator not available")
    
    try:
        import time
        start_time = time.time()
        
        # Determine which models to use
        target_models = []
        
        if request.model_ids:
            # Use specified models
            target_models = [mid for mid in request.model_ids if mid in available_agents]
        else:
            # Intelligent model selection based on task type
            if request.task_type:
                # Get diverse models for comparison
                all_agents = list(available_agents.keys())
                # Try to get different providers for better comparison
                providers_used = set()
                for agent_id in all_agents:
                    provider = agent_id.split('_')[0] if '_' in agent_id else 'unknown'
                    if provider not in providers_used and len(target_models) < request.max_models:
                        target_models.append(agent_id)
                        providers_used.add(provider)
            else:
                # Default: select first N available models
                target_models = list(available_agents.keys())[:request.max_models]
        
        if not target_models:
            raise HTTPException(status_code=404, detail="No suitable models found for comparison")
        
        # Create tasks for each model
        results = []
        tasks = []
        
        def simple_quality_score(text: str) -> float:
            # Heuristic: longer, more complete, and clear responses get higher score
            if not text or not text.strip():
                return 0.0
            length = len(text.split())
            if length < 5:
                return 0.1
            if length < 20:
                return 0.3
            if length < 50:
                return 0.6
            if length < 100:
                return 0.8
            return 1.0
        
        async def query_model(agent_id: str) -> ModelComparisonResult:
            """Query a single model and return timed result"""
            model_start = time.time()
            try:
                task = Task(
                    description=f"Split View Chat: {request.message}",
                    priority=TaskPriority.HIGH,  # High priority for split view
                    task_type=request.task_type or "split_view"
                )
                
                result = await coordinator.execute_task(task, preferred_agent=agent_id)
                model_end = time.time()
                
                model_info = available_agents.get(agent_id, {})
                response = result.result or ""
                response_length_words = len(response.split())
                response_length_chars = len(response)
                quality_score = simple_quality_score(response)
                
                return ModelComparisonResult(
                    agent_id=agent_id,
                    response=response,
                    model_info=model_info,
                    response_time=round(model_end - model_start, 2),
                    response_length_words=response_length_words,
                    response_length_chars=response_length_chars,
                    quality_score=quality_score,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"Error querying {agent_id}: {e}")
                return ModelComparisonResult(
                    agent_id=agent_id,
                    response="",
                    model_info=available_agents.get(agent_id, {}),
                    response_time=round(time.time() - model_start, 2),
                    response_length_words=0,
                    response_length_chars=0,
                    quality_score=0.0,
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
        
        # Execute all model queries concurrently
        tasks = [query_model(agent_id) for agent_id in target_models]
        results = await asyncio.gather(*tasks)
        
        # Calculate comparison stats
        successful_results = [r for r in results if not r.error]
        total_time = time.time() - start_time
        
        comparison_stats = {
            "total_models": len(target_models),
            "successful_responses": len(successful_results),
            "failed_responses": len(results) - len(successful_results),
            "total_comparison_time": round(total_time, 2),
            "average_response_time": round(
                sum(r.response_time for r in successful_results) / len(successful_results), 2
            ) if successful_results else 0,
            "fastest_model": min(successful_results, key=lambda x: x.response_time).agent_id if successful_results else None,
            "slowest_model": max(successful_results, key=lambda x: x.response_time).agent_id if successful_results else None,
            "longest_response": max(successful_results, key=lambda x: x.response_length_words).agent_id if successful_results else None,
            "shortest_response": min(successful_results, key=lambda x: x.response_length_words).agent_id if successful_results else None,
            "highest_quality": max(successful_results, key=lambda x: x.quality_score).agent_id if successful_results else None,
        }
        
        return SplitViewResponse(
            results=results,
            original_message=request.message,
            timestamp=datetime.now().isoformat(),
            comparison_stats=comparison_stats
        )
        
    except Exception as e:
        logger.error(f"Split View error: {e}")
        raise HTTPException(status_code=500, detail=f"Split View comparison failed: {str(e)}")


@app.post("/task")
async def execute_task(request: TaskRequest) -> TaskResponse:
    """Execute a complex task"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="BirLab AI coordinator not available")
    
    try:
        # Parse priority
        priority_map = {
            "low": TaskPriority.LOW,
            "medium": TaskPriority.MEDIUM,
            "high": TaskPriority.HIGH,
            "urgent": TaskPriority.URGENT
        }
        priority = priority_map.get(request.priority.lower(), TaskPriority.MEDIUM)
        
        # Intelligent agent selection
        agent_id = request.agent_id
        if not agent_id and request.task_type:
            agent_id = coordinator.get_best_agent_for_task(request.task_type)
        
        # Create and execute task
        task = Task(
            description=request.description,
            priority=priority
        )
        
        start_time = asyncio.get_event_loop().time()
        result = await coordinator.execute_task(task, preferred_agent=agent_id)
        execution_time = asyncio.get_event_loop().time() - start_time
        
        return TaskResponse(
            task_id=task.task_id,
            result=result.result,
            agent_used=result.agent_id or "unknown",
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Task execution error: {e}")
        raise HTTPException(status_code=500, detail=f"Task execution failed: {str(e)}")


@app.post("/multimodal")
async def multimodal_analysis(
    prompt: str,
    file: UploadFile = File(...),
    agent_id: Optional[str] = None
):
    """Analyze images with multimodal AI agents"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="BirLab AI coordinator not available")
    
    try:
        # Get multimodal agent
        if not agent_id:
            agent_id = coordinator.get_best_agent_for_task("multimodal")
        
        if not agent_id:
            raise HTTPException(status_code=404, detail="No multimodal agent available")
        
        # Read image data
        image_data = await file.read()
        
        # Create multimodal task
        task = Task(
            description=f"Analyze image: {prompt}",
            priority=TaskPriority.MEDIUM,
            metadata={"image_data": image_data, "filename": file.filename}
        )
        
        result = await coordinator.execute_task(task, preferred_agent=agent_id)
        
        return {
            "result": result.result,
            "agent_used": agent_id,
            "filename": file.filename,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Multimodal analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Multimodal analysis failed: {str(e)}")


@app.websocket("/ws")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    if not coordinator:
        await websocket.send_text(json.dumps({
            "error": "BirLab AI coordinator not available"
        }))
        await websocket.close()
        return
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Get agent
            agent_id = message_data.get("agent_id")
            task_type = message_data.get("task_type")
            
            if not agent_id and task_type:
                agent_id = coordinator.get_best_agent_for_task(task_type)
            
            if not agent_id:
                agent_id = next(iter(available_agents.keys())) if available_agents else None
            
            if not agent_id:
                await websocket.send_text(json.dumps({
                    "error": "No suitable agent found"
                }))
                continue
            
            # Process message
            task = Task(
                description=f"Chat: {message_data['message']}",
                priority=TaskPriority.MEDIUM
            )
            
            result = await coordinator.execute_task(task, preferred_agent=agent_id)
            
            # Send response
            response = {
                "response": result.result,
                "agent_used": agent_id,
                "timestamp": datetime.now().isoformat()
            }
            
            await websocket.send_text(json.dumps(response))
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_text(json.dumps({
            "error": f"WebSocket error: {str(e)}"
        }))


@app.get("/models/registry")
async def get_models_registry():
    """Get the complete BirLab AI models registry"""
    return {
        "total_models": len(BIRLAB_AI_MODELS),
        "models": BIRLAB_AI_MODELS,
        "registered_agents": len(available_agents),
        "providers": list(set(model["provider"] for model in BIRLAB_AI_MODELS.values()))
    }


@app.get("/agent/{agent_id}/info")
async def get_agent_info(agent_id: str):
    """Get detailed information about a specific agent"""
    if agent_id not in available_agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return available_agents[agent_id]


@app.get("/models/available")
async def get_available_models() -> Dict[str, Any]:
    """🔀 Get available models for Split View selector"""
    if not available_agents:
        raise HTTPException(status_code=503, detail="No models available")
    
    # Group models by provider for better organization
    models_by_provider = {}
    model_details = []
    
    for agent_id, agent_info in available_agents.items():
        provider = agent_id.split('_')[0] if '_' in agent_id else 'unknown'
        
        if provider not in models_by_provider:
            models_by_provider[provider] = []
        
        model_entry = {
            "id": agent_id,
            "name": agent_info.get("name", agent_id),
            "provider": provider,
            "capabilities": agent_info.get("capabilities", []),
            "context_length": agent_info.get("context_length", "Unknown"),
            "description": agent_info.get("description", ""),
        }
        
        models_by_provider[provider].append(model_entry)
        model_details.append(model_entry)
    
    return {
        "total_available": len(available_agents),
        "models_by_provider": models_by_provider,
        "all_models": model_details,
        "providers": list(models_by_provider.keys())
    }


@app.post("/test-agent-selection")
async def test_agent_selection(task_type: str):
    """Test the intelligent agent selection system"""
    if not coordinator:
        raise HTTPException(status_code=503, detail="Coordinator not available")
    
    best_agent = coordinator.get_best_agent_for_task(task_type)
    
    if not best_agent:
        return {"task_type": task_type, "best_agent": None, "message": "No suitable agent found"}
    
    return {
        "task_type": task_type,
        "best_agent": best_agent,
        "agent_info": available_agents.get(best_agent),
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    # Run the server
    logger.info("🌟 Starting BirLab AI FastAPI Server...")
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for production
        log_level="info"
    ) 