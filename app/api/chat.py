"""
──────────────────────────────────────────────────────────────────────────────
 API Router — FastAPI endpoints for Aria Chatbot
──────────────────────────────────────────────────────────────────────────────
"""

import uuid
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from pathlib import Path

from app.models.schemas import ChatRequest, ChatResponse, HealthResponse, StatsResponse
from app.services.pipeline_service import PipelineService
from app.vectorstore.chroma_store import ChromaStore
from app.services.llm_service import LLMService
from app.memory.session_memory import SessionMemoryManager

router = APIRouter()


def get_pipeline(request: Request) -> PipelineService:
    pipeline = request.app.state.pipeline
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline

def get_chroma_store(request: Request) -> ChromaStore:
    return request.app.state.chroma_store

def get_llm_service(request: Request) -> LLMService:
    return request.app.state.llm_service

def get_memory_manager(request: Request) -> SessionMemoryManager:
    return request.app.state.memory_manager


@router.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    pipeline: PipelineService = Depends(get_pipeline)
):
    """
    Main Chat Endpoint
    Processes user query through the RAG pipeline.
    """
    # Auto-generate session ID if not provided
    if not request.session_id:
        request.session_id = str(uuid.uuid4())

    response = await pipeline.process_chat(request)
    return response


@router.get("/api/health", response_model=HealthResponse)
async def health_check(
    chroma: ChromaStore = Depends(get_chroma_store),
    llm: LLMService = Depends(get_llm_service),
    memory: SessionMemoryManager = Depends(get_memory_manager)
):
    """
    Health check endpoint to monitor component status.
    """
    stats = chroma.get_stats()
    return HealthResponse(
        status="healthy",
        groq_available=llm.groq_available,
        vectorstore_ready=stats.get("is_indexed", False),
        documents_indexed=stats.get("total_docs", 0),
        active_sessions=memory.get_active_session_count(),
    )


@router.get("/api/stats", response_model=StatsResponse)
async def get_stats(
    chroma: ChromaStore = Depends(get_chroma_store),
    llm: LLMService = Depends(get_llm_service),
    memory: SessionMemoryManager = Depends(get_memory_manager)
):
    """
    Get pipeline statistics.
    """
    llm_stats = llm.get_stats()
    
    # Normally we'd track these in pipeline_service, but we'll mock them 
    # for simplicity unless specifically asked to track full global stats there
    return StatsResponse(
        total_queries=llm_stats.get("total_requests", 0),
        avg_pipeline_time_ms=0.0,
        intent_distribution={},
        escalation_count=0,
        query_rewrites=0,
        vectorstore=chroma.get_stats(),
        llm=llm_stats,
        active_sessions=memory.get_active_session_count(),
    )


@router.get("/api/debug/{session_id}")
async def debug_session(
    session_id: str,
    memory: SessionMemoryManager = Depends(get_memory_manager)
):
    """Get session memory info for debugging."""
    return memory.get_session_info(session_id)


@router.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the frontend chat UI."""
    html_path = Path("frontend/index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), status_code=200)
    return HTMLResponse(
        content="<h1>Aria Chat UI</h1><p>frontend/index.html not found</p>",
        status_code=404,
    )
