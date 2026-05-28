"""
──────────────────────────────────────────────────────────────────────────────
 ShopEase Aria — FastAPI Server Entry Point

 Uses Lifespan events to initialize the VectorStore and LLM Services
 before accepting requests.
──────────────────────────────────────────────────────────────────────────────
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Load environment variables
load_dotenv()

from app.api.chat import router as chat_router
from app.memory.session_memory import SessionMemoryManager
from app.retrievers.chroma_retriever import ChromaRetriever
from app.services.escalation_service import EscalationService
from app.services.intent_service import IntentService
from app.services.llm_service import LLMService
from app.services.pipeline_service import PipelineService
from app.vectorstore.chroma_store import ChromaStore

# Globals for app state
KB_PATH = os.getenv("KB_PATH", "knowledge/knowledge_base.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize all RAG pipeline components on server startup.
    """
    print("=" * 60)
    print("  ShopEase Aria — Production RAG Support Chatbot")
    print("=" * 60)

    # 1. Initialize core services
    chroma_store = ChromaStore()
    llm_service = LLMService()
    intent_service = IntentService()
    escalation_service = EscalationService()
    memory_manager = SessionMemoryManager()

    # 2. Load KB and build ChromaDB index
    chroma_stats = chroma_store.initialize(KB_PATH)

    # 3. Initialize Groq LLM service
    llm_stats = await llm_service.initialize()

    # 4. Create retriever and pipeline
    retriever = ChromaRetriever(chroma_store=chroma_store)
    pipeline = PipelineService(
        intent_service=intent_service,
        escalation_service=escalation_service,
        llm_service=llm_service,
        retriever=retriever,
        memory_manager=memory_manager,
    )

    # Attach to app state for dependency injection
    app.state.chroma_store = chroma_store
    app.state.llm_service = llm_service
    app.state.memory_manager = memory_manager
    app.state.pipeline = pipeline

    print(f"\n  Docs Indexed: {chroma_stats['total_docs']}")
    print(f"  LLM Mode:     {llm_stats['active_mode']}")
    print( "\n  Server ready at http://localhost:8000")
    print("=" * 60)

    yield

    # Cleanup on shutdown
    memory_manager.cleanup_all()
    print("\n[Server] Shut down cleanly.")


# Create FastAPI App
app = FastAPI(
    title="ShopEase Aria",
    description="Production RAG Customer Support Chatbot",
    version="2.0.0",
    lifespan=lifespan,
)

# Register API routes
app.include_router(chat_router)

# Mount static files (frontend)
os.makedirs("frontend", exist_ok=True)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
