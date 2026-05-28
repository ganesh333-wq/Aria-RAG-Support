import os
import pytest
import pytest_asyncio
from app.vectorstore.chroma_store import ChromaStore
from app.retrievers.chroma_retriever import ChromaRetriever
from app.services.intent_service import IntentService
from app.services.escalation_service import EscalationService
from app.services.llm_service import LLMService
from app.memory.session_memory import SessionMemoryManager
from app.services.pipeline_service import PipelineService

@pytest.fixture(scope="session")
def chroma_store():
    # Setup ChromaStore with the actual KB to test retrieval accuracy
    store = ChromaStore(persist_dir="./tests/test_chroma_db", collection_name="test_aria_faq")
    kb_path = os.getenv("KB_PATH", "knowledge/knowledge_base.json")
    if os.path.exists(kb_path):
        store.initialize(kb_path)
    return store

@pytest.fixture(scope="session")
def retriever(chroma_store):
    # Set a strict confidence threshold for testing
    return ChromaRetriever(chroma_store=chroma_store, min_confidence=0.35, top_k=3)

@pytest.fixture
def intent_service():
    return IntentService()

@pytest.fixture
def escalation_service():
    return EscalationService()

@pytest.fixture
def memory_manager():
    return SessionMemoryManager()

@pytest_asyncio.fixture
async def llm_service():
    service = LLMService()
    await service.initialize()
    return service

@pytest_asyncio.fixture
async def pipeline(chroma_store, retriever, intent_service, escalation_service, memory_manager, llm_service):
    return PipelineService(
        intent_service=intent_service,
        escalation_service=escalation_service,
        llm_service=llm_service,
        retriever=retriever,
        memory_manager=memory_manager
    )
