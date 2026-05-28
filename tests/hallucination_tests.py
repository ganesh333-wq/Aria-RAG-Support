import pytest
from app.models.schemas import ChatRequest

@pytest.mark.asyncio
async def test_hallucination_prevention(pipeline):
    """
    Test that queries unrelated to the knowledge base return a NO_MATCH intent,
    and prevent the LLM from hallucinating an answer.
    """
    request = ChatRequest(message="Do you sell car insurance?", session_id="hal-001")
    response = await pipeline.process_chat(request)
    
    # The retriever should filter out all documents because of low confidence
    assert len(response.retrieval["results"]) == 0
    
    # The pipeline should assign NO_MATCH intent
    assert response.intent.intent == "NO_MATCH"
    
    # The final response should be the fallback template
    assert "I could not find relevant information" in response.response
    assert response.generation.source == "template"

@pytest.mark.asyncio
@pytest.mark.parametrize("message", [
    "Do you sell insurance?",
    "Can I invest in your company?",
    "Can I buy Bitcoin?",
    "What is the weather today?",
    "Teach me machine learning",
])
async def test_out_of_domain_queries_return_no_match(pipeline, message):
    """
    Unsupported domains should bypass weak nearest-neighbor retrieval and fail safe.
    """
    request = ChatRequest(message=message, session_id=f"hal-{message[:8]}")
    response = await pipeline.process_chat(request)

    assert response.intent.intent == "NO_MATCH"
    assert response.retrieval["results"] == []
    assert response.retrieval["query_analysis"]["fallback_trigger_reason"].startswith("out_of_domain")
    assert "I could not find relevant information" in response.response

@pytest.mark.asyncio
async def test_competitor_rejection(pipeline):
    """
    Test that questions about competitors are rejected gracefully.
    """
    request = ChatRequest(message="How does Amazon's return policy compare to yours?", session_id="hal-002")
    response = await pipeline.process_chat(request)
    
    # The LLM Service should detect competitor keywords and return a deterministic block
    assert response.generation.source == "deterministic"
    assert "ShopEase" in response.response
