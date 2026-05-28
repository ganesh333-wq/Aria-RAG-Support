import pytest
from app.services import preprocessor

def test_followup_detection(memory_manager):
    """
    Test that the memory manager correctly detects follow-up questions.
    """
    session_id = "test_followup_session"
    
    # Simulate a prior turn
    memory_manager.add_turn(
        session_id=session_id,
        user_message="What is the return policy?",
        bot_response="You have 30 days.",
        intent={"intent": "FAQ_QUERY"},
        retrieval_results=[{"category": "Returns & Refunds"}]
    )

    # Test case 1: Pronoun reference
    query = "What about international?"
    tokens = preprocessor.tokenize(query)
    detection = memory_manager.detect_follow_up(session_id, tokens)
    
    assert detection["is_follow_up"] is True, "Failed to detect explicit follow-up phrase"
    assert detection["previous_category"] == "Returns & Refunds"

    # Test case 2: Short question
    query = "How long?"
    tokens = preprocessor.tokenize(query)
    detection = memory_manager.detect_follow_up(session_id, tokens)
    assert detection["is_follow_up"] is True, "Failed to detect short question follow-up"

@pytest.mark.asyncio
async def test_query_rewriting(llm_service, memory_manager):
    """
    Test that the LLM successfully rewrites an ambiguous follow-up query.
    Requires Groq to be available unless deterministic rewriting handles it.
    """
    session_id = "test_rewrite_session"
    memory_manager.add_turn(
        session_id=session_id,
        user_message="What is the return policy?",
        bot_response="You have 30 days.",
        intent={"intent": "FAQ_QUERY"},
        retrieval_results=[{"category": "Returns & Refunds"}]
    )

    context = memory_manager.get_conversation_context(session_id)
    raw_query = "What about international?"
    
    rewritten = await llm_service.rewrite_query(raw_query, context)
    
    # The rewritten query should be longer and contain context clues
    assert rewritten != raw_query
    assert "international" in rewritten.lower()
    assert "return" in rewritten.lower() or "policy" in rewritten.lower()
