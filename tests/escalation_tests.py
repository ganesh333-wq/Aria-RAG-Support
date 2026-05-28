import pytest
from app.models.schemas import ChatRequest

@pytest.mark.asyncio
async def test_fraud_escalation(pipeline):
    """
    Test that fraud keywords trigger an immediate critical escalation.
    """
    request = ChatRequest(message="I think my account was hacked", session_id="esc-001")
    response = await pipeline.process_chat(request)
    
    assert response.escalation.should_escalate is True
    assert response.escalation.urgency == "critical"
    assert response.intent.intent == "ESCALATION"

@pytest.mark.asyncio
async def test_legal_threat_escalation(pipeline):
    """
    Test that legal threats trigger an immediate critical escalation.
    """
    request = ChatRequest(message="I am going to sue you for this", session_id="esc-002")
    response = await pipeline.process_chat(request)
    
    assert response.escalation.should_escalate is True
    assert response.escalation.urgency == "critical"
    assert response.escalation.trigger == "legal_threat"

@pytest.mark.asyncio
async def test_human_agent_escalation(pipeline):
    """
    Test that explicit requests for a human agent trigger high escalation.
    """
    request = ChatRequest(message="Let me speak to a real person", session_id="esc-003")
    response = await pipeline.process_chat(request)
    
    assert response.escalation.should_escalate is True
    assert response.escalation.urgency == "high"
    assert response.intent.intent == "ESCALATION"

@pytest.mark.asyncio
async def test_escalation_does_not_persist_to_next_query(pipeline):
    """
    Escalation is message-scoped. A later unrelated FAQ should not stay locked
    in escalation mode just because the previous turn escalated.
    """
    session_id = "esc-transient"

    first = await pipeline.process_chat(ChatRequest(
        message="I want to talk to a real person",
        session_id=session_id,
    ))
    assert first.escalation.should_escalate is True
    assert first.intent.intent == "ESCALATION"

    second = await pipeline.process_chat(ChatRequest(
        message="What is your return policy?",
        session_id=session_id,
    ))
    assert second.escalation.should_escalate is False
    assert second.intent.intent == "FAQ_QUERY"
    assert second.retrieval["results"][0]["intent"] == "RETURN_POLICY"

@pytest.mark.asyncio
async def test_damaged_goods_escalation(pipeline):
    """
    Test that reporting damaged goods triggers escalation (via document flag).
    """
    request = ChatRequest(message="My item arrived completely broken", session_id="esc-004")
    response = await pipeline.process_chat(request)
    
    assert response.escalation.should_escalate is True
    # The escalation is caught by the retrieved document flag
    assert response.escalation.trigger == "doc_flag"
