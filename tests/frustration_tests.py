import pytest
from app.models.schemas import ChatRequest

@pytest.mark.asyncio
async def test_frustration_intent_classification(pipeline):
    """
    Test that emotional/frustrated language is correctly classified.
    """
    request = ChatRequest(message="This is terrible service, why is my order taking so long?", session_id="frust-001")
    response = await pipeline.process_chat(request)
    
    assert response.intent.intent == "FRUSTRATION"
    assert response.intent.sentiment == "negative"

@pytest.mark.asyncio
async def test_repeated_frustration_escalation(pipeline):
    """
    Test that expressing frustration twice in a row triggers automatic escalation.
    """
    session_id = "frust-002"
    
    # Turn 1
    req1 = ChatRequest(message="This is terrible service", session_id=session_id)
    resp1 = await pipeline.process_chat(req1)
    
    # Usually first frustration doesn't escalate if it's just general anger without severe keywords
    # But let's assume it gets classified as FRUSTRATION
    
    # Turn 2
    req2 = ChatRequest(message="I am still angry, you are completely useless", session_id=session_id)
    resp2 = await pipeline.process_chat(req2)
    
    assert resp2.escalation.should_escalate is True
    assert resp2.session.frustration_count >= 2
    assert resp2.intent.intent == "ESCALATION"

@pytest.mark.asyncio
async def test_severe_frustration_escalation(pipeline):
    """
    Test that extremely severe language escalates immediately.
    """
    request = ChatRequest(message="This is absolutely unacceptable, worst experience ever", session_id="frust-003")
    response = await pipeline.process_chat(request)
    
    assert response.escalation.should_escalate is True
    assert response.escalation.trigger == "severe_frustration_keyword"
