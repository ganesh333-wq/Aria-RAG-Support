"""
──────────────────────────────────────────────────────────────────────────────
 Pydantic Schemas — Request / Response models for the Aria API

 Used by FastAPI for automatic validation, serialization, and OpenAPI docs.
──────────────────────────────────────────────────────────────────────────────
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Inbound ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Customer message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")

    model_config = {"json_schema_extra": {"example": {"message": "What is your return policy?", "session_id": "abc-123"}}}


# ── Sub-models ────────────────────────────────────────────────────────────────

class IntentResult(BaseModel):
    intent: str                         # GREETING | FAREWELL | FAQ_QUERY | FRUSTRATION | ESCALATION | NO_MATCH
    confidence: float
    sentiment: str                      # positive | negative | neutral
    sentiment_score: float              # [-1.0, 1.0]
    meta: Dict[str, Any] = {}


class RetrievedChunk(BaseModel):
    id: str
    score: float                        # cosine similarity [0, 1]
    category: str
    intent: str
    text: str
    escalate: bool = False
    escalation_reason: Optional[str] = None
    tags: List[str] = []


class EscalationResult(BaseModel):
    should_escalate: bool
    reason: str = ""
    urgency: str = "none"               # critical | high | medium | none
    trigger: str = ""                   # doc_flag | intent | frustration_count | keyword


class GenerationResult(BaseModel):
    source: str                         # groq | deterministic | template
    model: str
    tokens_used: int = 0
    rewritten_query: Optional[str] = None


class FollowUpResult(BaseModel):
    is_follow_up: bool
    previous_category: Optional[str] = None
    previous_query: Optional[str] = None
    signals: Dict[str, bool] = {}


class SessionInfo(BaseModel):
    session_id: str
    turn_count: int
    is_escalated: bool
    last_category: Optional[str]
    last_intent: Optional[str] = None
    frustration_count: int
    avg_sentiment: float
    sentiment_trend: List[float]


class PipelineTrace(BaseModel):
    timings: Dict[str, Any]
    initial_intent: Optional[Dict] = None
    follow_up: Optional[Dict] = None
    augmented_query: Optional[str] = None
    escalation: Optional[Dict] = None
    retrieval_count: int = 0
    confidence_filtered: bool = False


# ── Outbound ──────────────────────────────────────────────────────────────────

class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: IntentResult
    retrieval: Dict[str, Any]           # results list + query analysis
    generation: GenerationResult
    escalation: EscalationResult
    session: SessionInfo
    pipeline_trace: PipelineTrace

    model_config = {"json_schema_extra": {"example": {
        "response": "You can return items within 30 days of delivery.",
        "session_id": "abc-123",
        "intent": {"intent": "FAQ_QUERY", "confidence": 0.9, "sentiment": "neutral", "sentiment_score": 0.0, "meta": {}},
    }}}


class HealthResponse(BaseModel):
    status: str
    groq_available: bool
    vectorstore_ready: bool
    documents_indexed: int
    active_sessions: int


class StatsResponse(BaseModel):
    total_queries: int
    avg_pipeline_time_ms: float
    intent_distribution: Dict[str, int]
    escalation_count: int
    query_rewrites: int
    vectorstore: Dict[str, Any]
    llm: Dict[str, Any]
    active_sessions: int
