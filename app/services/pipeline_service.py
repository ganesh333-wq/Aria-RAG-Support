"""
──────────────────────────────────────────────────────────────────────────────
 Pipeline Service — Main Orchestrator for RAG pipeline

 Connects all components:
   Customer Query
        ↓
   Preprocessing (Contractions, Synonyms, Tokenize)
        ↓
   Intent Classification (Regex, Sentiment)
        ↓
   Memory Lookup & Follow-Up Detection
        ↓
   Query Rewriting (if follow-up)
        ↓
   ChromaDB Semantic Retrieval & Confidence Filtering
        ↓
   Escalation Detection
        ↓
   LLM Response Generation (Groq)
        ↓
   Update Memory & Return Trace
──────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from app.memory.session_memory import SessionMemoryManager
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    EscalationResult,
    FollowUpResult,
    GenerationResult,
    IntentResult,
    PipelineTrace,
    SessionInfo,
)
from app.retrievers.chroma_retriever import ChromaRetriever
from app.services.escalation_service import EscalationService
from app.services.intent_service import IntentService
from app.services.llm_service import LLMService
from app.services import preprocessor


class PipelineService:
    """
    Main orchestrator for the Aria RAG pipeline.
    """

    def __init__(
        self,
        intent_service: IntentService,
        escalation_service: EscalationService,
        llm_service: LLMService,
        retriever: ChromaRetriever,
        memory_manager: SessionMemoryManager,
    ):
        self.intent_service = intent_service
        self.escalation_service = escalation_service
        self.llm_service = llm_service
        self.retriever = retriever
        self.memory_manager = memory_manager

    async def process_chat(self, request: ChatRequest) -> ChatResponse:
        """
        Process a single chat request through the full pipeline.
        """
        start_time = time.time()
        trace_timings = {}
        session_id = request.session_id

        # ─── 1. Preprocessing ──────────────────────────────────────────────────
        t0 = time.time()
        processed_query = preprocessor.process(request.message)
        trace_timings["preprocessing_ms"] = round((time.time() - t0) * 1000, 2)

        # ─── 2. Initial Intent Classification ──────────────────────────────────
        t0 = time.time()
        intent_dict = self.intent_service.classify(request.message, processed_query)
        trace_timings["intent_classification_ms"] = round((time.time() - t0) * 1000, 2)
        initial_intent = intent_dict.copy()

        # ─── 3. Memory & Follow-Up Detection ───────────────────────────────────
        t0 = time.time()
        follow_up_dict = self.memory_manager.detect_follow_up(session_id, processed_query["raw_tokens"])
        conversation_context = self.memory_manager.get_conversation_context(session_id)
        
        augmented_query = request.message

        if follow_up_dict["is_follow_up"] and conversation_context:
            t_rw = time.time()
            # Use context-aware query rewriting with previous intent and category
            previous_intent = follow_up_dict.get("previous_intent")
            previous_category = follow_up_dict.get("previous_category")
            augmented_query = await self.llm_service.rewrite_query(
                request.message, 
                conversation_context,
                previous_intent=previous_intent,
                previous_category=previous_category,
            )
            trace_timings["query_rewrite_ms"] = round((time.time() - t_rw) * 1000, 2)
            trace_timings["rewrite_context"] = {
                "previous_intent": previous_intent,
                "previous_category": previous_category,
            }

        trace_timings["follow_up_detection_ms"] = round((time.time() - t0) * 1000, 2)

        # ─── 4. Retrieval & Filtering ──────────────────────────────────────────
        t0 = time.time()
        retrieval_results = []
        retrieval_analysis = {}
        filtered_count = 0

        if intent_dict["intent"] in ("FAQ_QUERY", "FRUSTRATION", "ESCALATION", "NO_MATCH"):
            # Pass previous category to retriever for category-aware boosting
            previous_category = follow_up_dict.get("previous_category") if follow_up_dict["is_follow_up"] else None
            previous_intent = follow_up_dict.get("previous_intent") if follow_up_dict["is_follow_up"] else None
            retrieval = self.retriever.retrieve(
                augmented_query,
                previous_category=previous_category,
                previous_intent=previous_intent,
            )
            retrieval_results = retrieval["results"]
            retrieval_analysis = retrieval["query_analysis"]
            filtered_count = retrieval["filtered_count"]

        trace_timings["retrieval_ms"] = round((time.time() - t0) * 1000, 2)

        # ─── 5. Post-Retrieval Intent Refinement ───────────────────────────────
        t0 = time.time()
        intent_dict = self.intent_service.refine_with_retrieval(intent_dict, retrieval_results)
        trace_timings["intent_refinement_ms"] = round((time.time() - t0) * 1000, 2)

        # ─── 6. Escalation Check ───────────────────────────────────────────────
        t0 = time.time()
        session_info_dict = self.memory_manager.get_session_info(session_id)
        escalation_dict = self.escalation_service.check(
            query=request.message,
            intent=intent_dict,
            retrieval_results=retrieval_results,
            session_frustration_count=session_info_dict.get("frustration_count", 0),
        )
        
        if escalation_dict["should_escalate"]:
            intent_dict["intent"] = "ESCALATION"

        trace_timings["escalation_check_ms"] = round((time.time() - t0) * 1000, 2)

        # ─── 7. LLM Response Generation ────────────────────────────────────────
        t0 = time.time()
        gen_result_dict = await self.llm_service.generate_response(
            query=request.message,
            retrieved_docs=retrieval_results,
            intent=intent_dict,
            conversation_history=conversation_context,
        )
        trace_timings["generation_ms"] = round((time.time() - t0) * 1000, 2)

        # Apply specific escalation message formatting if needed
        if escalation_dict["should_escalate"]:
            # Override response with specific urgency message
            gen_result_dict["response"] = self.escalation_service.get_escalation_response(
                urgency=escalation_dict["urgency"],
                reason=escalation_dict["reason"]
            )
            gen_result_dict["source"] = "deterministic"

        # ─── 8. Update Memory ──────────────────────────────────────────────────
        self.memory_manager.add_turn(
            session_id=session_id,
            user_message=request.message,
            bot_response=gen_result_dict["response"],
            intent=intent_dict,
            retrieval_results=retrieval_results,
        )

        trace_timings["total_pipeline_ms"] = round((time.time() - start_time) * 1000, 2)

        # ─── 9. Evaluation Logging ─────────────────────────────────────────────
        self._log_evaluation(
            session_id=session_id,
            request=request,
            augmented_query=augmented_query,
            intent_dict=intent_dict,
            retrieval_results=retrieval_results,
            retrieval_analysis=retrieval_analysis,
            escalation_dict=escalation_dict,
            gen_result_dict=gen_result_dict,
            trace_timings=trace_timings
        )

        # ─── Assemble Final Response ───────────────────────────────────────────
        
        updated_session_info = self.memory_manager.get_session_info(session_id)

        return ChatResponse(
            response=gen_result_dict["response"],
            session_id=session_id,
            intent=IntentResult(**intent_dict),
            retrieval={
                "results": retrieval_results,
                "query_analysis": retrieval_analysis,
            },
            generation=GenerationResult(
                source=gen_result_dict["source"],
                model=gen_result_dict["model"],
                tokens_used=gen_result_dict["tokens_used"],
                rewritten_query=augmented_query if augmented_query != request.message else None,
            ),
            escalation=EscalationResult(**escalation_dict),
            session=SessionInfo(**updated_session_info),
            pipeline_trace=PipelineTrace(
                timings=trace_timings,
                initial_intent=initial_intent,
                follow_up=follow_up_dict,
                augmented_query=augmented_query,
                escalation=escalation_dict,
                retrieval_count=len(retrieval_results),
                confidence_filtered=filtered_count > 0,
            ),
        )

    def _filter_by_topic(self, results: list, target_category: str) -> list:
        """Filter retrieved results to favor the previous topic for follow-ups."""
        if not results or not target_category:
            return results
        
        filtered = [r for r in results if r.get("category") == target_category]
        return filtered if filtered else results

    def _log_evaluation(self, session_id: str, request: ChatRequest, augmented_query: str,
                        intent_dict: Dict, retrieval_results: list, retrieval_analysis: Dict,
                        escalation_dict: Dict, gen_result_dict: Dict, trace_timings: Dict) -> None:
        """
        Log detailed retrieval and execution metrics for offline evaluation and debugging.
        Enhanced to include: rewritten queries, category context, reranking scores, empathy triggers.
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "raw_query": request.message,
            "augmented_query": augmented_query if augmented_query != request.message else None,
            "query_rewritten": augmented_query != request.message,
            "detected_intent": intent_dict.get("intent"),
            "intent_confidence": intent_dict.get("confidence"),
            "sentiment": intent_dict.get("sentiment"),
            "sentiment_score": intent_dict.get("sentiment_score"),
            "retrieved_chunks": [
                {
                    "id": r["id"],
                    "score": r["score"],
                    "category": r["category"],
                    "intent": r.get("intent"),
                    "rerank_boost": r.get("rerank_boost"),
                    "category_boost": r.get("category_boost"),
                    "intent_boost": r.get("intent_boost"),
                    "semantic_validation": r.get("semantic_validation"),
                } for r in retrieval_results
            ],
            "rejected_chunk_ids": retrieval_analysis.get("rejected_chunk_ids", []),
            "retrieval_rejection_reason": retrieval_analysis.get("rejection_reason"),
            "fallback_trigger_reason": retrieval_analysis.get("fallback_trigger_reason"),
            "selected_chunk_id": retrieval_results[0]["id"] if retrieval_results else None,
            "top_confidence_score": retrieval_results[0]["score"] if retrieval_results else 0.0,
            "top_category": retrieval_results[0].get("category") if retrieval_results else None,
            "retrieval_latency_ms": trace_timings.get("retrieval_ms", 0.0),
            "query_rewrite_latency_ms": trace_timings.get("query_rewrite_ms"),
            "rewrite_context": trace_timings.get("rewrite_context"),
            "category_boost_applied": retrieval_analysis.get("category_boost_applied", False),
            "final_response": gen_result_dict.get("response"),
            "response_source": gen_result_dict.get("source"),
            "response_model": gen_result_dict.get("model"),
            "response_validation": gen_result_dict.get("response_validation"),
            "empathy_trigger_reason": gen_result_dict.get("empathy_trigger_reason"),
            "empathy_injected": "I'm sorry" in gen_result_dict.get("response", "") or \
                               "I understand" in gen_result_dict.get("response", "") or \
                               "apologize" in gen_result_dict.get("response", ""),
            "escalation_triggered": escalation_dict.get("should_escalate", False),
            "escalation_reason": escalation_dict.get("reason", ""),
            "escalation_trigger": escalation_dict.get("trigger", ""),
            "escalation_urgency": escalation_dict.get("urgency", ""),
            "timings": {
                "preprocessing_ms": trace_timings.get("preprocessing_ms"),
                "intent_classification_ms": trace_timings.get("intent_classification_ms"),
                "follow_up_detection_ms": trace_timings.get("follow_up_detection_ms"),
                "query_rewrite_ms": trace_timings.get("query_rewrite_ms"),
                "retrieval_ms": trace_timings.get("retrieval_ms"),
                "intent_refinement_ms": trace_timings.get("intent_refinement_ms"),
                "escalation_check_ms": trace_timings.get("escalation_check_ms"),
                "generation_ms": trace_timings.get("generation_ms"),
                "total_pipeline_ms": trace_timings.get("total_pipeline_ms"),
            }
        }

        os.makedirs("logs", exist_ok=True)
        try:
            with open("logs/retrieval_evals.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            print(f"[PipelineService] Failed to write evaluation log: {e}")
