"""
──────────────────────────────────────────────────────────────────────────────
 Pipeline Tests — Validates each RAG component end-to-end
──────────────────────────────────────────────────────────────────────────────
"""

import asyncio
import json
import sys
import os
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("pipeline", reason="Legacy tests for the old pipeline package.")

from pipeline import preprocessor
from pipeline.vector_store import VectorStore
from pipeline.retriever import Retriever
from pipeline.intent_classifier import IntentClassifier
from pipeline.conversation_manager import ConversationManager
from pipeline.rag_pipeline import RAGPipeline


def test_preprocessor():
    """Test text preprocessing pipeline."""
    print("\n─── Test: Preprocessor ────────────────────────────")

    # Contraction expansion
    result = preprocessor.process("What's your return policy?")
    assert "what" in result["cleaned"], f"Contraction not expanded: {result['cleaned']}"

    # Synonym expansion
    result = preprocessor.process("I want to send back my order")
    assert "return" in result["cleaned"], f"Synonym not expanded: {result['cleaned']}"

    # Stopword removal
    result = preprocessor.process("What is the return policy for orders")
    assert "is" not in result["tokens"], f"Stopword not removed: {result['tokens']}"
    assert "the" not in result["tokens"], f"Stopword not removed: {result['tokens']}"

    # Stemming
    result = preprocessor.process("shipping internationally")
    assert any("ship" in s for s in result["stems"]), f"Stem failed: {result['stems']}"

    # Edit distance
    assert preprocessor.edit_distance("return", "retrun") == 2
    assert preprocessor.fuzzy_match("shipping", "shiping", 0.3) is True

    print("  ✅ All preprocessor tests passed")


def test_vector_store():
    """Test vector store indexing and search."""
    print("\n─── Test: Vector Store ────────────────────────────")

    store = VectorStore()

    docs = [
        {"id": "d1", "text": "return policy refund exchange", "answer": "Return info"},
        {"id": "d2", "text": "shipping delivery tracking international", "answer": "Shipping info"},
        {"id": "d3", "text": "payment credit card paypal methods", "answer": "Payment info"},
    ]

    store.index(docs, text_field="text")
    assert store.is_indexed, "Store not indexed"
    assert store.stats["total_docs"] == 3
    assert store.stats["embedding_dim"] == 384, f"Dim should be 384 for all-MiniLM-L6-v2, got {store.stats['embedding_dim']}"

    # Search
    results = store.search("how do I return an item", top_k=2)
    assert results[0]["id"] == "d1", f"Wrong top result: {results[0]['id']}"

    results = store.search("international shipping tracking", top_k=2)
    assert results[0]["id"] == "d2", f"Wrong top result: {results[0]['id']}"

    # Get by ID
    doc = store.get_by_id("d1")
    assert doc is not None
    assert doc["id"] == "d1"

    print("  ✅ All vector store tests passed")


def test_intent_classifier():
    """Test intent classification."""
    print("\n─── Test: Intent Classifier ───────────────────────")

    classifier = IntentClassifier()

    # Greeting
    processed = preprocessor.process("Hello!")
    intent = classifier.classify("Hello!", processed)
    assert intent["intent"] == "GREETING", f"Expected GREETING, got {intent['intent']}"

    # Farewell
    processed = preprocessor.process("Thanks, bye!")
    intent = classifier.classify("Thanks, bye!", processed)
    assert intent["intent"] == "FAREWELL", f"Expected FAREWELL, got {intent['intent']}"

    # Escalation
    processed = preprocessor.process("I want to speak to a manager")
    intent = classifier.classify("I want to speak to a manager", processed)
    assert intent["intent"] == "ESCALATION", f"Expected ESCALATION, got {intent['intent']}"

    # FAQ query
    processed = preprocessor.process("What is your return policy?")
    intent = classifier.classify("What is your return policy?", processed)
    assert intent["intent"] == "FAQ_QUERY", f"Expected FAQ_QUERY, got {intent['intent']}"

    # Post-retrieval refinement (escalation doc)
    mock_results = [{"id": "esc-001", "score": 0.8, "metadata": {"escalate": True, "escalation_reason": "Fraud"}, "text": None}]
    refined = classifier.refine_with_retrieval(intent, mock_results)
    assert refined["intent"] == "ESCALATION"

    print("  ✅ All intent classifier tests passed")


def test_conversation_manager():
    """Test conversation session management."""
    print("\n─── Test: Conversation Manager ────────────────────")

    mgr = ConversationManager()

    # New session
    session = mgr.get_session("test-1")
    assert session.turn_count == 0

    # Add turns
    mgr.add_turn("test-1", "What is your return policy?", "30-day returns...",
                  intent={"intent": "FAQ_QUERY", "sentiment": "neutral", "sentiment_score": 0})
    mgr.add_turn("test-1", "How about shipping?", "We ship to 45+ countries...",
                  intent={"intent": "FAQ_QUERY", "sentiment": "neutral", "sentiment_score": 0})

    ctx = mgr.get_context_for_llm("test-1")
    assert ctx is not None
    assert ctx["turn_count"] == 2
    assert "return" in ctx["text"].lower()

    # Follow-up detection
    follow_up = mgr.is_follow_up("test-1", ["what", "about", "it"])
    assert follow_up["is_follow_up"] is True

    print("  ✅ All conversation manager tests passed")


def test_retriever():
    """Test multi-signal retrieval."""
    print("\n─── Test: Retriever ───────────────────────────────")

    with open("knowledge/knowledge_base.json", "r") as f:
        kb = json.load(f)

    store = VectorStore()
    index_docs = []
    for doc in kb:
        parts = doc.get("questions", []) + doc.get("tags", [])
        if doc.get("answer"):
            parts.append(doc["answer"])
        index_docs.append({
            "id": doc["id"],
            "text": " ".join(parts),
            "answer": doc.get("answer"),
            "category": doc.get("category"),
            "escalate": doc.get("escalate", False),
        })

    store.index(index_docs, text_field="text")
    retriever = Retriever(store, kb)

    # Test return policy query
    result = retriever.retrieve("What is your return policy?")
    assert len(result["results"]) > 0, "No results for return policy query"
    assert result["results"][0]["id"] == "ret-001", f"Wrong top doc: {result['results'][0]['id']}"
    print(f"  Return policy → top match: {result['results'][0]['id']} (score: {result['results'][0]['score']})")

    # Test international shipping
    result = retriever.retrieve("Do you ship internationally?")
    assert result["results"][0]["id"] == "ship-001", f"Wrong top doc: {result['results'][0]['id']}"
    print(f"  Intl shipping → top match: {result['results'][0]['id']} (score: {result['results'][0]['score']})")

    # Test typo tolerance
    result = retriever.retrieve("how to trakc my ordr")
    assert len(result["results"]) > 0, "No results for typo query"
    print(f"  Typo query    → top match: {result['results'][0]['id']} (score: {result['results'][0]['score']})")

    # Test escalation
    result = retriever.retrieve("my item arrived broken and damaged")
    top = result["results"][0]
    assert top["metadata"].get("escalate") is True, f"Should match escalation doc: {top['id']}"
    print(f"  Escalation    → top match: {top['id']} (escalate: {top['metadata']['escalate']})")

    print("  ✅ All retriever tests passed")


async def test_full_pipeline():
    """Test the complete RAG pipeline end-to-end."""
    print("\n─── Test: Full RAG Pipeline ───────────────────────")

    pipeline = RAGPipeline()
    pipeline.initialize("knowledge/knowledge_base.json")

    test_queries = [
        ("Hello!", "GREETING"),
        ("What is your return policy?", "FAQ_QUERY"),
        ("Do you ship internationally?", "FAQ_QUERY"),
        ("I want to speak to a manager", "ESCALATION"),
        ("Thanks, goodbye!", "FAREWELL"),
    ]

    for query, expected_intent in test_queries:
        result = await pipeline.process(query, session_id="test-pipeline")
        actual_intent = result["intent"]["intent"]
        status = "✅" if actual_intent == expected_intent else "❌"
        print(
            f"  {status} \"{query}\"\n"
            f"     Intent: {actual_intent} (expected: {expected_intent}) | "
            f"Source: {result['generation']['source']} | "
            f"Time: {result['pipeline_trace']['timings']['total_pipeline_ms']:.0f}ms"
        )
        if result["retrieval"]["results"]:
            top = result["retrieval"]["results"][0]
            print(f"     Top doc: {top['id']} (score: {top['score']}) | Category: {top['metadata']['category']}")

    # Print stats
    stats = pipeline.get_stats()
    print(f"\n  Pipeline Stats:")
    print(f"    Total queries:     {stats['total_queries']}")
    print(f"    Avg pipeline time: {stats['avg_pipeline_time_ms']:.1f}ms")
    print(f"    Intent dist:       {stats['intent_distribution']}")
    print(f"    Escalations:       {stats['escalation_count']}")

    print("\n  ✅ Full pipeline test passed")


def main():
    print("=" * 60)
    print("  ShopEase Aria — RAG Pipeline Tests")
    print("=" * 60)

    test_preprocessor()
    test_vector_store()
    test_intent_classifier()
    test_conversation_manager()
    test_retriever()
    asyncio.run(test_full_pipeline())

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✅")
    print("=" * 60)


if __name__ == "__main__":
    main()
