import pytest

def test_retrieval_accuracy(retriever):
    """
    Test that specific user queries retrieve the correct FAQ chunks.
    Validates semantic matching against aliases, typos, and standard queries.
    """
    test_cases = [
        # Standard queries
        ("What is your return policy?", "RETURN_POLICY"),
        ("How long does delivery take?", "DELIVERY_TIMEFRAME"),
        # Semantic aliases
        ("Can I send my shoes back?", "RETURN_POLICY"),
        ("Where is my money?", "REFUND_STATUS"),
        ("Is shipping free?", "SHIPPING_COST"),
        # Typos
        ("trak my pakage", "TRACK_ORDER"),
        ("wrong item recieved", "WRONG_ITEM"),
        ("retun policcy", "RETURN_POLICY"),
        # Edge cases
        ("I threw away the box", "NO_BOX")
    ]

    for query, expected_intent in test_cases:
        result = retriever.retrieve(query)
        chunks = result["results"]
        
        # Ensure we got at least one result
        assert len(chunks) > 0, f"No chunks retrieved for query: '{query}'"
        
        # Ensure the top result matches the expected intent
        top_chunk = chunks[0]
        assert top_chunk["intent"] == expected_intent, (
            f"Query '{query}' retrieved {top_chunk['intent']} instead of {expected_intent}"
        )
        
        # Verify confidence threshold is working
        assert top_chunk["score"] >= 0.35, "Score is below confidence threshold"

def test_confidence_filtering(retriever):
    """
    Test that irrelevant queries get filtered out by the confidence threshold.
    """
    # A completely unrelated query shouldn't return results
    result = retriever.retrieve("Do you sell car insurance?")
    
    # It might return raw results from ChromaDB, but the retriever should filter them
    assert len(result["results"]) == 0, "Unrelated query passed the confidence filter"
    assert result["filtered_count"] > 0, "No documents were filtered out"
