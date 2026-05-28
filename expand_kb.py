#!/usr/bin/env python3
"""
Knowledge Base Expansion Script

Adds semantic variants, typos, and emotional phrases to improve:
- International return/shipping handling
- Refund delay handling
- Damaged goods handling
- Order tracking
- Payment issues
"""

import json

def expand_knowledge_base():
    """Expand the knowledge base with additional variants."""
    
    with open("knowledge/knowledge_base.json", "r", encoding="utf-8") as f:
        kb = json.load(f)
    
    # Create mapping of intent to entry for easy updates
    kb_by_intent = {entry.get("intent"): entry for entry in kb}
    
    # ENHANCEMENT 1: International returns
    if "RETURN_POLICY" in kb_by_intent:
        entry = kb_by_intent["RETURN_POLICY"]
        additional_variants = [
            "international return",
            "return from overseas",
            "return to different country",
            "can I return from abroad",
            "return outside USA",
            "how to return international",
            "retrn policy international",
            "foreign country return",
            "can I send back from outside",
        ]
        for variant in additional_variants:
            if variant not in entry["semantic_aliases"] and variant not in entry["questions"]:
                entry["questions"].append(variant)
    
    # ENHANCEMENT 2: Refund delays with emotional phrases
    if "REFUND_STATUS" in kb_by_intent:
        entry = kb_by_intent["REFUND_STATUS"]
        emotional_variants = [
            "I'm frustrated, my refund still hasn't arrived",
            "It's been forever, where's my refund?",
            "My refund is taking way too long",
            "I'm upset about my missing refund",
            "This refund delay is ridiculous",
            "Why hasn't my refund been credited yet?",
            "refund still not received",
            "refund never came",
            "where did my refund go",
            "refund disappeared",
            "refund stuck",
            "delayed refund", 
            "frustrated about refund",
            "angry about refund",
            "terrible refund process",
        ]
        for variant in emotional_variants:
            if variant not in entry["semantic_aliases"] and variant not in entry["questions"]:
                entry["questions"].append(variant)
    
    # ENHANCEMENT 3: Damaged/defective goods
    if "DAMAGED_ITEM" in [e.get("intent") for e in kb]:
        for entry in kb:
            if entry.get("intent") == "DAMAGED_ITEM":
                defect_variants = [
                    "Item arrived broken",
                    "The shoe is defective",
                    "My item doesn't work",
                    "This is completely broken",
                    "Arrived damaged and I'm upset",
                    "You sent me damaged goods",
                    "Item is not functioning",
                    "Defective product received",
                    "damaged goods",
                    "broken item",
                    "defectiv item",
                    "not working",
                    "came broken",
                    "arrived faulty",
                ]
                for variant in defect_variants:
                    if variant not in entry["semantic_aliases"] and variant not in entry["questions"]:
                        entry["questions"].append(variant)
    
    # ENHANCEMENT 4: Order tracking with typos
    if "TRACK_ORDER" in kb_by_intent:
        entry = kb_by_intent["TRACK_ORDER"]
        tracking_variants = [
            "Where's my package",
            "Still no package",
            "Why hasn't my order arrived",
            "Track my shipment",
            "Check delivery status",
            "Give me tracking info",
            "package delivery status",
            "shipment tracker",
            "trak shipment",
            "pakage status",
            "order status",
            "where is my order",
            "when will it arrive",
            "has my package shipped",
        ]
        for variant in tracking_variants:
            if variant not in entry["semantic_aliases"] and variant not in entry["questions"]:
                entry["questions"].append(variant)
    
    # ENHANCEMENT 5: Delayed delivery with emotional context
    if "DELAYED_DELIVERY" in kb_by_intent:
        entry = kb_by_intent["DELAYED_DELIVERY"]
        delay_variants = [
            "My order is late",
            "Order delayed, I'm frustrated",
            "Why is my delivery taking so long",
            "This delay is ridiculous",
            "I'm upset about the late delivery",
            "Order still hasn't arrived",
            "My delivery is delayed",
            "When will I get my order",
            "My order hasn't shipped yet",
            "order stuck",
            "delivery stuck",
            "still waiting for delivery",
            "order is late",
            "package delayed",
        ]
        for variant in delay_variants:
            if variant not in entry["semantic_aliases"] and variant not in entry["questions"]:
                entry["questions"].append(variant)
    
    # ENHANCEMENT 6: Payment failures
    if "PAYMENT_FAILED" in kb_by_intent:
        entry = kb_by_intent["PAYMENT_FAILED"]
        payment_variants = [
            "My card was declined",
            "Payment wouldn't go through",
            "Why did my payment fail",
            "Card rejected",
            "Payment error",
            "My payment failed",
            "Transaction declined",
            "Card not accepted",
            "payment wont work",
            "card declined error",
            "payment issue",
            "billing problem",
        ]
        for variant in payment_variants:
            if variant not in entry["semantic_aliases"] and variant not in entry["questions"]:
                entry["questions"].append(variant)
    
    # ENHANCEMENT 7: International shipping
    if "INTERNATIONAL_SHIPPING" in kb_by_intent:
        entry = kb_by_intent["INTERNATIONAL_SHIPPING"]
        intl_variants = [
            "Do you ship internationally?",
            "Can you ship to my country?",
            "International shipping cost",
            "How long for international delivery",
            "Do you deliver outside USA",
            "Ship to abroad",
            "Overseas shipping",
            "international delivery",
            "global shipping",
            "can you ship here",
            "ship to europe",
            "ship to asia",
            "world wide shipping",
        ]
        for variant in intl_variants:
            if variant not in entry["semantic_aliases"] and variant not in entry["questions"]:
                entry["questions"].append(variant)
    
    # ENHANCEMENT 8: Exchange
    if "EXCHANGE" in [e.get("intent") for e in kb]:
        for entry in kb:
            if entry.get("intent") == "EXCHANGE":
                exchange_variants = [
                    "Can I get a different size",
                    "I want to trade this for another color",
                    "Can I exchange my order",
                    "Swap for different item",
                    "I need a size exchange",
                    "different size",
                    "different color",
                    "want to swap",
                    "trade item",
                    "excange order",
                ]
                for variant in exchange_variants:
                    if variant not in entry["semantic_aliases"] and variant not in entry["questions"]:
                        entry["questions"].append(variant)
    
    # ENHANCEMENT 9: Wrong item
    if "WRONG_ITEM" in kb_by_intent:
        entry = kb_by_intent["WRONG_ITEM"]
        wrong_variants = [
            "I got the wrong item",
            "This is not what I ordered",
            "You sent me the wrong thing",
            "Wrong product arrived",
            "I received the incorrect item",
            "Not the right item",
            "wrong order",
            "wrong item received",
            "got wrong thing",
            "sent wrong item",
            "incorrect item",
            "different item than ordered",
        ]
        for variant in wrong_variants:
            if variant not in entry["semantic_aliases"] and variant not in entry["questions"]:
                entry["questions"].append(variant)
    
    # ENHANCEMENT 10: Missing items
    if "MISSING_ITEM" in kb_by_intent:
        entry = kb_by_intent["MISSING_ITEM"]
        missing_variants = [
            "Something is missing from my order",
            "I didn't receive everything",
            "Part of my order is missing",
            "My order arrived incomplete",
            "One item is missing",
            "Where's the rest of my order",
            "incomplete order",
            "missing items",
            "didnt get everything",
            "order incomplete",
        ]
        for variant in missing_variants:
            if variant not in entry["semantic_aliases"] and variant not in entry["questions"]:
                entry["questions"].append(variant)
    
    # Save expanded knowledge base
    with open("knowledge/knowledge_base.json", "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Knowledge base expanded")
    print(f"   Total entries: {len(kb)}")
    print(f"   New variants added for international returns, refund delays, damages, tracking, and payments")

if __name__ == "__main__":
    expand_knowledge_base()
