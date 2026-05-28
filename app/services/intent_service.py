"""
──────────────────────────────────────────────────────────────────────────────
 Intent Classifier — Detects user intent before and after RAG retrieval

 Pre-retrieval intents (pattern + sentiment):
   GREETING       – "hi", "hello", "hey"
   FAREWELL       – "bye", "thanks", "that's all"
   ESCALATION     – "speak to human", "manager", "agent"
   FRUSTRATION    – angry/upset language + negative sentiment
   FAQ_QUERY      – everything else → route to RAG

 Post-retrieval refinement:
   If top retrieved doc has escalate=True  → override to ESCALATION
   If no docs matched                      → set NO_MATCH
──────────────────────────────────────────────────────────────────────────────
"""

import re
from typing import Any, Dict, List, Optional


class IntentService:
    """Rule-based intent classifier with lexicon sentiment analysis."""

    def __init__(self):
        # ── Pattern Rules ────────────────────────────────────────────────
        self.patterns = {
            "GREETING": re.compile(
                r"^(hi|hello|hey|howdy|hiya|greetings|good\s+(morning|afternoon|evening|day)|sup|yo|what'?s\s+up)\b",
                re.IGNORECASE,
            ),
            "FAREWELL": re.compile(
                r"\b(bye|goodbye|thanks|thank\s*you|cheers|see\s*you|take\s*care|"
                r"that'?s\s*all|no\s*more\s*questions|all\s*good|have\s*a\s*(good|great|nice)\s*(day|one))\b",
                re.IGNORECASE,
            ),
            # ENHANCED: Much more comprehensive frustration keywords
            "FRUSTRATION": re.compile(
                r"\b("
                # Anger/Fury
                r"angry|furious|rage|enraged|livid|seething|infuriated|"
                # Disappointment
                r"disappointed|let\s*down|devastated|heartbroken|"
                # Frustration/Upset
                r"frustrated|upset|annoyed|irritated|exasperated|agitated|"
                # Quality issues
                r"terrible|awful|horrible|worst|dreadful|abysmal|pathetic|"
                # Service issues
                r"unacceptable|ridiculous|outrageous|absurd|disgusting|"
                # Support issues
                r"useless|incompetent|incompetence|worthless|rubbish|junk|"
                # Fraud/Scam related
                r"scam|ripped?\s*off|fraud|cheated|stolen|thieved|"
                # Fed up / Done
                r"fed\s*up|sick\s*of|done\s*with|can't\s*take\s*it|"
                r"exhausted|tired\s+of|drained|overwhelmed|"
                # Never again
                r"never\s*again|never\s*coming\s*back|not\s*coming\s*back|"
                # Waste of time
                r"waste\s*of\s*(time|money|effort)|"
                # Hate/Dislike
                r"hate|despise|detest|loathe|abhor|"
                # Emotional intensity
                r"so\s*frustrated|completely\s*frustrated|extremely\s*frustrated|very\s*frustrated|"
                r"so\s*angry|very\s*angry|extremely\s*angry|"
                r"so\s*disappointed|very\s*disappointed|extremely\s*disappointed|"
                # Help/Support not received
                r"nobody\s+(is\s+)?helping|nobody\s+helps|no\s+one\s+(is\s+)?helping|no\s+help|not\s+helping|"
                r"worst\s+support|terrible\s+support|poor\s+support|"
                r"worst\s+service|terrible\s+service|poor\s+service|"
                r"worst\s+experience|terrible\s+experience|poor\s+experience|"
                # Repeated issues
                r"still\s+waiting|still\s+not|still\s+haven't|still\s+no|"
                # Legal/Complaint threats
                r"complaint|complain|report|lawsuit|sue|lawyer|court|legal"
                r")\b",
                re.IGNORECASE,
            ),
            "ESCALATION": re.compile(
                r"\b(human|real\s*person|manager|supervisor|escalate|transfer\s*me|"
                r"speak\s*to\s*(someone|a\s*human|agent|person)|"
                r"talk\s*to\s*(someone|a\s*human|agent|person)|"
                r"connect\s*me|live\s*agent|live\s*chat|call\s*me|"
                r"legal|lawyer|sue|court|consumer\s*forum|file\s*complaint)\b",
                re.IGNORECASE,
            ),
        }

        # ── Sentiment Lexicons ───────────────────────────────────────────
        self.positive_words = {
            "great", "good", "love", "excellent", "awesome", "amazing",
            "perfect", "helpful", "fast", "quick", "easy", "best",
            "thank", "thanks", "happy", "wonderful", "fantastic",
            "brilliant", "superb", "outstanding", "satisfied",
        }
        # ENHANCED: Expanded negative word list for better sentiment detection
        self.negative_words = {
            "bad", "terrible", "awful", "hate", "worst", "horrible",
            "poor", "slow", "broken", "wrong", "fail", "failed",
            "issue", "problem", "disappointed", "frustrat", "angry",
            "useless", "scam", "pathetic", "ridiculous", "disgust",
            "unacceptable", "outrageous", "incompetent", "rubbish",
            "angry", "furious", "upset", "annoyed", "irritated",
            "agitated", "exasperated", "enraged", "infuriated",
            "devastated", "heartbroken", "devastation",
            "disaster", "catastrophe", "nightmare",
            "abysmal", "dreadful", "disgusting", "absurd",
            "fraud", "scammed", "ripped", "cheated", "stolen",
            "waste", "wasted", "worthless", "junk", "ruined",
            "never", "unfair", "unjust", "unreasonable",
            "sick", "tired", "exhausted", "overwhelmed",
        }
        self.negation_words = {
            "not", "no", "never", "neither", "nor", "dont", "doesnt",
            "didnt", "cant", "wont", "hardly", "barely", "without",
        }

    def classify(self, raw_text: str, processed: Dict) -> Dict[str, Any]:
        """
        Classify user intent from raw input.

        Returns:
            {
                "intent": str,
                "confidence": float,
                "sentiment": str,       # "positive" | "negative" | "neutral"
                "sentiment_score": float,  # [-1, 1]
                "meta": dict,
            }
        """
        text = raw_text.lower().strip()
        tokens = processed["tokens"]
        sentiment = self._analyze_sentiment(tokens)

        # 1. Greeting (only if short message)
        if len(text) < 30 and self.patterns["GREETING"].search(text):
            return self._result("GREETING", 0.95, sentiment)

        # 2. Farewell (only if short)
        if self.patterns["FAREWELL"].search(text) and len(tokens) <= 5:
            return self._result("FAREWELL", 0.90, sentiment)

        # 3. Explicit escalation / legal threat request
        if self.patterns["ESCALATION"].search(text):
            reason = "Customer requested human agent or made legal threat."
            trigger = "explicit_request"
            if re.search(r"\b(legal|lawyer|sue|court|forum|complaint)\b", text, re.IGNORECASE):
                trigger = "legal_threat"
                reason = "Customer made a legal threat."
            return self._result(
                "ESCALATION", 0.95, sentiment,
                meta={"reason": reason, "trigger": trigger},
            )

        # 4. Frustration (angry language + negative sentiment, OR strong negative sentiment alone)
        # Check both explicit frustration keywords AND sentiment-based frustration
        has_frustration_keywords = bool(self.patterns["FRUSTRATION"].search(text))
        has_strong_negative_sentiment = sentiment["score"] < -0.5
        has_negative_sentiment = sentiment["score"] < -0.2
        
        # Trigger frustration if:
        # 1. Explicit frustration keywords + negative sentiment
        # 2. OR very strong negative sentiment alone (strong emotional response)
        if (has_frustration_keywords and has_negative_sentiment) or has_strong_negative_sentiment:
            return self._result(
                "FRUSTRATION", 0.85 if has_frustration_keywords else 0.75, sentiment,
                meta={"reason": "Customer appears frustrated or upset."},
            )

        # 5. Default → FAQ query
        return self._result("FAQ_QUERY", 0.70, sentiment)

    def refine_with_retrieval(
        self,
        initial_intent: Dict,
        retrieval_results: List[Dict],
    ) -> Dict[str, Any]:
        """
        Post-retrieval intent refinement.

        - If top retrieved doc has escalate=True → override to ESCALATION
        - If no results found → set NO_MATCH
        """
        if not retrieval_results:
            if initial_intent["intent"] == "FAQ_QUERY":
                return {
                    **initial_intent,
                    "intent": "NO_MATCH",
                    "confidence": 0.60,
                    "meta": {"reason": "No relevant documents found in knowledge base."},
                }
            return initial_intent

        top = retrieval_results[0]

        if top.get("escalate", False):
            return {
                **initial_intent,
                "intent": "ESCALATION",
                "confidence": max(initial_intent["confidence"], top["score"]),
                "meta": {
                    "reason": top.get("escalation_reason", "Matched escalation trigger."),
                    "trigger": "knowledge_base_match",
                    "matched_doc_id": top["id"],
                },
            }

        return initial_intent

    # ── Sentiment Analysis ────────────────────────────────────────────────────

    def _analyze_sentiment(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Lexicon-based sentiment analysis with negation handling.
        Returns { "label": str, "score": float }
        """
        score = 0
        negation_active = False

        for token in tokens:
            if token in self.negation_words:
                negation_active = True
                continue

            token_score = 0
            if any(token.startswith(p) for p in self.positive_words):
                token_score = 1
            elif any(token.startswith(n) for n in self.negative_words):
                token_score = -1

            if negation_active and token_score != 0:
                token_score *= -1
                negation_active = False

            score += token_score

        max_possible = len(tokens) or 1
        normalized = max(-1.0, min(1.0, score / (max_possible ** 0.5)))

        label = "neutral"
        if normalized > 0.15:
            label = "positive"
        elif normalized < -0.15:
            label = "negative"

        return {"label": label, "score": round(normalized, 2)}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _result(
        self,
        intent: str,
        confidence: float,
        sentiment: Dict,
        meta: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        return {
            "intent": intent,
            "confidence": round(confidence, 2),
            "sentiment": sentiment["label"],
            "sentiment_score": sentiment["score"],
            "meta": meta or {},
        }
