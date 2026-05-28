"""
──────────────────────────────────────────────────────────────────────────────
 Chroma Retriever — LangChain-compatible retriever with confidence filtering

 Wraps the ChromaStore to provide:
   - Top-K semantic retrieval using cosine similarity
   - Confidence threshold filtering (default min score = 0.55)
   - Structured result format for the pipeline
   - Query analysis for debugging / tracing
──────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import time
from typing import Any, Dict, List, Optional

from app.vectorstore.chroma_store import ChromaStore


class ChromaRetriever:
    """
    Retriever layer that wraps ChromaStore and enforces:
      - Minimum confidence threshold to prevent hallucination
      - Structured result format compatible with the pipeline
    """

    def __init__(
        self,
        chroma_store: ChromaStore,
        min_confidence: float = 0.55,
        top_k: int = 5,
    ):
        self.chroma_store = chroma_store
        self.min_confidence = float(os.getenv("MIN_ANSWER_CONFIDENCE", min_confidence))
        self.top_k = int(os.getenv("TOP_K_RETRIEVAL", top_k))

    def retrieve(
        self,
        query: str,
        previous_category: Optional[str] = None,
        previous_intent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Retrieve semantically relevant FAQ documents for a query.

        Pipeline:
          1. Embed query with all-MiniLM-L6-v2
          2. Cosine similarity search in ChromaDB HNSW index
          3. Apply lightweight semantic reranking
          4. Apply memory-aware boosting if previous context is provided
          5. Validate semantic relevance before returning chunks
          6. Filter by confidence threshold and return top-K results

        Args:
            query: User query (raw or rewritten)
            previous_category: Previous conversation category to boost (optional)
            previous_intent: Previous retrieved FAQ intent to boost (optional)

        Returns:
            {
                "results": List[dict],        # filtered, ranked results
                "query_analysis": dict,       # debug info
                "filtered_count": int,        # docs dropped by threshold
            }
        """
        start = time.time()

        out_of_scope_reason = self._out_of_scope_reason(query)
        if out_of_scope_reason:
            elapsed_ms = round((time.time() - start) * 1000, 2)
            return {
                "results": [],
                "query_analysis": {
                    "raw_query": query,
                    "top_raw_score": 0.0,
                    "top_reranked_score": 0.0,
                    "min_confidence_threshold": self.min_confidence,
                    "total_retrieved": 1,
                    "passed_threshold": 0,
                    "filtered_out": 1,
                    "retrieval_time_ms": elapsed_ms,
                    "category_boost_applied": False,
                    "previous_category": previous_category,
                    "previous_intent": previous_intent,
                    "fallback_trigger_reason": out_of_scope_reason,
                    "filter_note": "Query classified as outside ecommerce support scope.",
                    "rejection_reason": out_of_scope_reason,
                    "rejected_chunk_ids": [],
                    "reranking": [],
                },
                "filtered_count": 1,
            }

        # 1. Raw retrieval from ChromaDB
        raw_k = max(self.top_k, min(self.top_k * 3, 12))
        raw_results = self.chroma_store.similarity_search(query, k=raw_k)
        ranked_results = self._rerank(query, raw_results)

        # 2. Apply memory-aware boosting if previous context is available
        if previous_category or previous_intent:
            ranked_results = self._apply_context_boost(
                ranked_results,
                previous_category=previous_category,
                previous_intent=previous_intent,
            )

        # 3. Secondary semantic relevance validation + confidence filtering
        validated_results, rejected_results = self._validate_relevance(
            query,
            ranked_results,
            previous_category=previous_category,
            previous_intent=previous_intent,
        )
        filtered_results = [r for r in validated_results if r["score"] >= self.min_confidence][:self.top_k]
        low_confidence_rejections = [
            {**r, "rejection_reason": f"score_below_threshold:{r['score']}<{self.min_confidence}"}
            for r in validated_results
            if r["score"] < self.min_confidence
        ]
        all_rejections = rejected_results + low_confidence_rejections
        filtered_count = len(ranked_results) - len(filtered_results)

        elapsed_ms = round((time.time() - start) * 1000, 2)

        analysis = {
            "raw_query": query,
            "top_raw_score": raw_results[0]["score"] if raw_results else 0.0,
            "top_reranked_score": ranked_results[0]["score"] if ranked_results else 0.0,
            "min_confidence_threshold": self.min_confidence,
            "total_retrieved": len(raw_results),
            "passed_threshold": len(filtered_results),
            "filtered_out": filtered_count,
            "retrieval_time_ms": elapsed_ms,
            "category_boost_applied": previous_category is not None,
            "previous_category": previous_category,
            "previous_intent": previous_intent,
            "semantic_validation_applied": True,
            "fallback_trigger_reason": None,
            "rejection_reason": None,
            "rejected_chunk_ids": [r.get("id") for r in all_rejections],
            "reranking": [
                {
                    "id": r.get("id"),
                    "intent": r.get("intent"),
                    "category": r.get("category"),
                    "raw_score": r.get("raw_score", r.get("score")),
                    "score": r.get("score"),
                    "rerank_boost": r.get("rerank_boost", 0.0),
                    "category_boost": r.get("category_boost", 0.0),
                    "intent_boost": r.get("intent_boost", 0.0),
                    "validation": r.get("semantic_validation", {}),
                }
                for r in ranked_results[: self.top_k]
            ],
        }

        if not filtered_results:
            if all_rejections:
                top_rejection = all_rejections[0].get("rejection_reason", "semantic_mismatch")
                analysis["fallback_trigger_reason"] = top_rejection
                analysis["rejection_reason"] = top_rejection
            else:
                analysis["fallback_trigger_reason"] = "no_retrieval_candidates"
                analysis["rejection_reason"] = "no_retrieval_candidates"

        if filtered_count > 0:
            analysis["filter_note"] = (
                f"{filtered_count} doc(s) dropped by confidence or semantic validation"
            )

        return {
            "results": filtered_results,
            "query_analysis": analysis,
            "filtered_count": filtered_count,
        }

    def _out_of_scope_reason(self, query: str) -> Optional[str]:
        """Reject unsupported domains before semantic search."""
        text = query.lower()
        unsupported_patterns = {
            "insurance": r"\b(insurance|policy\s+premium|claim\s+settlement)\b",
            "investment": r"\b(invest|investment|investor|shares?|stock|ipo|equity|funding|valuation)\b",
            "crypto": r"\b(bitcoin|crypto|cryptocurrency|ethereum|wallet\s+address|nft)\b",
            "weather": r"\b(weather|temperature|rain|forecast|humidity)\b",
            "education": r"\b(teach\s+me|machine\s+learning|homework|write\s+an\s+essay|tutorial)\b",
            "financial_services": r"\b(mutual\s+fund|trading|loan|mortgage|bank\s+account\s+opening)\b",
        }
        ecommerce_terms = self._supported_domain_terms()
        has_ecommerce_context = any(re.search(rf"\b{re.escape(term)}\b", text) for term in ecommerce_terms)

        for reason, pattern in unsupported_patterns.items():
            if re.search(pattern, text) and not has_ecommerce_context:
                return f"out_of_domain:{reason}"

        # Very broad informational requests with no ecommerce support anchor
        generic_question = bool(re.search(r"\b(what|how|can|do|does|is|are|tell|teach|explain)\b", text))
        has_supported_signal = any(re.search(rf"\b{re.escape(term)}\b", text) for term in ecommerce_terms)
        if generic_question and not has_supported_signal:
            identity_or_greeting = bool(re.search(r"\b(your\s+name|who\s+are\s+you|hello|hi|hey|thanks|bye)\b", text))
            if not identity_or_greeting:
                return "out_of_domain:unsupported_topic"

        return None

    def _supported_domain_terms(self) -> set:
        return {
            "return", "returns", "refund", "refunds", "shipping", "ship",
            "delivery", "deliver", "payment", "payments", "order", "orders",
            "tracking", "track", "package", "item", "product", "coupon",
            "promo", "discount", "account", "password", "login", "wallet",
            "damaged", "broken", "defective", "wrong", "missing", "exchange",
            "size", "address", "invoice", "billing", "card", "subscription",
            "support", "agent", "human", "shopease", "store", "store credit",
            "international", "customs", "label", "pickup", "cancel",
            "send", "back", "shoes", "shoe", "footwear", "box", "packaging",
            "tags", "received", "arrived", "courier", "shipment",
            "money", "credited", "credit", "bank", "deducted",
        }

    def _rerank(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply small intent-specific lexical boosts on top of Chroma cosine scores.

        This keeps semantic retrieval as the primary signal while correcting
        common close-call FAQ collisions, e.g. "send shoes back" matching
        packaging because both mention shoeboxes.
        """
        query_text = query.lower()
        reranked = []

        for result in results:
            item = result.copy()
            score = float(item["score"])
            item["raw_score"] = round(score, 4)
            boost = self._lexical_boost(query_text, item)
            item["score"] = round(min(1.0, max(0.0, score + boost)), 4)
            if boost:
                item["rerank_boost"] = round(boost, 4)
            reranked.append(item)

        reranked.sort(key=lambda r: r["score"], reverse=True)
        return reranked

    def _validate_relevance(
        self,
        query: str,
        results: List[Dict[str, Any]],
        previous_category: Optional[str] = None,
        previous_intent: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Reject semantically weak chunks even when cosine returns a nearest neighbor."""
        accepted = []
        rejected = []
        query_terms = self._content_terms(query)
        query_text = query.lower()
        supported_terms = self._supported_domain_terms()
        domain_terms = {term for term in query_terms if term in supported_terms}

        for result in results:
            item = result.copy()
            doc_terms = self._content_terms(
                " ".join([
                    item.get("category", ""),
                    item.get("intent", "").replace("_", " "),
                    " ".join(item.get("tags", [])),
                    item.get("text", ""),
                ])
            )
            overlap = query_terms & doc_terms
            category_aligned = self._category_aligned(query_text, item.get("category", ""), item.get("intent", ""))
            memory_aligned = bool(
                previous_category and item.get("category") == previous_category
                or previous_intent and item.get("intent") == previous_intent
            )
            high_semantic_score = item.get("raw_score", item["score"]) >= max(self.min_confidence + 0.10, 0.65)

            validation = {
                "keyword_overlap": sorted(overlap)[:8],
                "keyword_overlap_count": len(overlap),
                "domain_terms": sorted(domain_terms),
                "category_aligned": category_aligned,
                "memory_aligned": memory_aligned,
                "high_semantic_score": high_semantic_score,
            }
            item["semantic_validation"] = validation

            is_relevant = (
                (domain_terms and (overlap or category_aligned or memory_aligned))
                or (domain_terms and high_semantic_score)
                or (memory_aligned and len(overlap) >= 1)
            )

            if is_relevant:
                accepted.append(item)
            else:
                item["rejection_reason"] = (
                    "semantic_mismatch:"
                    f"overlap={len(overlap)},domain_terms={len(domain_terms)},"
                    f"category_aligned={category_aligned},memory_aligned={memory_aligned}"
                )
                rejected.append(item)

        return accepted, rejected

    def _content_terms(self, text: str) -> set:
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "do", "does",
            "did", "can", "could", "would", "should", "i", "you", "your",
            "my", "me", "we", "our", "to", "of", "in", "on", "for", "with",
            "and", "or", "but", "what", "how", "when", "where", "why", "about",
            "this", "that", "it", "there", "be", "have", "has", "had",
        }
        aliases = {
            "trak": "track",
            "pakage": "package",
            "recieved": "received",
            "retun": "return",
            "retrn": "return",
            "shiping": "shipping",
            "mony": "money",
        }
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        return {aliases.get(token, token) for token in tokens if len(token) > 1 and token not in stopwords}

    def _category_aligned(self, query_text: str, category: str, intent: str) -> bool:
        category_text = f"{category} {intent}".lower()
        category_patterns = {
            "return": r"\b(return|returns|send\s+back|label|pickup|exchange)\b",
            "refund": r"\b(refund|money\s+back|credited|credit|where\s+money)\b",
            "shipping": r"\b(ship|shipping|delivery|international|customs|package|courier)\b",
            "orders": r"\b(order|tracking|track|wrong|missing|cancel|modify)\b",
            "payment": r"\b(payment|card|billing|invoice|charged|transaction|pay)\b",
            "account": r"\b(account|password|login|email|phone|profile)\b",
            "products": r"\b(product|damaged|broken|defective|size|warranty)\b",
        }
        for label, pattern in category_patterns.items():
            if re.search(pattern, query_text) and label in category_text:
                return True
        return False

    def _apply_context_boost(
        self,
        results: List[Dict[str, Any]],
        previous_category: Optional[str] = None,
        previous_intent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Apply previous-turn category and intent boosts to results.
        
        Documents matching the previous topic are boosted to improve follow-up
        context preservation while keeping Chroma cosine similarity primary.
        """
        boosted = []
        previous_category_lower = (previous_category or "").lower()
        previous_intent_upper = (previous_intent or "").upper()

        related_intents = {
            "RETURN_POLICY": {"RETURN_POLICY", "REFUND_STATUS", "RETURN_COST", "RETURN_LABEL", "EXCHANGE_PROCESS"},
            "REFUND_STATUS": {"REFUND_STATUS", "RETURN_POLICY", "PAYMENT_FAILED"},
            "INTERNATIONAL_SHIPPING": {"INTERNATIONAL_SHIPPING", "DELIVERY_TIMEFRAME", "SHIPPING_COST", "RETURN_POLICY"},
            "EXCHANGE_PROCESS": {"EXCHANGE_PROCESS", "RETURN_POLICY", "RETURN_COST"},
            "TRACK_ORDER": {"TRACK_ORDER", "DELIVERY_TIMEFRAME", "DELAYED_DELIVERY"},
            "PAYMENT_FAILED": {"PAYMENT_FAILED", "REFUND_STATUS", "PAYMENT_METHODS"},
        }
        
        for result in results:
            item = result.copy()
            result_category = item.get("category", "").lower()
            result_intent = item.get("intent", "").upper()
            
            # Apply boost if categories match or are semantically related
            category_match = result_category == previous_category_lower
            
            # Semantic category relationships for cross-boost
            # e.g., returns & refunds category should boost refund-related documents
            category_relationships = {
                "returns & refunds": {"refund", "return", "exchange"},
                "shipping": {"delivery", "shipping", "track", "order status"},
                "shipping & delivery": {"delivery", "shipping", "track", "order status"},
                "payment": {"payment", "card", "billing", "account"},
                "payment & accounts": {"payment", "card", "billing", "account"},
                "orders": {"order", "product", "item"},
                "products": {"order", "product", "item"},
                "products & orders": {"order", "product", "item"},
            }
            
            category_related = False
            if previous_category_lower and previous_category_lower in category_relationships:
                related_keywords = category_relationships[previous_category_lower]
                result_intent_lower = item.get("intent", "").lower()
                for keyword in related_keywords:
                    if keyword in result_intent_lower:
                        category_related = True
                        break
            
            if category_match:
                # Strong boost for exact category match
                boost = 0.20
                item["category_boost"] = round(boost, 4)
                item["score"] = round(min(1.0, item["score"] + boost), 4)
            elif category_related:
                # Moderate boost for related categories
                boost = 0.08
                item["category_boost"] = round(boost, 4)
                item["score"] = round(min(1.0, item["score"] + boost), 4)

            if previous_intent_upper and result_intent:
                if result_intent == previous_intent_upper:
                    boost = 0.18
                elif result_intent in related_intents.get(previous_intent_upper, set()):
                    boost = 0.10
                else:
                    boost = 0.0

                if boost:
                    item["intent_boost"] = round(boost, 4)
                    item["score"] = round(min(1.0, item["score"] + boost), 4)
            
            boosted.append(item)
        
        # Re-sort by updated scores
        boosted.sort(key=lambda r: r["score"], reverse=True)
        return boosted

    def _lexical_boost(self, query_text: str, result: Dict[str, Any]) -> float:
        intent = result.get("intent", "")
        boost = 0.0

        # ── RETURN POLICY BOOSTS ──────────────────────────────────────────
        return_action = bool(
            re.search(
                r"\b(send|sending|sent|ship|shipping|give|return|mail)\b.*\b(back|return)\b",
                query_text,
            )
            or re.search(r"\b(return|returns|returned|returning|refund)\b", query_text)
        )
        footwear = bool(re.search(r"\b(shoe|shoes|sneaker|sneakers|footwear|boots?|sandal)\b", query_text))
        packaging_terms = bool(re.search(r"\b(box|packag|tag|shoebox|carton|label|damage|damaged)\b", query_text))
        international = bool(re.search(r"\b(international|overseas|abroad|outside|other\s+country|world|global)\b", query_text))
        exchange = bool(re.search(r"\b(exchange|swap|replace|different\s+size|different\s+color)\b", query_text))

        if intent == "RETURN_POLICY" and (return_action or (international and re.search(r"\b(order|orders|policy|return|returns)\b", query_text))):
            boost += 0.15
            if footwear:
                boost += 0.08
            if international:
                boost += 0.18
            if exchange:
                boost += 0.10

        if intent == "RETURN_POLICY" and re.search(r"\b(retun|retrn|return)\b.*\b(policcy|policy|polcy)\b", query_text):
            boost += 0.22

        if intent == "NO_BOX" and footwear and not packaging_terms:
            boost -= 0.12

        # ── REFUND STATUS BOOSTS ──────────────────────────────────────────
        refund_terms = bool(re.search(r"\b(refund|money\s+back|my\s+money|credited|credit|reimburse)\b", query_text))
        refund_delay_terms = bool(
            re.search(
                r"\b(still|pending|delay|delayed|arrived|received|credited|come|came|get|got|where|status|not|waiting|process)\b",
                query_text,
            )
        )
        
        # IMPROVED: Distinguish between refund and store credit
        # "where is my money" in refund context should strongly boost REFUND_STATUS
        my_money_with_context = bool(
            re.search(r"\b(where|where'?s|where\s+is)\b.*\b(my\s+money|money|refund)\b", query_text)
            or re.search(r"\b(my\s+money|refund|money\s+back)\b.*\b(where|arrived|come|got|still|pending|status)\b", query_text)
        )

        if intent == "REFUND_STATUS" and (refund_terms or my_money_with_context):
            boost += 0.16
            if refund_delay_terms:
                boost += 0.12
            if international:
                boost += 0.10
        
        # Penalize store credit for refund queries
        if intent == "STORE_CREDIT_BALANCE" and my_money_with_context:
            boost -= 0.15

        if intent == "RETURN_POLICY" and refund_terms and not return_action:
            boost -= 0.08  # Slightly lower confidence for refund queries on return policy

        # ── DELIVERY TIMEFRAME BOOSTS ──────────────────────────────────────
        delivery_time_terms = bool(
            re.search(r"\b(how\s+long|when|eta|delivery\s+time|shipping\s+time|arrive|take|days?)\b", query_text)
        )
        delay_terms = bool(re.search(r"\b(delay|delayed|late|not\s+arrived|hasn'?t\s+arrived|taking\s+too\s+long)\b", query_text))

        if intent == "DELIVERY_TIMEFRAME" and delivery_time_terms and not delay_terms:
            boost += 0.15
            if international:
                boost += 0.12

        if intent == "DELAYED_DELIVERY" and delivery_time_terms and delay_terms:
            boost += 0.16

        if intent == "DELAYED_DELIVERY" and delay_terms:
            boost += 0.10

        # ── SHIPPING COST BOOSTS ──────────────────────────────────────────
        shipping_price_terms = bool(
            re.search(r"\b(shipping|delivery|ship|freight)\b", query_text)
            and re.search(r"\b(free|cost|fee|price|charge|charges|paid|pay)\b", query_text)
        )

        if intent == "SHIPPING_COST" and shipping_price_terms:
            boost += 0.18
            if international:
                boost += 0.12

        if intent == "RETURN_COST" and shipping_price_terms and return_action:
            boost += 0.14

        if intent == "RETURN_COST" and shipping_price_terms and not return_action:
            boost -= 0.12

        # ── ORDER TRACKING BOOSTS ──────────────────────────────────────────
        tracking_terms = bool(re.search(r"\b(track|tracking|where|status|package|trak|pakage)\b", query_text))
        order_terms = bool(re.search(r"\b(order|my\s+order|my\s+purchase)\b", query_text))

        if intent == "TRACK_ORDER" and tracking_terms:
            boost += 0.15
            if order_terms:
                boost += 0.08

        # ── WRONG ITEM / DAMAGED BOOSTS ────────────────────────────────────
        wrong_terms = bool(re.search(r"\b(wrong|incorrect|different|not\s+what|not\s+the)\b", query_text))
        received_terms = bool(re.search(r"\b(received|got|arrived|sent|recieved)\b", query_text))
        damaged_terms = bool(re.search(r"\b(damaged|broken|defective|not\s+working|not\s+work)\b", query_text))

        if intent == "WRONG_ITEM" and wrong_terms and received_terms:
            boost += 0.18

        if intent in ("DAMAGED_ITEM", "DAMAGED_PRODUCT") and damaged_terms:
            boost += 0.16
            if received_terms:
                boost += 0.08

        # ── PAYMENT / CARD ISSUES BOOSTS ───────────────────────────────────
        payment_terms = bool(re.search(r"\b(payment|card|credit|debit|charge|bill|invoice)\b", query_text))
        failed_terms = bool(re.search(r"\b(failed|decline|rejected|error|not\s+accepted)\b", query_text))

        if intent == "PAYMENT_FAILED" and payment_terms and failed_terms:
            boost += 0.16

        # ── EXCHANGE / SIZE / COLOR BOOSTS ────────────────────────────────
        size_terms = bool(re.search(r"\b(size|fit|too\s+small|too\s+big|larger|smaller)\b", query_text))
        color_terms = bool(re.search(r"\b(color|colour|shade|different\s+color)\b", query_text))

        if intent in ("EXCHANGE", "EXCHANGE_PROCESS") and exchange:
            boost += 0.14
            if size_terms or color_terms:
                boost += 0.08

        # ── TYPO HANDLING ──────────────────────────────────────────────────
        # Common typos in ecommerce queries
        if re.search(r"\b(trak|track)\b", query_text) and intent == "TRACK_ORDER":
            boost += 0.08
        if re.search(r"\b(pakage|package)\b", query_text) and intent in ("TRACK_ORDER", "DELIVERY_TIMEFRAME"):
            boost += 0.08
        if re.search(r"\b(recieved|received)\b", query_text) and intent in ("WRONG_ITEM", "DAMAGED_ITEM"):
            boost += 0.06
        if re.search(r"\b(refund|refund)\b", query_text) and intent == "REFUND_STATUS":
            boost += 0.08
        if re.search(r"\b(retrn|retun)\b", query_text) and intent == "RETURN_POLICY":
            boost += 0.08
        if re.search(r"\b(where\s+money|where\s+is\s+my\s+mony|refund\s+not\s+came)\b", query_text) and intent == "REFUND_STATUS":
            boost += 0.10

        return boost

    def get_langchain_retriever(self):
        """Return raw LangChain VectorStoreRetriever for chain integration."""
        return self.chroma_store.get_langchain_retriever(k=self.top_k)
