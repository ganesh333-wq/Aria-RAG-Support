"""
──────────────────────────────────────────────────────────────────────────────
 LLM Service — Groq-only generation for the RAG pipeline

 Architecture:
   Provider: ChatGroq
             - Model: llama-3.3-70b-versatile by default
             - Used for response generation and LLM-assisted query rewriting

 If Groq is not configured or the API call fails, Aria uses deterministic or
 template responses instead of routing to another LLM provider.
──────────────────────────────────────────────────────────────────────────────
"""

import os
import re
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import StrOutputParser

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    ChatGroq = None
    GROQ_AVAILABLE = False

from app.prompts.templates import (
    RAG_PROMPT,
    FRUSTRATION_PROMPT,
    QUERY_REWRITE_PROMPT,
    QUERY_REWRITE_WITH_CONTEXT_PROMPT,
    GREETING_RESPONSE,
    FAREWELL_RESPONSE,
    NO_MATCH_RESPONSE,
    ESCALATION_RESPONSE,
)


class LLMService:
    """
    Groq-only LLM service for Aria.

    Usage:
        llm_service = LLMService()
        await llm_service.initialize()
        response = await llm_service.generate_response(...)
    """

    def __init__(self):
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")

        self.groq_available: bool = bool(self.groq_api_key and GROQ_AVAILABLE)

        self._llm = None

        # Stats
        self.request_count = 0
        self.total_tokens_used = 0
        self.groq_requests = 0

    async def initialize(self) -> Dict[str, Any]:
        """
        Initialize the Groq chat model.
        Call once at startup.
        """
        if self.groq_available and GROQ_AVAILABLE:
            self._llm = ChatGroq(
                model=self.groq_model,
                api_key=self.groq_api_key,
                temperature=0.3,
                max_tokens=512,
            )
            print(f"[LLMService] LLM provider: Groq ({self.groq_model})")
        else:
            print("[LLMService] Groq not configured - deterministic/template responses only")

        active = "groq" if self.groq_available and self._llm else "template"
        print(f"[LLMService] Active mode: {active}")

        return {
            "groq_available": self.groq_available,
            "active_mode": active,
            "groq_model": self.groq_model,
        }

    # ── LLM Selection ─────────────────────────────────────────────────────────

    def _get_active_llm(self):
        """Return the configured Groq LLM, or None when unavailable."""
        if self.groq_available and self._llm:
            return self._llm, "groq"
        return None, "template"

    # ── Query Rewriting ───────────────────────────────────────────────────────

    async def rewrite_query(
        self, 
        query: str, 
        conversation_history: str,
        previous_intent: Optional[str] = None,
        previous_category: Optional[str] = None,
    ) -> str:
        """
        Rewrite a follow-up query into a standalone query using LLM.

        Example:
            History: "Customer asked about return policy"
            Follow-up: "What about international?"
            Rewritten: "What is the international return policy?"

        Args:
            query: The follow-up query to rewrite
            conversation_history: Formatted conversation history
            previous_intent: The intent from the previous turn (e.g., "RETURN_POLICY")
            previous_category: The category from the previous turn (e.g., "Returns & Refunds")

        Falls back to the original query if rewriting fails.
        """
        if not conversation_history or not conversation_history.strip():
            return query

        inferred_intent, inferred_category = self._infer_previous_topic(
            conversation_history,
            previous_intent,
            previous_category,
        )
        deterministic = self._deterministic_rewrite(query, inferred_intent, inferred_category)
        if deterministic != query:
            print(f"[LLMService] Query rewritten deterministically: '{query}' → '{deterministic}'")
            return deterministic

        llm, source = self._get_active_llm()
        if llm is None:
            return query

        try:
            # Use context-aware prompt if previous intent/category available
            if inferred_intent and inferred_category:
                chain = QUERY_REWRITE_WITH_CONTEXT_PROMPT | llm | StrOutputParser()
                rewritten = await chain.ainvoke({
                    "history": conversation_history[-500:],
                    "question": query,
                    "previous_intent": inferred_intent,
                    "previous_category": inferred_category,
                })
            else:
                # Fall back to basic query rewrite
                chain = QUERY_REWRITE_PROMPT | llm | StrOutputParser()
                rewritten = await chain.ainvoke({
                    "history": conversation_history[-500:],
                    "question": query,
                })
            
            rewritten = rewritten.strip()

            # Validate rewrite: reject if it looks like an answer, not a question
            if (
                len(rewritten) > len(query) + 120
                or "here is" in rewritten.lower()
                or rewritten.startswith(("I ", "Sure", "Of course"))
            ):
                return query

            print(f"[LLMService] Query rewritten: '{query}' → '{rewritten}'")
            return rewritten

        except Exception as e:
            print(f"[LLMService] Query rewrite failed: {e}")
            return self._deterministic_rewrite(query, inferred_intent, inferred_category)

    def _infer_previous_topic(
        self,
        conversation_history: str,
        previous_intent: Optional[str] = None,
        previous_category: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[str]]:
        """Infer prior ecommerce topic when callers only pass conversation text."""
        if previous_intent or previous_category:
            return previous_intent, previous_category

        history = (conversation_history or "").lower()
        if re.search(r"\b(return|returns|send back|30 days|return policy)\b", history):
            return "RETURN_POLICY", "Returns & Refunds"
        if re.search(r"\b(refund|money back|credited)\b", history):
            return "REFUND_STATUS", "Returns & Refunds"
        if re.search(r"\b(track|tracking|package|shipment)\b", history):
            return "TRACK_ORDER", "Shipping"
        if re.search(r"\b(payment|card|billing|transaction)\b", history):
            return "PAYMENT_FAILED", "Payment"

        return previous_intent, previous_category

    def _deterministic_rewrite(
        self,
        query: str,
        previous_intent: Optional[str] = None,
        previous_category: Optional[str] = None,
    ) -> str:
        """Rewrite common ecommerce follow-ups without requiring an LLM."""
        text = query.lower().strip()
        previous_intent = (previous_intent or "").upper()
        previous_category = (previous_category or "").lower()

        international = bool(re.search(r"\b(international|overseas|abroad|outside|country|countries|global)\b", text))
        followup = bool(re.search(r"\b(what about|how about|and|also|what if|but what)\b", text)) or len(text.split()) <= 5

        if not followup:
            return query

        if previous_intent == "RETURN_POLICY" and international:
            return "What is the international return policy?"
        if previous_intent == "REFUND_STATUS" and international:
            return "What is the refund status policy for international returns?"
        if previous_intent in {"RETURN_POLICY", "EXCHANGE_PROCESS"} and re.search(r"\b(exchange|swap|size|color|colour)\b", text):
            return "How does exchange work for returned items?"
        if previous_intent == "TRACK_ORDER" and re.search(r"\b(international|overseas|abroad|outside|country|countries|global)\b", text):
            return "How can I track an international order?"
        if previous_category == "returns & refunds" and international:
            return "What is the international returns and refunds policy?"

        return query

    # ── Response Generation ───────────────────────────────────────────────────

    async def generate_response(
        self,
        query: str,
        retrieved_docs: List[Dict],
        intent: Dict,
        conversation_history: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response using RAG context + LLM.

        Flow:
          1. Check for deterministic responses (greetings, farewells, escalations)
          2. Build context from retrieved documents
          3. Select appropriate prompt template (RAG vs Frustration)
          4. Try Groq → fallback to template if unavailable or failed

        Args:
            query: User question
            retrieved_docs: List of retrieved FAQ chunks with scores
            intent: Classified intent dict
            conversation_history: Formatted prior conversation string

        Returns:
            { "response": str, "source": str, "model": str, "tokens_used": int }
        """
        intent_type = intent.get("intent", "FAQ_QUERY")
        history_str = conversation_history or ""

        # ── Deterministic shortcuts (no LLM needed) ───────────────────────
        deterministic = self._get_deterministic_response(query, intent_type, retrieved_docs)
        if deterministic:
            return deterministic

        # ── Build context from retrieved docs ─────────────────────────────
        context = self._format_context(retrieved_docs)

        # ── Select prompt template ────────────────────────────────────────
        prompt = FRUSTRATION_PROMPT if intent_type == "FRUSTRATION" else RAG_PROMPT

        # ── Try Groq, then safe template fallback ────────────────────────
        result = await self._invoke_groq_or_template(
            prompt=prompt,
            context=context,
            question=query,
            history=history_str,
        )

        # ── Inject empathy for frustrated customers ────────────────────────
        # This ensures empathy even if the LLM didn't produce it
        empathy_reason = self.empathy_trigger_reason(intent, query)
        result["response"] = self.inject_empathy(result["response"], intent, query)
        result["empathy_trigger_reason"] = empathy_reason

        # ── Validate response quality ──────────────────────────────────────
        validation = self.validate_response_quality(result["response"])
        result["response_validation"] = validation
        
        # If response quality is poor and we have retrieved docs, fall back to template
        if not validation["is_valid"] and result["source"] != "template":
            print(f"[LLMService] Response quality poor (score: {validation['score']}), falling back to template")
            result["response"] = self._template_fallback(context)["response"]
            result["response"] = self.inject_empathy(result["response"], intent, query)
            result["source"] = "template"
            result["response_validation"] = self.validate_response_quality(result["response"])

        self.request_count += 1
        return result

    async def _invoke_groq_or_template(
        self,
        prompt,
        context: str,
        question: str,
        history: str,
    ) -> Dict[str, Any]:
        """Try Groq, then return a safe template response if unavailable."""
        input_vars = {
            "context": context,
            "question": question,
            "history": history,
        }

        if self.groq_available and self._llm:
            try:
                chain = prompt | self._llm | StrOutputParser()
                response = await chain.ainvoke(input_vars)
                self.groq_requests += 1
                return {
                    "response": response.strip(),
                    "source": "groq",
                    "model": self.groq_model,
                    "tokens_used": 0,
                }
            except Exception as e:
                print(f"[LLMService] Groq failed: {e} - using template fallback")

        # Template fallback
        return self._template_fallback(context)

    # ── Deterministic Responses ───────────────────────────────────────────────

    def _get_deterministic_response(
        self,
        query: str,
        intent_type: str,
        retrieved_docs: List[Dict],
    ) -> Optional[Dict[str, Any]]:
        """
        Return pre-defined responses for cases where LLM is unnecessary
        or must not be used (greetings, farewells, identity questions).
        """
        text = query.lower().strip()

        # Identity / creator questions
        if self._is_identity_question(text):
            return self._static("I'm **Aria**, the ShopEase support assistant. How can I help you today? 😊")

        # Competitor questions
        if self._is_competitor_question(text):
            return self._static(
                "I can only assist with ShopEase-related questions. "
                "Is there something I can help you with regarding your ShopEase account or orders?"
            )

        # Intent-based deterministic responses
        if intent_type == "GREETING":
            return self._static(GREETING_RESPONSE)
        if intent_type == "FAREWELL":
            return self._static(FAREWELL_RESPONSE)
        if intent_type == "ESCALATION":
            return self._static(ESCALATION_RESPONSE)

        return None

    def _is_identity_question(self, text: str) -> bool:
        import re
        return bool(
            re.search(r"\b(what'?s|what\s+is|tell\s+me)\s+(your\s+)?name\b", text)
            or re.search(r"\b(who|what)\s+are\s+you\b", text)
            or re.search(r"\b(who|which\s*company|what\s*team).*\b(made|created|built|developed|trained)\b", text)
            or re.search(r"\b(made|created|built|developed|trained)\s+you\b", text)
            or re.search(r"\b(openai|google|anthropic|gpt|llama|claude|gemini)\b", text)
        )

    def _is_competitor_question(self, text: str) -> bool:
        import re
        competitors = (
            "amazon", "flipkart", "myntra", "ajio", "nykaa", "meesho",
            "snapdeal", "zara", "shein", "asos", "ebay", "alibaba",
        )
        return any(re.search(rf"\b{re.escape(c)}\b", text) for c in competitors)

    # ── Context / Template Helpers ────────────────────────────────────────────

    def _format_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents into a context block for the prompt."""
        if not retrieved_docs:
            return "No relevant information found in the knowledge base."

        parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            category = doc.get("category", "General")
            score = doc.get("score", 0.0)
            text = doc.get("text", "")
            # Extract only the Answer line from the chunk text for cleaner context
            answer_match = re.search(r"Answer:\s*(.+?)\s*$", text, re.DOTALL)
            answer_text = answer_match.group(1).strip() if answer_match else text[:500]

            parts.append(
                f"[Source {i} | Category: {category} | Relevance: {score*100:.0f}%]\n{answer_text}"
            )

        return "\n\n---\n\n".join(parts)

    def inject_empathy(
        self,
        response: str,
        intent: Dict,
        query: str,
    ) -> str:
        """
        Inject empathy statement before the response for frustrated customers.
        
        Args:
            response: The original response
            intent: The classified intent dict
            query: The original user query
            
        Returns:
            Response with empathy prefix (if applicable)
        """
        # Only inject empathy for frustrated customers
        if intent.get("intent") != "FRUSTRATION":
            return response
        
        # Don't inject if already starts with empathy
        empathy_starts = (
            "i understand", "i'm sorry", "i apologize", "i regret", "i hear you",
            "that's frustrating", "that must be", "i appreciate your", "thank you for your patience"
        )
        if any(response.lower().startswith(p) for p in empathy_starts):
            return response
        
        # Select appropriate empathy statement based on query content
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["refund", "money", "refunded", "payment"]):
            empathy = "I'm genuinely sorry for the frustration regarding your refund."
        elif any(word in query_lower for word in ["damaged", "broken", "defective", "not working"]):
            empathy = "I completely understand your frustration with receiving a damaged item."
        elif any(word in query_lower for word in ["wrong", "incorrect", "mistake"]):
            empathy = "I sincerely apologize for this mistake with your order."
        elif any(word in query_lower for word in ["delay", "delayed", "late", "taking long", "still waiting"]):
            empathy = "I understand your frustration with the delay, and I'm here to help."
        elif any(word in query_lower for word in ["lost", "missing", "not received", "never arrived"]):
            empathy = "I'm sorry to hear your order hasn't arrived. That must be really frustrating."
        elif "support" in query_lower or "help" in query_lower:
            empathy = "I apologize for any frustration you're experiencing."
        else:
            empathy = "I understand your frustration, and I'm here to help."
        
        # Prepend empathy to response
        return f"{empathy} {response}"

    def empathy_trigger_reason(self, intent: Dict, query: str) -> Optional[str]:
        """Return the debug reason for empathy injection, if it should activate."""
        if intent.get("intent") != "FRUSTRATION":
            return None
        query_lower = query.lower()
        emotional_terms = (
            "frustrated", "angry", "upset", "disappointed", "annoyed",
            "irritated", "exhausted", "terrible", "worst", "nobody",
            "no one", "not helping", "still waiting",
        )
        matched = [term for term in emotional_terms if term in query_lower]
        if matched:
            return f"frustration_intent emotional_terms={matched[:3]}"
        return "frustration_intent negative_sentiment"

    def validate_response_quality(self, response: str) -> Dict[str, Any]:
        """
        Validate that a response is complete and not truncated.
        
        Returns:
            {
                "is_valid": bool,
                "issues": list[str],
                "score": float,  # 0-1
            }
        """
        issues = []
        score = 1.0
        
        # Check 1: Empty response
        if not response or not response.strip():
            return {
                "is_valid": False,
                "issues": ["Response is empty"],
                "score": 0.0,
            }
        
        # Check 2: Too short response (might be truncated)
        response_length = len(response)
        if response_length < 20:
            issues.append("Response is very short (< 20 chars)")
            score -= 0.3
        
        # Check 3: Incomplete sentence detection
        # Common patterns of truncated responses
        truncation_patterns = [
            r":\s*$",  # Ends with colon (list continuation)
            r"\.\.\.\s*$",  # Ends with ellipsis
            r"\s*\(\s*$",  # Ends with open parenthesis
            r"^Here",  # Starts with "Here" but is incomplete
        ]
        
        for pattern in truncation_patterns:
            if re.search(pattern, response.strip()):
                issues.append("Response appears truncated")
                score -= 0.4
                break
        
        # Check 4: Complete sentence check (should end with punctuation)
        if response.strip() and not re.search(r"[.!?:)\]]$", response.strip()):
            issues.append("Response doesn't end with proper punctuation")
            score -= 0.1
        
        # Check 5: Markdown completeness
        # Check for unclosed markdown elements
        open_bold = response.count("**") % 2
        open_italic = response.count("*") % 2
        open_code = response.count("`") % 2
        open_bracket = response.count("[") - response.count("]")
        
        if open_bold != 0:
            issues.append("Unclosed bold markdown")
            score -= 0.1
        if open_italic != 0:
            issues.append("Unclosed italic markdown")
            score -= 0.1
        if open_code != 0:
            issues.append("Unclosed code markdown")
            score -= 0.1
        if open_bracket > 0:
            issues.append("Unclosed markdown link")
            score -= 0.1
        
        # Check 6: List completeness
        # If response starts a list, ensure it has items
        if re.search(r"^\s*[-*•]\s+", response, re.MULTILINE):
            # Check if list has at least 2 items or is completed
            list_items = len(re.findall(r"^\s*[-*•]\s+", response, re.MULTILINE))
            if list_items < 2 and not re.search(r"[.!?]", response):
                issues.append("List appears incomplete")
                score -= 0.15
        
        # Final score bounds
        score = max(0.0, min(1.0, score))
        is_valid = score >= 0.6  # Response is valid if score >= 0.6
        
        return {
            "is_valid": is_valid,
            "issues": issues,
            "score": round(score, 2),
        }

    def _template_fallback(self, context: str) -> Dict[str, Any]:
        """Return a template-based response when Groq is unavailable."""
        if context and context != "No relevant information found in the knowledge base.":
            # Extract the first answer from context
            match = re.search(r"\[Source 1.*?\]\n(.+?)(?:\n---|\Z)", context, re.DOTALL)
            if match:
                return self._static(match.group(1).strip(), source="template")
        return self._static(NO_MATCH_RESPONSE, source="template")

    def _static(self, text: str, source: str = "deterministic") -> Dict[str, Any]:
        return {
            "response": text,
            "source": source,
            "model": "none",
            "tokens_used": 0,
        }

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            "groq_available": self.groq_available,
            "groq_model": self.groq_model,
            "total_requests": self.request_count,
            "groq_requests": self.groq_requests,
            "total_tokens_used": self.total_tokens_used,
        }
