"""
──────────────────────────────────────────────────────────────────────────────
 Escalation Service — Determines when to escalate to a human agent

Escalation triggers (any one is sufficient):
   1. Retrieved FAQ document has escalate=True flag
   2. Intent classified as ESCALATION
   3. Session frustration_count >= 2 (repeated complaints)
   4. Legal threat keywords detected in query
   5. Fraud / security keywords detected in query
   6. Severity modifier: critical priority doc
──────────────────────────────────────────────────────────────────────────────
"""

import re
from typing import Any, Dict, List, Optional


# ── Keyword Patterns ──────────────────────────────────────────────────────────

LEGAL_THREAT_PATTERN = re.compile(
    r"\b(sue|lawyer|court|legal\s*action|consumer\s*forum|file\s*a\s*complaint|"
    r"file\s*complaint|going\s*to\s*court|solicitor|attorney)\b",
    re.IGNORECASE,
)

FRAUD_PATTERN = re.compile(
    r"\b(fraud|hacked|unauthorized|stolen|identity\s*theft|"
    r"someone\s*else.*account|account.*compromised|suspicious\s*(transaction|activity)|"
    r"fake\s*transaction|not\s*me.*purchase)\b",
    re.IGNORECASE,
)

SEVERE_FRUSTRATION_PATTERN = re.compile(
    r"\b(this\s*is\s*(absolutely|completely|totally)\s*(unacceptable|terrible|awful|ridiculous)|"
    r"worst\s*(service|experience|support)\s*(ever|i|i've)|"
    r"you\s*are\s*(scammers?|thieves?|liars?)|never\s*shopping\s*here\s*again)\b",
    re.IGNORECASE,
)


class EscalationService:
    """
    Decides whether a conversation should be escalated to a human agent
    and determines the urgency level.
    """

    def check(
        self,
        query: str,
        intent: Dict[str, Any],
        retrieval_results: List[Dict],
        session_frustration_count: int,
    ) -> Dict[str, Any]:
        """
        Evaluate current-message escalation signals and return a decision.

        Returns:
            {
                "should_escalate": bool,
                "reason": str,
                "urgency": "critical" | "high" | "medium" | "none",
                "trigger": str,
            }
        """
        # ── Fraud / security keywords in current message ─────────────────
        if FRAUD_PATTERN.search(query):
            return self._escalate(
                reason="Fraud or account security issue detected.",
                urgency="critical",
                trigger="fraud_keyword",
            )

        # ── Legal threat keywords in current message ──────────────────────
        if LEGAL_THREAT_PATTERN.search(query):
            return self._escalate(
                reason="Customer made a legal threat.",
                urgency="critical",
                trigger="legal_threat",
            )

        # ── Severe frustration in current message ─────────────────────────
        if SEVERE_FRUSTRATION_PATTERN.search(query):
            return self._escalate(
                reason="Customer expressed severe frustration.",
                urgency="high",
                trigger="severe_frustration_keyword",
            )

        # ── Knowledge base escalation flag ────────────────────────────────
        if retrieval_results:
            top = retrieval_results[0]
            if top.get("escalate", False):
                reason = top.get("escalation_reason", "Matched escalation document.")
                reason_lower = reason.lower()
                urgency = (
                    "critical"
                    if any(term in reason_lower for term in ("fraud", "security", "legal", "threat"))
                    else "high"
                )
                return self._escalate(
                    reason=reason,
                    urgency=urgency,
                    trigger="doc_flag",
                )

        # ── Intent-based escalation ───────────────────────────────────────
        if intent.get("intent") == "ESCALATION":
            trigger = intent.get("meta", {}).get("trigger", "explicit_request")
            urgency = "critical" if trigger == "legal_threat" else "high"
            return self._escalate(
                reason=intent.get("meta", {}).get("reason", "Customer requested escalation."),
                urgency=urgency,
                trigger=trigger,
            )

        # ── Repeated frustration ending with the current message ──────────
        if intent.get("intent") == "FRUSTRATION" and session_frustration_count >= 1:
            return self._escalate(
                reason=f"Customer has expressed frustration {session_frustration_count + 1} times in this session.",
                urgency="high",
                trigger="frustration_count",
            )

        # ── No escalation needed ──────────────────────────────────────────
        return {
            "should_escalate": False,
            "reason": "",
            "urgency": "none",
            "trigger": "",
        }

    def _escalate(self, reason: str, urgency: str, trigger: str) -> Dict[str, Any]:
        return {
            "should_escalate": True,
            "reason": reason,
            "urgency": urgency,
            "trigger": trigger,
        }

    def get_escalation_response(self, urgency: str, reason: str) -> str:
        """Return an appropriate empathetic escalation message based on urgency."""
        if urgency == "critical":
            return (
                "⚠️ I understand this is an urgent matter. I'm connecting you with a senior "
                "support specialist right away. Please hold — someone will assist you shortly. "
                "Your case reference has been flagged as **high priority**."
            )
        elif urgency == "high":
            return (
                "I'm sorry for the trouble you've experienced. I'm escalating this to our "
                "customer support team who can resolve this for you directly. "
                "A support executive will reach out within **24 hours**. "
                "You may also reach us at **support@shopease.com**."
            )
        else:
            return (
                "I understand your concern, and I want to make sure you get the right help. "
                "Let me connect you with one of our support executives who can assist you further."
            )
