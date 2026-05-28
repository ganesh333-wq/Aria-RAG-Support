"""
──────────────────────────────────────────────────────────────────────────────
 Session Memory — Custom per-session memory management

Two-layer memory:
   1. SimpleConversationMemory — stores recent chat for LLM context
   2. SessionData — tracks metadata: category, frustration, escalation history

 Per-session state:
   - conversation history (for LLM prompts)
   - last retrieved category (for follow-up detection)
   - frustration count (triggers auto-escalation at >= 2)
   - escalation history for analytics only
   - sentiment trend
   - turn count

 Session timeout: configurable (default 30 minutes)
──────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── Session Data ──────────────────────────────────────────────────────────────

class SimpleConversationMemory:
    """Small replacement for LangChain's removed ConversationBufferMemory API."""

    def __init__(
        self,
        memory_key: str = "history",
        human_prefix: str = "Customer",
        ai_prefix: str = "Aria",
        max_turns: int = 10,
    ):
        self.memory_key = memory_key
        self.human_prefix = human_prefix
        self.ai_prefix = ai_prefix
        self.max_turns = max_turns
        self._turns: List[Dict[str, str]] = []

    def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
        self._turns.append({
            "input": inputs.get("input", ""),
            "output": outputs.get("output", ""),
        })
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]

    def load_memory_variables(self, _: Dict) -> Dict[str, str]:
        history = []
        for turn in self._turns:
            history.append(f"{self.human_prefix}: {turn['input']}")
            history.append(f"{self.ai_prefix}: {turn['output']}")
        return {self.memory_key: "\n".join(history)}

@dataclass
class SessionData:
    """Custom metadata for a single conversation session."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    turn_count: int = 0
    last_category: Optional[str] = None
    last_intent: Optional[str] = None
    frustration_count: int = 0
    sentiment_trend: List[float] = field(default_factory=list)
    escalation_history: List[Dict] = field(default_factory=list)
    turn_history: List[Dict] = field(default_factory=list)   # lightweight history

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            # Escalation is evaluated per message. This field is kept false for
            # response-schema compatibility and must not drive control flow.
            "is_escalated": False,
            "last_category": self.last_category,
            "last_intent": self.last_intent,
            "frustration_count": self.frustration_count,
            "avg_sentiment": self._avg_sentiment(),
            "sentiment_trend": self.sentiment_trend[-5:],
        }

    def _avg_sentiment(self) -> float:
        if not self.sentiment_trend:
            return 0.0
        return round(sum(self.sentiment_trend) / len(self.sentiment_trend), 2)


# ── Session Memory Manager ────────────────────────────────────────────────────

class SessionMemoryManager:
    """
    Manages per-session conversation memory.

    Each session has:
      - SimpleConversationMemory (for LLM prompt context)
      - SessionData (for metadata: category, frustration, etc.)
    """

    MAX_HISTORY_TURNS = 10
    SESSION_TIMEOUT = int(os.getenv("SESSION_TIMEOUT_SECONDS", 1800))  # 30 min

    def __init__(self):
        # session_id → {"memory": SimpleConversationMemory, "data": SessionData}
        self._sessions: Dict[str, Dict] = {}

    # ── Session CRUD ──────────────────────────────────────────────────────────

    def get_or_create(self, session_id: str) -> Dict:
        """Get existing session or create a new one."""
        # Clean up expired sessions periodically
        self._maybe_cleanup()

        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "memory": SimpleConversationMemory(
                    memory_key="history",
                    human_prefix="Customer",
                    ai_prefix="Aria",
                    max_turns=self.MAX_HISTORY_TURNS,
                ),
                "data": SessionData(session_id=session_id),
            }
        else:
            session = self._sessions[session_id]
            # Check for timeout
            if time.time() - session["data"].last_activity > self.SESSION_TIMEOUT:
                print(f"[Memory] Session {session_id} expired, resetting.")
                self._sessions[session_id] = {
                    "memory": SimpleConversationMemory(
                        memory_key="history",
                        human_prefix="Customer",
                        ai_prefix="Aria",
                        max_turns=self.MAX_HISTORY_TURNS,
                    ),
                    "data": SessionData(session_id=session_id),
                }

        return self._sessions[session_id]

    def add_turn(
        self,
        session_id: str,
        user_message: str,
        bot_response: str,
        intent: Dict,
        retrieval_results: List[Dict],
    ) -> None:
        """
        Record a completed conversation turn.
        Updates both conversation memory and custom session metadata.
        """
        session = self.get_or_create(session_id)
        data: SessionData = session["data"]
        memory: SimpleConversationMemory = session["memory"]

        # Update conversation memory
        # Strip markdown from bot response for cleaner context
        clean_response = re.sub(r"[*#_`|]", "", bot_response)[:400]
        memory.save_context(
            {"input": user_message},
            {"output": clean_response},
        )

        # Update custom session metadata
        data.turn_count += 1
        data.last_activity = time.time()

        # Track frustration
        if intent.get("intent") == "FRUSTRATION" or (
            intent.get("intent") == "ESCALATION"
            and intent.get("sentiment") == "negative"
        ):
            data.frustration_count += 1
        if intent.get("intent") == "ESCALATION":
            data.escalation_history.append({
                "turn": data.turn_count,
                "reason": intent.get("meta", {}).get("reason", ""),
                "timestamp": time.time(),
            })

        # Track sentiment trend
        sentiment_score = intent.get("sentiment_score", 0.0)
        data.sentiment_trend.append(sentiment_score)
        if len(data.sentiment_trend) > self.MAX_HISTORY_TURNS:
            data.sentiment_trend = data.sentiment_trend[-self.MAX_HISTORY_TURNS:]

        # Track last category from retrieval
        if retrieval_results and retrieval_results[0].get("category"):
            data.last_category = retrieval_results[0]["category"]
        if retrieval_results and retrieval_results[0].get("intent"):
            data.last_intent = retrieval_results[0]["intent"]
        elif intent.get("intent") not in {"FAQ_QUERY", "FRUSTRATION", "NO_MATCH"}:
            data.last_intent = intent.get("intent")

        # Keep lightweight turn history
        data.turn_history.append({
            "index": data.turn_count,
            "user": user_message,
            "intent": intent.get("intent"),
            "retrieved_intent": retrieval_results[0].get("intent") if retrieval_results else None,
            "category": retrieval_results[0].get("category") if retrieval_results else None,
        })
        if len(data.turn_history) > self.MAX_HISTORY_TURNS:
            data.turn_history = data.turn_history[-self.MAX_HISTORY_TURNS:]

    def get_conversation_context(self, session_id: str) -> Optional[str]:
        """
        Get formatted conversation history string for LLM prompt injection.
        Returns None if no history exists.
        """
        session = self.get_or_create(session_id)
        memory: SimpleConversationMemory = session["memory"]

        try:
            history = memory.load_memory_variables({}).get("history", "")
            return history if history.strip() else None
        except Exception:
            return None

    def detect_follow_up(self, session_id: str, current_tokens: List[str]) -> Dict[str, Any]:
        """
        Detect if the current message is a follow-up to the previous topic.

        Signals:
          - Follow-up phrases: "what about", "and", "also", "how about"
          - Short query with pronoun references: "it", "that", "this"
          - Very short question (1–3 tokens) after an FAQ answer
        """
        session = self.get_or_create(session_id)
        data: SessionData = session["data"]

        if not data.turn_history:
            return {"is_follow_up": False}

        last_turn = data.turn_history[-1]
        current_text = " ".join(current_tokens).lower()

        follow_up_phrases = [
            "also", "and", "what about", "another", "one more",
            "btw", "additionally", "plus", "how about", "what if",
            "but what", "but how", "and what", "and how",
            "international orders", "international order", "outside country",
            "overseas orders", "abroad orders",
        ]
        has_follow_up_phrase = any(p in current_text for p in follow_up_phrases)

        is_short = len(current_tokens) <= 5

        pronouns = {"it", "that", "this", "they", "them", "those", "its", "their"}
        has_pronouns = bool(set(current_tokens) & pronouns)

        is_very_short_question = (
            len(current_tokens) <= 3
            and current_tokens
            and current_tokens[0].lower() in {
                "how", "when", "why", "does", "can", "what", "where",
                "who", "which", "will", "would", "is", "are",
            }
        )

        is_follow_up = (
            has_follow_up_phrase
            or (is_short and has_pronouns)
            or is_very_short_question
        )

        return {
            "is_follow_up": is_follow_up,
            "previous_category": last_turn.get("category"),
            "previous_query": last_turn.get("user"),
            "previous_intent": last_turn.get("retrieved_intent") or data.last_intent or last_turn.get("intent"),
            "signals": {
                "has_follow_up_phrase": has_follow_up_phrase,
                "is_short": is_short,
                "has_pronouns": has_pronouns,
                "is_very_short_question": is_very_short_question,
            },
        }

    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get session metadata for API response / debugging."""
        session = self.get_or_create(session_id)
        return session["data"].to_dict()

    def get_active_session_count(self) -> int:
        return len(self._sessions)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def _maybe_cleanup(self) -> None:
        """Remove expired sessions. Called periodically."""
        if len(self._sessions) < 100:
            return  # Don't bother for small session counts
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s["data"].last_activity > self.SESSION_TIMEOUT
        ]
        for sid in expired:
            del self._sessions[sid]
        if expired:
            print(f"[Memory] Cleaned up {len(expired)} expired session(s)")

    def cleanup_all(self) -> int:
        """Remove all expired sessions. Call on shutdown."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if now - s["data"].last_activity > self.SESSION_TIMEOUT
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)
