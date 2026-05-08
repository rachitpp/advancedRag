"""
Conversational memory subsystem (Tier 1: short-term, session-scoped).

What it does:
  - Stores per-session conversation history (user + assistant messages).
  - Provides a query rewriter that resolves follow-ups into standalone questions
    using recent history.
  - Manages session lifecycle (TTL eviction).

What it doesn't do:
  - Persist across process restarts (in-memory by default).
  - Cross-session anything — that's user_memory.py.

Storage:
  - Default backend: InMemoryConversationStore (dict + lock + TTL).
  - The ConversationStore Protocol lets you swap in Redis/DynamoDB without
    touching the pipeline. Just implement get/upsert/delete.

Design choices:
  - Session ID is opaque to the pipeline — caller assigns it (e.g., uuid per
    web session, or user_id for single-session-per-user apps).
  - History is bounded: only the last N turns are kept in memory per session,
    AND only the last N are sent to the rewriter. Two separate caps so you
    can store more for audit but rewrite from a smaller window.
  - The rewriter uses structured output (RewrittenQuery) so we get a clean
    boolean signal for whether rewriting was actually needed — useful for
    skipping retrieval-rewrite logging on first turns.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

from langchain_core.documents import Document  # noqa: F401  (typing convenience)

from .config import CRAGConfig
from .llm import LLMClients
from .observability import TelemetryCollector
from .prompts import CONVERSATIONAL_REWRITE_PROMPT
from .schemas import RewrittenQuery

logger = logging.getLogger(__name__)


# ============================================================================
# Data model
# ============================================================================

@dataclass
class Message:
    role: str            # "user" | "assistant"
    content: str
    turn: int            # 1-indexed turn number within the conversation


@dataclass
class Conversation:
    """Session-scoped conversation state. Bounded history."""
    session_id: str
    messages: list[Message] = field(default_factory=list)
    turn_count: int = 0
    last_active_ts: float = field(default_factory=time.time)
    user_id: Optional[str] = None   # optional link to user memory

    # ---- Mutation -----------------------------------------------------

    def add_user_message(self, content: str, max_history: int) -> None:
        self.turn_count += 1
        self.messages.append(Message("user", content, self.turn_count))
        self.last_active_ts = time.time()
        self._truncate(max_history)

    def add_assistant_message(self, content: str, max_history: int) -> None:
        self.messages.append(Message("assistant", content, self.turn_count))
        self.last_active_ts = time.time()
        self._truncate(max_history)

    def _truncate(self, max_history: int) -> None:
        # max_history is in TURNS, each turn = 1 user + 1 assistant message.
        max_msgs = max_history * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]

    # ---- Read ---------------------------------------------------------

    def history_for_rewriter(self, last_n_turns: int) -> list[Message]:
        """Return the last N completed turns as messages."""
        if not self.messages:
            return []
        return self.messages[-(last_n_turns * 2):]

    def is_first_turn(self) -> bool:
        return self.turn_count <= 1


# ============================================================================
# Storage protocol + in-memory backend
# ============================================================================

class ConversationStore(Protocol):
    """Storage interface. Swap with Redis/DynamoDB in production multi-process."""

    def get(self, session_id: str) -> Optional[Conversation]: ...
    def upsert(self, conversation: Conversation) -> None: ...
    def delete(self, session_id: str) -> None: ...
    def evict_stale(self, ttl_seconds: int) -> int: ...


class InMemoryConversationStore:
    """
    Default backend. Threadsafe via a single lock.

    For multi-process deployments swap with a Redis-backed implementation
    that conforms to the same Protocol. The pipeline never touches the store
    directly — it goes through ConversationManager.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Conversation] = {}
        self._lock = threading.Lock()

    def get(self, session_id: str) -> Optional[Conversation]:
        with self._lock:
            return self._sessions.get(session_id)

    def upsert(self, conversation: Conversation) -> None:
        with self._lock:
            self._sessions[conversation.session_id] = conversation

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def evict_stale(self, ttl_seconds: int) -> int:
        cutoff = time.time() - ttl_seconds
        with self._lock:
            stale = [
                sid for sid, conv in self._sessions.items()
                if conv.last_active_ts < cutoff
            ]
            for sid in stale:
                del self._sessions[sid]
        return len(stale)


# ============================================================================
# Manager (used by pipeline)
# ============================================================================

class ConversationManager:
    """
    Owns conversation state. Lives for the pipeline's lifetime.

    Responsibilities:
      - Get-or-create per session_id
      - Append user + assistant messages with bounded history
      - Periodic stale-session eviction (lazy: triggered on get)
    """

    # Eviction is best-effort: we evict on get if enough time has passed.
    # Avoids a background thread for the in-memory case.
    _EVICTION_INTERVAL_S = 300

    def __init__(self, cfg: CRAGConfig, store: Optional[ConversationStore] = None):
        self.cfg = cfg
        self.store = store or InMemoryConversationStore()
        self._last_eviction = time.time()
        self._eviction_lock = threading.Lock()

    def _maybe_evict(self) -> None:
        now = time.time()
        if now - self._last_eviction < self._EVICTION_INTERVAL_S:
            return
        with self._eviction_lock:
            if now - self._last_eviction < self._EVICTION_INTERVAL_S:
                return  # another thread did it
            evicted = self.store.evict_stale(self.cfg.conversation_session_ttl_s)
            self._last_eviction = now
            if evicted:
                logger.info("[conversation] Evicted %d stale session(s).", evicted)

    def get_or_create(
        self, session_id: str, user_id: Optional[str] = None
    ) -> Conversation:
        self._maybe_evict()
        conv = self.store.get(session_id)
        if conv is None:
            conv = Conversation(session_id=session_id, user_id=user_id)
            self.store.upsert(conv)
        elif user_id and not conv.user_id:
            # Backfill user_id if it wasn't set at creation.
            conv.user_id = user_id
            self.store.upsert(conv)
        return conv

    def record_user_message(self, conv: Conversation, content: str) -> None:
        conv.add_user_message(content, max_history=self.cfg.conversation_history_turns)
        self.store.upsert(conv)

    def record_assistant_message(self, conv: Conversation, content: str) -> None:
        conv.add_assistant_message(content, max_history=self.cfg.conversation_history_turns)
        self.store.upsert(conv)

    def end_session(self, session_id: str) -> None:
        self.store.delete(session_id)


# ============================================================================
# Conversational query rewriter
# ============================================================================

class ConversationalQueryRewriter:
    """
    Rewrites follow-up questions into standalone questions using history.

    Skips the LLM call entirely on the first turn (nothing to rewrite from).
    Returns the original question on any failure (graceful degradation).
    """

    def __init__(self, cfg: CRAGConfig, clients: LLMClients):
        self.cfg = cfg
        self._chain = (
            CONVERSATIONAL_REWRITE_PROMPT
            | clients.fast.with_structured_output(RewrittenQuery)
        )

    @staticmethod
    def _format_history(messages: list[Message]) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in messages)

    def rewrite(
        self,
        question: str,
        conversation: Optional[Conversation],
        telemetry: TelemetryCollector,
    ) -> str:
        # Skip if no conversation, first turn, or feature disabled.
        if not self.cfg.conversation_enabled:
            return question
        if conversation is None or conversation.is_first_turn():
            return question

        history = conversation.history_for_rewriter(self.cfg.conversation_history_turns)
        if not history:
            return question

        with telemetry.time_stage("conversational_rewrite"):
            try:
                result = self._chain.invoke({
                    "history": self._format_history(history),
                    "question": question,
                })
                telemetry.record_llm_call("conversational_rewrite")
            except Exception as exc:
                logger.warning("Conversational rewrite failed: %s", exc)
                return question

        rewritten = (result.standalone_question or "").strip()
        if not rewritten:
            return question

        telemetry.stage("conversational_rewrite").notes.update({
            "rewritten": result.needed_rewriting,
            "original": question[: self.cfg.log_truncate_chars],
            "standalone": rewritten[: self.cfg.log_truncate_chars],
        })
        if result.needed_rewriting:
            logger.info(
                "[conversation] Rewrote: '%s' → '%s'",
                question[:60], rewritten[:60],
            )
        return rewritten
