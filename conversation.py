"""
Conversational memory subsystem (Tier 1: short-term, session-scoped).

Async-first. The default in-memory backend uses asyncio.Lock for protection
across concurrent sessions on the same loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .config import CRAGConfig
from .llm import LLMClients
from .observability import TelemetryCollector
from .prompts import CONVERSATIONAL_REWRITE_PROMPT
from .schemas import RewrittenQuery

logger = logging.getLogger(__name__)


@dataclass
class Message:
    role: str
    content: str
    turn: int


@dataclass
class Conversation:
    session_id: str
    messages: list[Message] = field(default_factory=list)
    turn_count: int = 0
    last_active_ts: float = field(default_factory=time.time)
    user_id: Optional[str] = None

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
        max_msgs = max_history * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]

    def history_for_rewriter(self, last_n_turns: int) -> list[Message]:
        if not self.messages:
            return []
        return self.messages[-(last_n_turns * 2):]

    def is_first_turn(self) -> bool:
        return self.turn_count <= 1


class ConversationStore(Protocol):
    async def get(self, session_id: str) -> Optional[Conversation]: ...
    async def upsert(self, conversation: Conversation) -> None: ...
    async def delete(self, session_id: str) -> None: ...
    async def evict_stale(self, ttl_seconds: int) -> int: ...


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Conversation] = {}
        self._lock = asyncio.Lock()

    async def get(self, session_id: str) -> Optional[Conversation]:
        async with self._lock:
            return self._sessions.get(session_id)

    async def upsert(self, conversation: Conversation) -> None:
        async with self._lock:
            self._sessions[conversation.session_id] = conversation

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def evict_stale(self, ttl_seconds: int) -> int:
        cutoff = time.time() - ttl_seconds
        async with self._lock:
            stale = [
                sid for sid, conv in self._sessions.items()
                if conv.last_active_ts < cutoff
            ]
            for sid in stale:
                del self._sessions[sid]
        return len(stale)


class ConversationManager:
    _EVICTION_INTERVAL_S = 300

    def __init__(self, cfg: CRAGConfig, store: Optional[ConversationStore] = None):
        self.cfg = cfg
        self.store: ConversationStore = store or InMemoryConversationStore()
        self._last_eviction = time.time()
        self._eviction_lock = asyncio.Lock()

    async def _maybe_evict(self) -> None:
        now = time.time()
        if now - self._last_eviction < self._EVICTION_INTERVAL_S:
            return
        async with self._eviction_lock:
            if now - self._last_eviction < self._EVICTION_INTERVAL_S:
                return
            evicted = await self.store.evict_stale(self.cfg.conversation_session_ttl_s)
            self._last_eviction = now
            if evicted:
                logger.info("[conversation] Evicted %d stale session(s).", evicted)

    async def get_or_create(
        self, session_id: str, user_id: Optional[str] = None
    ) -> Conversation:
        await self._maybe_evict()
        conv = await self.store.get(session_id)
        if conv is None:
            conv = Conversation(session_id=session_id, user_id=user_id)
            await self.store.upsert(conv)
        elif user_id and not conv.user_id:
            conv.user_id = user_id
            await self.store.upsert(conv)
        return conv

    async def record_user_message(self, conv: Conversation, content: str) -> None:
        conv.add_user_message(content, max_history=self.cfg.conversation_history_turns)
        await self.store.upsert(conv)

    async def record_assistant_message(self, conv: Conversation, content: str) -> None:
        conv.add_assistant_message(content, max_history=self.cfg.conversation_history_turns)
        await self.store.upsert(conv)

    async def end_session(self, session_id: str) -> None:
        await self.store.delete(session_id)


class ConversationalQueryRewriter:
    def __init__(self, cfg: CRAGConfig, clients: LLMClients):
        self.cfg = cfg
        self._chain = (
            CONVERSATIONAL_REWRITE_PROMPT
            | clients.fast.with_structured_output(RewrittenQuery)
        )

    @staticmethod
    def _format_history(messages: list[Message]) -> str:
        return "\n".join(f"{m.role}: {m.content}" for m in messages)

    async def rewrite(
        self,
        question: str,
        conversation: Optional[Conversation],
        telemetry: TelemetryCollector,
    ) -> str:
        if not self.cfg.conversation_enabled:
            return question
        if conversation is None or conversation.is_first_turn():
            return question

        history = conversation.history_for_rewriter(self.cfg.conversation_history_turns)
        if not history:
            return question

        async with telemetry.atime_stage("conversational_rewrite"):
            try:
                result = await self._chain.ainvoke({
                    "history": self._format_history(history),
                    "question": question,
                })
                telemetry.record_llm_call("conversational_rewrite")
            except Exception as exc:
                logger.warning("Rewrite failed: %s", exc)
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
            logger.info("[conversation] '%s' → '%s'", question[:60], rewritten[:60])
        return rewritten
