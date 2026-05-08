"""
User memory subsystem (Tier 2: long-term, cross-session).

What it does:
  - Extracts durable facts about the user from each conversation turn.
  - Stores them in a per-user namespace in a vector store.
  - Retrieves relevant facts at query time, semantically matched to the question.
  - Injects retrieved facts into the answer prompt as a separate context block.

What it doesn't do:
  - Replace conversational memory — they're complementary. Conversational handles
    "it/that" within a session; user memory handles "I'm allergic to X" across sessions.
  - Auto-decay or fact contradiction resolution. (Future work — see __doc__ of
    UserMemoryStore for the design hooks already in place.)

Why a separate Chroma collection per user:
  - Strong isolation: a similarity search for user A NEVER returns user B's facts.
    Filtering by metadata works but is fragile — separate collections is safer.
  - Trades per-user index overhead for security. At scale (>10k users) you'd
    move to a shared collection with strict metadata filtering + row-level
    security at the DB layer. Not yet.

Async extraction:
  - Extraction runs AFTER the response is sent (background thread).
  - The user gets their answer with no added latency.
  - If extraction fails, we log and move on — never block the response.
  - Caller passes a single shared ThreadPoolExecutor (lifecycle managed by pipeline).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import CRAGConfig
from .llm import LLMClients
from .observability import TelemetryCollector
from .prompts import USER_FACT_EXTRACTOR_PROMPT, USER_MEMORY_CONTEXT_HEADER
from .schemas import ExtractedUserFacts

logger = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

def _safe_user_dir(user_id: str) -> str:
    """
    Sanitize user_id for filesystem use.
    SHA-256 prefix avoids exposing real user IDs in directory names.
    """
    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:16]
    return f"user_{digest}"


# ============================================================================
# Store
# ============================================================================

@dataclass
class RetrievedFact:
    text: str
    score: float    # similarity score, lower = more similar (Chroma default)


class UserMemoryStore:
    """
    Per-user vector store of durable facts.

    Stores Chroma collections under {cfg.user_memory_dir}/user_<hash>/. Each
    write is one fact = one Document. Reads are top-K similarity search
    against the incoming question.

    Future hooks (not implemented yet):
      - Fact decay: timestamp metadata is stored, so a future job can prune
        facts older than N months.
      - Contradiction resolution: when a new fact contradicts an old one
        (e.g., "I work in finance" vs "I just changed jobs to healthcare"),
        we'd use an LLM to merge or replace. For now we just append.
    """

    def __init__(self, cfg: CRAGConfig, clients: LLMClients):
        self.cfg = cfg
        self.clients = clients
        self._stores: dict[str, Chroma] = {}     # user_id -> Chroma
        self._lock = threading.Lock()

    def _get_store(self, user_id: str) -> Chroma:
        with self._lock:
            if user_id in self._stores:
                return self._stores[user_id]
            persist_dir = os.path.join(
                self.cfg.user_memory_dir, _safe_user_dir(user_id)
            )
            os.makedirs(persist_dir, exist_ok=True)
            store = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.clients.embeddings,
            )
            self._stores[user_id] = store
            return store

    # ---- Write --------------------------------------------------------

    def add_facts(self, user_id: str, facts: list[str]) -> int:
        """Append facts as new documents. Returns count actually added."""
        if not facts:
            return 0
        try:
            store = self._get_store(user_id)
            import time as _time  # local import — keep store module import light
            now = _time.time()
            store.add_texts(
                texts=facts,
                metadatas=[{"timestamp": now, "user_id": user_id} for _ in facts],
            )
            return len(facts)
        except Exception as exc:
            logger.warning(
                "Failed to add %d facts for user %s: %s", len(facts), user_id, exc
            )
            return 0

    # ---- Read ---------------------------------------------------------

    def retrieve(
        self, user_id: str, question: str, top_k: int, min_relevance: float
    ) -> list[RetrievedFact]:
        """
        Return facts relevant to the question.
        Chroma returns distance (lower = more similar); we convert to a
        normalized relevance score and threshold on it.
        """
        try:
            store = self._get_store(user_id)
            results = store.similarity_search_with_score(question, k=top_k)
        except Exception as exc:
            logger.warning("User memory retrieval failed for %s: %s", user_id, exc)
            return []

        out: list[RetrievedFact] = []
        for doc, distance in results:
            # Chroma returns L2 distance for default embeddings. We convert
            # to a [0,1] relevance score with a soft transform — exact mapping
            # depends on embedding model, so the threshold is a tunable knob.
            relevance = 1.0 / (1.0 + float(distance))
            if relevance >= min_relevance:
                out.append(RetrievedFact(text=doc.page_content, score=relevance))
        return out


# ============================================================================
# Extractor
# ============================================================================

class UserFactExtractor:
    """LLM-driven extraction of durable facts from a single conversation turn."""

    def __init__(self, cfg: CRAGConfig, clients: LLMClients):
        self.cfg = cfg
        self._chain = (
            USER_FACT_EXTRACTOR_PROMPT
            | clients.fast.with_structured_output(ExtractedUserFacts)
        )

    def extract(self, user_message: str, assistant_message: str) -> list[str]:
        """Returns durable facts. Empty list if nothing memory-worthy."""
        try:
            result = self._chain.invoke({
                "user_message": user_message,
                "assistant_message": assistant_message,
            })
        except Exception as exc:
            logger.warning("User fact extraction failed: %s", exc)
            return []

        # Sanity filtering — drop trivially short facts and questions.
        cleaned: list[str] = []
        for fact in result.facts:
            stripped = fact.strip().rstrip(".")
            if not stripped:
                continue
            if len(stripped.split()) < 3:
                continue
            if stripped.endswith("?"):
                continue   # extractor occasionally returns user questions
            cleaned.append(stripped)
        return cleaned


# ============================================================================
# Manager (composition root for user memory)
# ============================================================================

class UserMemoryManager:
    """
    Coordinates extraction + storage + retrieval. Owned by the pipeline.

    The pipeline calls:
      - retrieve_for_question(...) BEFORE generation, to inject facts
      - schedule_extraction(...)   AFTER generation, to update memory

    Both are no-ops if user_memory_enabled=False or user_id is None.
    """

    def __init__(
        self,
        cfg: CRAGConfig,
        clients: LLMClients,
        extraction_pool: ThreadPoolExecutor,
    ):
        self.cfg = cfg
        self.store = UserMemoryStore(cfg, clients)
        self.extractor = UserFactExtractor(cfg, clients)
        self._extraction_pool = extraction_pool

    @property
    def enabled(self) -> bool:
        return self.cfg.user_memory_enabled

    # ---- Read path ----------------------------------------------------

    def retrieve_for_question(
        self,
        user_id: Optional[str],
        question: str,
        telemetry: TelemetryCollector,
    ) -> list[RetrievedFact]:
        if not self.enabled or not user_id:
            return []

        with telemetry.time_stage("user_memory_retrieve"):
            facts = self.store.retrieve(
                user_id=user_id,
                question=question,
                top_k=self.cfg.user_memory_top_k,
                min_relevance=self.cfg.user_memory_min_relevance,
            )

        telemetry.stage("user_memory_retrieve").notes["count"] = len(facts)
        if facts:
            logger.info("[user_memory] Retrieved %d fact(s) for user.", len(facts))
        return facts

    # ---- Write path (async) -------------------------------------------

    def schedule_extraction(
        self,
        user_id: Optional[str],
        user_message: str,
        assistant_message: str,
    ) -> None:
        """
        Fire-and-forget extraction. Runs in a background thread so the user's
        response isn't delayed. Errors are logged but never raised.
        """
        if not self.enabled or not user_id:
            return
        if not user_message or not assistant_message:
            return

        def _run() -> None:
            try:
                facts = self.extractor.extract(user_message, assistant_message)
                if facts:
                    added = self.store.add_facts(user_id, facts)
                    if added:
                        logger.info(
                            "[user_memory] Stored %d new fact(s) for user.", added
                        )
            except Exception as exc:
                # Defensive — extractor + store already swallow their errors,
                # but a process-wide background failure should never propagate.
                logger.warning("[user_memory] Background extraction failed: %s", exc)

        try:
            self._extraction_pool.submit(_run)
        except RuntimeError:
            # Pool is shut down — happens during graceful shutdown. Skip silently.
            logger.debug("[user_memory] Skipped extraction; pool shut down.")

    # ---- Context formatting ------------------------------------------

    @staticmethod
    def format_for_prompt(facts: list[RetrievedFact]) -> str:
        """Render retrieved facts as the user_context_block for the answer prompt."""
        if not facts:
            return ""
        bullets = "\n".join(f"- {f.text}" for f in facts)
        return f"{USER_MEMORY_CONTEXT_HEADER}\n{bullets}\n"

    # ---- Optional management ops -------------------------------------

    def clear_user(self, user_id: str) -> None:
        """For 'forget me' / GDPR-style user-initiated wipes."""
        try:
            store = self.store._get_store(user_id)
            # Chroma doesn't expose collection-drop neatly; delete by metadata.
            store.delete(where={"user_id": user_id})
            logger.info("[user_memory] Cleared all facts for user.")
        except Exception as exc:
            logger.warning("[user_memory] Failed to clear user: %s", exc)
