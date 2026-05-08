"""
CRAG pipeline orchestrator.

This module is deliberately thin — all real logic lives in retrieval/, grading/,
verification/, conversation/, and user_memory/. The orchestrator's job is to:
  1. Hold the subsystems together
  2. Implement the graded router
  3. Stream the answer to a caller-supplied callback
  4. Assemble the CRAGResult
  5. Coordinate memory: rewrite query (conversational), inject facts (user),
     record the turn (both), schedule extraction (user, async).

Threading model:
  - Three ThreadPoolExecutors: chunk grading, claim verification, memory extraction.
  - Created at CRAGPipeline construction, shut down via .close() (or context mgr).
  - NOT module-level globals — caller owns the lifecycle.

Memory model:
  - Both tiers are OPTIONAL. Pipeline runs identically without them.
  - Conversational memory: pass session_id to enable per-session history.
  - User memory: pass user_id AND set cfg.user_memory_enabled=True.
  - Either can be None — pipeline degrades gracefully.

Streaming model:
  - run_query takes an optional on_token callback.
  - Defaults to a no-op so HTTP API callers don't need to think about it.
  - CLI passes a stdout-writer adapter.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from langchain_core.documents import Document
from langsmith import traceable

from .config import CRAGConfig
from .conversation import (
    Conversation,
    ConversationManager,
    ConversationStore,
    ConversationalQueryRewriter,
)
from .grading import ChunkGrader, GradeReport
from .llm import LLMClients
from .observability import FallbackReason, TelemetryCollector, extract_usage
from .prompts import ANSWER_PROMPT
from .retrieval import (
    HybridRetriever,
    QueryAnalyzer,
    RetrieverIndex,
    WebSearcher,
    deduplicate_docs,
)
from .user_memory import RetrievedFact, UserMemoryManager
from .verification import FaithfulnessVerifier, SelfCorrector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- Result type

@dataclass
class CRAGResult:
    """Structured pipeline output. Caller-friendly, no LangSmith dependency to read."""
    answer: str
    faithfulness_score: float
    fallback_reason: FallbackReason = FallbackReason.NONE
    used_web_search: bool = False
    self_corrected: bool = False
    correction_regressed: bool = False
    source_docs: list[Document] = field(default_factory=list)

    # ---- Memory metadata (populated when memory is used) ----
    standalone_question: Optional[str] = None     # post-rewrite, if rewritten
    user_facts_used: list[str] = field(default_factory=list)
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def is_fallback(self) -> bool:
        return self.fallback_reason != FallbackReason.NONE


# ---------------------------------------------------------------- Helpers

def format_docs(docs: list[Document]) -> str:
    """Format docs with provenance — used both for generation AND verification."""
    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Chunk {i} | {src}, p.{page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


OnToken = Callable[[str], None]


def _noop_on_token(_token: str) -> None:
    pass


# ---------------------------------------------------------------- Pipeline

class CRAGPipeline:
    """
    Owns the subsystems and exposes run_query.

    Use as a context manager OR call .close() explicitly to shut down thread pools.

    Memory:
      - Pass session_id to enable conversational memory for that session.
      - Pass user_id to enable user memory (also requires cfg.user_memory_enabled).
      - Both default to None — pipeline runs as a stateless QA system.
    """

    def __init__(
        self,
        cfg: CRAGConfig,
        clients: Optional[LLMClients] = None,
        conversation_store: Optional[ConversationStore] = None,
    ):
        self.cfg = cfg
        self.clients = clients or LLMClients.from_config(cfg)

        # ---- Core subsystems ----
        # Retrieval is split: index owns infrastructure, retriever owns algorithm.
        self.index = RetrieverIndex(cfg, self.clients)
        self.query_analyzer = QueryAnalyzer(cfg, self.clients)
        self.retriever = HybridRetriever(cfg, self.index, self.query_analyzer)
        self.web_searcher = WebSearcher(cfg, self.clients)
        self.grader = ChunkGrader(cfg, self.clients)

        # Verification is split: verifier scores, corrector decides.
        self.verifier = FaithfulnessVerifier(cfg, self.clients)
        self.corrector = SelfCorrector(cfg, self.clients, self.verifier)

        # ---- Memory subsystems ----
        self.conversation_manager = ConversationManager(cfg, store=conversation_store)
        self.query_rewriter = ConversationalQueryRewriter(cfg, self.clients)

        # ---- Thread pools (lifecycle owned here) ----
        self._chunk_pool = ThreadPoolExecutor(
            max_workers=cfg.max_chunk_workers, thread_name_prefix="crag-chunk"
        )
        self._claim_pool = ThreadPoolExecutor(
            max_workers=cfg.max_claim_workers, thread_name_prefix="crag-claim"
        )
        # Memory extraction runs on its own small pool. Kept separate from
        # claim/chunk pools because memory work happens AFTER the response is
        # returned and shouldn't compete with in-flight pipeline work.
        self._memory_pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="crag-memory"
        )

        self.user_memory = UserMemoryManager(
            cfg, self.clients, extraction_pool=self._memory_pool
        )

    # ---- Lifecycle -----------------------------------------------------

    def close(self, wait: bool = True) -> None:
        """Shut down thread pools. Call from app shutdown handler or use `with`."""
        self._chunk_pool.shutdown(wait=wait)
        self._claim_pool.shutdown(wait=wait)
        # Always wait for in-flight memory extractions — losing them silently
        # would mean the user "told" the system something it never remembered.
        self._memory_pool.shutdown(wait=True)

    def __enter__(self) -> "CRAGPipeline":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close(wait=True)

    # ---- Routing -------------------------------------------------------

    def _route(self, local_grades: GradeReport) -> tuple[str, bool]:
        """
        Graded router. Returns (mode, used_web).
          "local"  -> use local relevant_docs only
          "web"    -> use web results only
          "hybrid" -> combine local relevant + graded web
        """
        if local_grades.has_strong_local_signal:
            return "local", False
        if local_grades.has_any_relevant:
            return "hybrid", True
        return "web", True

    # ---- Generation ----------------------------------------------------

    def _generate(
        self,
        question: str,
        labelled_context: str,
        user_context_block: str,
        on_token: OnToken,
        telemetry: TelemetryCollector,
    ) -> str:
        """Stream the answer, accumulate, return full string."""
        prompt_value = ANSWER_PROMPT.invoke({
            "context": labelled_context,
            "question": question,
            "user_context_block": user_context_block,
        })

        chunks: list[str] = []
        last_chunk: Any = None

        with telemetry.time_stage("generate"):
            for chunk in self.clients.main.stream(prompt_value):
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    chunks.append(content)
                    on_token(content)
                last_chunk = chunk
            telemetry.record_llm_call("generate")

        ti, to = extract_usage(last_chunk)
        if ti or to:
            telemetry.stage("generate").tokens_in += ti
            telemetry.stage("generate").tokens_out += to

        return "".join(chunks)

    # ---- Public API: run_query ----------------------------------------

    @traceable(name="crag.run_query", tags=["crag"])
    def run_query(
        self,
        question: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        on_token: Optional[OnToken] = None,
    ) -> CRAGResult:
        """
        Run the full CRAG pipeline for a single question.

        Args:
          question:    user query (raw, possibly with pronouns / references)
          session_id:  enables conversational memory if provided
          user_id:     enables user memory if provided AND cfg.user_memory_enabled
          on_token:    streaming callback; defaults to no-op

        Memory flow:
          1. Get-or-create conversation (if session_id given), record user msg
          2. Rewrite question into standalone form using history
          3. Retrieve user facts relevant to the standalone question
          4. ...(normal CRAG pipeline using standalone_question for retrieval)...
          5. Generate using ORIGINAL question + injected user facts
          6. Record assistant message in conversation
          7. Schedule async fact extraction from this turn (if non-fallback)
        """
        on_token = on_token or _noop_on_token
        telemetry = TelemetryCollector()

        # ---- Memory: setup conversation -----------------------------------
        conversation: Optional[Conversation] = None
        if session_id and self.cfg.conversation_enabled:
            conversation = self.conversation_manager.get_or_create(
                session_id, user_id=user_id
            )
            self.conversation_manager.record_user_message(conversation, question)

        # ---- Memory: rewrite follow-ups into standalone questions ---------
        standalone_question = self.query_rewriter.rewrite(
            question, conversation, telemetry
        )
        was_rewritten = standalone_question != question

        # ---- Memory: retrieve relevant user facts -------------------------
        user_facts: list[RetrievedFact] = self.user_memory.retrieve_for_question(
            user_id=user_id,
            question=standalone_question,
            telemetry=telemetry,
        )
        user_context_block = self.user_memory.format_for_prompt(user_facts)

        # ---- 1. Local retrieval ------------------------------------------
        with telemetry.time_stage("retrieve_total"):
            local_docs = self.retriever.retrieve(standalone_question, telemetry)

        # ---- 2. Grade local docs -----------------------------------------
        local_report = self.grader.grade(
            local_docs, standalone_question, self._chunk_pool,
            telemetry, label="local",
        )

        # ---- 3. Route ----------------------------------------------------
        mode, used_web = self._route(local_report)
        logger.info(
            "[router] mode=%s relevant=%d/%d mean_conf=%.2f",
            mode, local_report.relevant_count, local_report.total,
            local_report.mean_confidence,
        )

        # ---- 4. Build final docs based on mode ---------------------------
        final_docs: list[Document]
        if mode == "local":
            final_docs = local_report.relevant_docs

        elif mode == "web":
            web_docs = self.web_searcher.search(standalone_question, telemetry)
            if not web_docs:
                return self._build_fallback_result(
                    FallbackReason.WEB_SEARCH_FAILED, telemetry,
                    used_web=True, conversation=conversation,
                    standalone=standalone_question if was_rewritten else None,
                    session_id=session_id, user_id=user_id,
                )
            web_report = self.grader.grade(
                web_docs, standalone_question, self._chunk_pool,
                telemetry, label="web",
            )
            final_docs = web_report.relevant_docs

        else:  # hybrid
            web_docs = self.web_searcher.search(standalone_question, telemetry)
            if web_docs:
                web_report = self.grader.grade(
                    web_docs, standalone_question, self._chunk_pool,
                    telemetry, label="web",
                )
                final_docs = deduplicate_docs(
                    local_report.relevant_docs + web_report.relevant_docs
                )
            else:
                final_docs = local_report.relevant_docs

        if not final_docs:
            return self._build_fallback_result(
                FallbackReason.EMPTY_RETRIEVAL, telemetry,
                used_web=used_web, conversation=conversation,
                standalone=standalone_question if was_rewritten else None,
                session_id=session_id, user_id=user_id,
            )

        # ---- 5. Generate -------------------------------------------------
        # Use the ORIGINAL question for generation (so the answer matches
        # what the user actually typed). The standalone form was only for
        # retrieval, where keyword density and reference resolution matter.
        labelled_context = format_docs(final_docs)
        answer = self._generate(
            question=question,
            labelled_context=labelled_context,
            user_context_block=user_context_block,
            on_token=on_token,
            telemetry=telemetry,
        )

        # ---- 6. Verify faithfulness --------------------------------------
        verification = self.verifier.verify_answer(
            question, answer, labelled_context, self._claim_pool, telemetry,
        )
        logger.info(
            "[verify] score=%.2f relevance=%s claims=%d/%d",
            verification.score, verification.answer_relevance,
            len(verification.supported), verification.total,
        )

        # ---- 7. Self-correct if needed -----------------------------------
        correction = self.corrector.correct_if_needed(
            question, answer, verification,
            labelled_context, self._claim_pool, telemetry,
        )

        final_answer = correction.final_answer
        is_off_topic = correction.fallback_reason == FallbackReason.OFF_TOPIC

        # ---- 8. Record assistant message in conversation -----------------
        if conversation is not None:
            self.conversation_manager.record_assistant_message(
                conversation, final_answer
            )

        # ---- 9. Schedule async user-memory extraction --------------------
        # Skip extraction on fallbacks — extracting facts from system fallback
        # text would pollute memory with non-user content.
        if correction.fallback_reason == FallbackReason.NONE:
            self.user_memory.schedule_extraction(
                user_id=user_id,
                user_message=question,
                assistant_message=final_answer,
            )

        return CRAGResult(
            answer=final_answer,
            faithfulness_score=correction.final_score,
            fallback_reason=correction.fallback_reason,
            used_web_search=used_web,
            self_corrected=correction.was_corrected,
            correction_regressed=correction.correction_regressed,
            source_docs=final_docs if not is_off_topic else [],
            standalone_question=standalone_question if was_rewritten else None,
            user_facts_used=[f.text for f in user_facts],
            session_id=session_id,
            user_id=user_id,
            telemetry=telemetry.summary(),
        )

    # ---- Fallback helper ----------------------------------------------

    def _build_fallback_result(
        self,
        reason: FallbackReason,
        telemetry: TelemetryCollector,
        used_web: bool,
        conversation: Optional[Conversation] = None,
        standalone: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> CRAGResult:
        logger.warning("[pipeline] Returning fallback: %s", reason.value)
        fallback_msg = self.cfg.hallucination_fallback

        # Even on fallback, record the turn — otherwise the next rewriter sees
        # a hole in conversation history and behaves oddly.
        if conversation is not None:
            self.conversation_manager.record_assistant_message(
                conversation, fallback_msg
            )

        return CRAGResult(
            answer=fallback_msg,
            faithfulness_score=0.0,
            fallback_reason=reason,
            used_web_search=used_web,
            standalone_question=standalone,
            session_id=session_id,
            user_id=user_id,
            telemetry=telemetry.summary(),
        )

    # ---- Memory management API (for callers) --------------------------

    def end_session(self, session_id: str) -> None:
        """Drop conversation history for this session."""
        self.conversation_manager.end_session(session_id)

    def forget_user(self, user_id: str) -> None:
        """Wipe all user memory for the given user (GDPR / 'forget me')."""
        self.user_memory.clear_user(user_id)