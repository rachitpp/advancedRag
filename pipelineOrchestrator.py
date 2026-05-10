"""
CRAG pipeline orchestrator. Async-first.

Responsibilities:
  1. Hold subsystems together.
  2. Implement the graded router.
  3. Stream the answer via async generator OR via on_token callback.
  4. Coordinate two-tier memory.
  5. Assemble CRAGResult with telemetry.

Concurrency model: bounded async via asyncio.Semaphore for grading and
claim verification; background fact extraction via tracked task set
drained on close().
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional, Union

from langchain_core.documents import Document
from langsmith import traceable

from .cache import RetrievalCache
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
    QdrantRetrieverIndex,
    QueryAnalyzer,
    WebSearcher,
    deduplicate_docs,
)
from .user_memory import (
    PostgresUserMemoryStore,
    RetrievedFact,
    UserMemoryManager,
)
from .verification import FaithfulnessVerifier, SelfCorrector

logger = logging.getLogger(__name__)


# ============================================================================
# Result types
# ============================================================================

@dataclass
class CRAGResult:
    answer: str
    faithfulness_score: float
    fallback_reason: FallbackReason = FallbackReason.NONE
    used_web_search: bool = False
    self_corrected: bool = False
    correction_regressed: bool = False
    source_docs: list[Document] = field(default_factory=list)
    standalone_question: Optional[str] = None
    user_facts_used: list[str] = field(default_factory=list)
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    telemetry: dict[str, Any] = field(default_factory=dict)

    @property
    def is_fallback(self) -> bool:
        return self.fallback_reason != FallbackReason.NONE


def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Chunk {i} | {src}, p.{page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


OnToken = Callable[[str], Union[None, Awaitable[None]]]


async def _maybe_await(value: Union[None, Awaitable[None]]) -> None:
    if asyncio.iscoroutine(value) or isinstance(value, Awaitable):
        await value  # type: ignore[arg-type]


def _noop_on_token(_token: str) -> None:
    return None


# ============================================================================
# Pipeline
# ============================================================================

class CRAGPipeline:
    """
    Owns subsystems, exposes async run_query and astream_query.

    Use as an async context manager:
        async with CRAGPipeline(cfg) as pipeline:
            result = await pipeline.run_query("...")

    Streaming:
        async for chunk in pipeline.astream_query("..."):
            if isinstance(chunk, str): print(chunk, end="")
            else: result = chunk  # final CRAGResult
    """

    def __init__(
        self,
        cfg: CRAGConfig,
        clients: Optional[LLMClients] = None,
        conversation_store: Optional[ConversationStore] = None,
        user_memory_store: Optional[PostgresUserMemoryStore] = None,
    ):
        self.cfg = cfg
        self.clients = clients or LLMClients.from_config(cfg)

        # Cache + retrieval infra
        self.cache = RetrievalCache.from_config(cfg)
        self.index = QdrantRetrieverIndex(cfg, self.clients)
        self.query_analyzer = QueryAnalyzer(cfg, self.clients)
        self.retriever = HybridRetriever(cfg, self.index, self.query_analyzer, self.cache)
        self.web_searcher = WebSearcher(cfg, self.clients)
        self.grader = ChunkGrader(cfg, self.clients)

        # Verification
        self.verifier = FaithfulnessVerifier(cfg, self.clients)
        self.corrector = SelfCorrector(cfg, self.clients, self.verifier)

        # Conversation memory
        self.conversation_manager = ConversationManager(cfg, store=conversation_store)
        self.query_rewriter = ConversationalQueryRewriter(cfg, self.clients)

        # User memory (Postgres)
        self.user_memory_store = user_memory_store or PostgresUserMemoryStore(
            cfg, self.clients
        )
        self._background_tasks: set[asyncio.Task] = set()
        self.user_memory = UserMemoryManager(
            cfg, self.clients, self.user_memory_store, self._background_tasks
        )

        # Concurrency primitives
        self._chunk_sem = asyncio.Semaphore(cfg.max_chunk_concurrency)
        self._claim_sem = asyncio.Semaphore(cfg.max_claim_concurrency)

    # ---- Lifecycle ----

    async def aclose(self) -> None:
        # Drain background fact-extraction so user-said-X is persisted.
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        await self.cache.aclose()
        await self.index.aclose()
        await self.user_memory_store.aclose()
        await self.clients.aclose()

    async def __aenter__(self) -> "CRAGPipeline":
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.aclose()

    # ---- Routing ----

    def _route(self, local_grades: GradeReport) -> tuple[str, bool]:
        if local_grades.has_strong_local_signal(self.cfg):
            return "local", False
        if local_grades.has_any_relevant:
            return "hybrid", True
        return "web", True

    # ---- Generation ----

    async def _generate_stream(
        self,
        question: str,
        labelled_context: str,
        user_context_block: str,
        telemetry: TelemetryCollector,
    ) -> AsyncIterator[str]:
        prompt_value = ANSWER_PROMPT.invoke({
            "context": labelled_context,
            "question": question,
            "user_context_block": user_context_block,
        })
        last_chunk: Any = None
        tokens_in = tokens_out = 0
        async with telemetry.atime_stage("generate"):
            async for chunk in self.clients.main.astream(prompt_value):
                content = getattr(chunk, "content", None)
                if content:
                    yield content
                last_chunk = chunk
            telemetry.record_llm_call("generate")
        ti, to = extract_usage(last_chunk)
        if ti or to:
            telemetry.stage("generate").tokens_in += ti
            telemetry.stage("generate").tokens_out += to
        _ = (tokens_in, tokens_out)  # silence unused

    # ---- Public API: streaming generator ----

    @traceable(name="crag.astream_query", tags=["crag"])
    async def astream_query(
        self,
        question: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> AsyncIterator[Union[str, CRAGResult]]:
        """
        Stream tokens as they're generated, then yield the final CRAGResult.

        Note on streaming + verification: tokens are emitted during generation;
        verification and self-correction happen after streaming completes. If
        the answer fails verification, the FINAL yielded CRAGResult contains
        the corrected (or fallback) answer, while the streamed text was the
        original. Consumers should display the CRAGResult.answer as the
        authoritative value once it arrives.
        """
        telemetry = TelemetryCollector()

        # ---- Memory setup
        conversation: Optional[Conversation] = None
        if session_id and self.cfg.conversation_enabled:
            conversation = await self.conversation_manager.get_or_create(
                session_id, user_id=user_id
            )
            await self.conversation_manager.record_user_message(conversation, question)

        standalone_question = await self.query_rewriter.rewrite(
            question, conversation, telemetry
        )
        was_rewritten = standalone_question != question

        user_facts: list[RetrievedFact] = await self.user_memory.retrieve_for_question(
            user_id=user_id,
            question=standalone_question,
            telemetry=telemetry,
            tenant_id=tenant_id,
        )
        user_context_block = self.user_memory.format_for_prompt(user_facts)

        # ---- Retrieve
        async with telemetry.atime_stage("retrieve_total"):
            local_docs = await self.retriever.retrieve(
                standalone_question, telemetry,
                filters=filters, tenant_id=tenant_id,
            )

        # ---- Grade
        local_report = await self.grader.grade(
            local_docs, standalone_question,
            self._chunk_sem, telemetry, label="local",
        )

        mode, used_web = self._route(local_report)
        logger.info(
            "[router] mode=%s relevant=%d/%d mean_conf=%.2f",
            mode, local_report.relevant_count, local_report.total,
            local_report.mean_confidence,
        )

        # ---- Build final docs
        if mode == "local":
            final_docs: list[Document] = local_report.relevant_docs
        elif mode == "web":
            web_docs = await self.web_searcher.search(standalone_question, telemetry)
            if not web_docs:
                result = self._build_fallback_result(
                    FallbackReason.WEB_SEARCH_FAILED, telemetry,
                    used_web=True, conversation=conversation,
                    standalone=standalone_question if was_rewritten else None,
                    session_id=session_id, user_id=user_id, tenant_id=tenant_id,
                )
                if conversation is not None:
                    await self.conversation_manager.record_assistant_message(
                        conversation, result.answer
                    )
                yield result
                return
            web_report = await self.grader.grade(
                web_docs, standalone_question,
                self._chunk_sem, telemetry, label="web",
            )
            final_docs = web_report.relevant_docs
        else:
            web_docs = await self.web_searcher.search(standalone_question, telemetry)
            if web_docs:
                web_report = await self.grader.grade(
                    web_docs, standalone_question,
                    self._chunk_sem, telemetry, label="web",
                )
                final_docs = deduplicate_docs(
                    local_report.relevant_docs + web_report.relevant_docs
                )
            else:
                final_docs = local_report.relevant_docs

        if not final_docs:
            result = self._build_fallback_result(
                FallbackReason.EMPTY_RETRIEVAL, telemetry,
                used_web=used_web, conversation=conversation,
                standalone=standalone_question if was_rewritten else None,
                session_id=session_id, user_id=user_id, tenant_id=tenant_id,
            )
            if conversation is not None:
                await self.conversation_manager.record_assistant_message(
                    conversation, result.answer
                )
            yield result
            return

        # ---- Generate (stream tokens)
        labelled_context = format_docs(final_docs)
        chunks: list[str] = []
        async for tok in self._generate_stream(
            question, labelled_context, user_context_block, telemetry
        ):
            chunks.append(tok)
            yield tok
        original_answer = "".join(chunks)

        # ---- Verify
        verification = await self.verifier.verify_answer(
            question, original_answer, labelled_context,
            self._claim_sem, telemetry,
        )
        logger.info(
            "[verify] score=%.2f relevance=%s claims=%d/%d",
            verification.score, verification.answer_relevance,
            len(verification.supported), verification.total,
        )

        # ---- Self-correct
        correction = await self.corrector.correct_if_needed(
            question, original_answer, verification, labelled_context,
            self._claim_sem, telemetry,
        )

        final_answer = correction.final_answer
        is_off_topic = correction.fallback_reason == FallbackReason.OFF_TOPIC

        # ---- Record assistant message
        if conversation is not None:
            await self.conversation_manager.record_assistant_message(
                conversation, final_answer
            )

        # ---- Schedule async fact extraction
        if correction.fallback_reason == FallbackReason.NONE:
            self.user_memory.schedule_extraction(
                user_id=user_id,
                user_message=question,
                assistant_message=final_answer,
                tenant_id=tenant_id,
            )

        yield CRAGResult(
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
            tenant_id=tenant_id,
            telemetry=telemetry.summary(),
        )

    # ---- Public API: callback-based ----

    @traceable(name="crag.run_query", tags=["crag"])
    async def run_query(
        self,
        question: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        on_token: Optional[OnToken] = None,
    ) -> CRAGResult:
        """Async, callback-based variant. Returns the final CRAGResult."""
        callback = on_token or _noop_on_token
        result: Optional[CRAGResult] = None
        async for chunk in self.astream_query(
            question,
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            filters=filters,
        ):
            if isinstance(chunk, CRAGResult):
                result = chunk
            else:
                await _maybe_await(callback(chunk))
        assert result is not None
        return result

    # ---- Fallback helper ----

    def _build_fallback_result(
        self,
        reason: FallbackReason,
        telemetry: TelemetryCollector,
        used_web: bool,
        conversation: Optional[Conversation] = None,
        standalone: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> CRAGResult:
        logger.warning("[pipeline] Fallback: %s", reason.value)
        return CRAGResult(
            answer=self.cfg.hallucination_fallback,
            faithfulness_score=0.0,
            fallback_reason=reason,
            used_web_search=used_web,
            standalone_question=standalone,
            session_id=session_id,
            user_id=user_id,
            tenant_id=tenant_id,
            telemetry=telemetry.summary(),
        )

    # ---- Memory management API ----

    async def end_session(self, session_id: str) -> None:
        await self.conversation_manager.end_session(session_id)

    async def forget_user(
        self, user_id: str, tenant_id: Optional[str] = None
    ) -> None:
        await self.user_memory.clear_user(user_id, tenant_id)
