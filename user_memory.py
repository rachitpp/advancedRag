"""
User memory (Tier 2): durable per-user facts in Postgres + pgvector.

Single shared table, filtered by user_id (and optional tenant_id) at query
time. Replaces the previous per-user Chroma collections, which did not scale.

Schema (created idempotently at startup):
  user_facts(
    id BIGSERIAL PK,
    user_id TEXT NOT NULL,
    tenant_id TEXT,
    fact_text TEXT NOT NULL,
    embedding VECTOR(d) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
  )
  + IVFFlat index on (embedding) and B-tree on (user_id, tenant_id)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    String,
    Text,
    delete,
    func,
    select,
    text,
)
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .config import CRAGConfig
from .llm import LLMClients
from .observability import TelemetryCollector
from .prompts import USER_FACT_EXTRACTOR_PROMPT, USER_MEMORY_CONTEXT_HEADER
from .schemas import ExtractedUserFacts

logger = logging.getLogger(__name__)


# ============================================================================
# ORM
# ============================================================================

class _Base(DeclarativeBase):
    pass


def make_user_fact_model(embedding_dim: int) -> type:
    class UserFact(_Base):
        __tablename__ = "user_facts"

        id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
        user_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
        tenant_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
        fact_text: Mapped[str] = mapped_column(Text, nullable=False)
        embedding: Mapped[list[float]] = mapped_column(Vector(embedding_dim), nullable=False)
        created_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True), server_default=func.now()
        )

        __table_args__ = (
            Index("ix_user_facts_user_tenant", "user_id", "tenant_id"),
        )

    return UserFact


@dataclass
class RetrievedFact:
    text: str
    score: float


# ============================================================================
# Store
# ============================================================================

class PostgresUserMemoryStore:
    """
    Async pgvector-backed user-fact store. Idempotent schema init.
    """

    def __init__(self, cfg: CRAGConfig, clients: LLMClients):
        self.cfg = cfg
        self.clients = clients
        self._engine: Optional[AsyncEngine] = None
        self._session_maker: Optional[async_sessionmaker[AsyncSession]] = None
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self.UserFact = make_user_fact_model(cfg.embedding_dim)

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            self._engine = create_async_engine(
                self.cfg.postgres_dsn,
                pool_size=self.cfg.postgres_pool_size,
                max_overflow=self.cfg.postgres_max_overflow,
                pool_pre_ping=True,
            )
            self._session_maker = async_sessionmaker(
                self._engine, expire_on_commit=False
            )

            async with self._engine.begin() as conn:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                await conn.run_sync(_Base.metadata.create_all)
                # Create IVFFlat index for cosine; idempotent guard.
                await conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_user_facts_embedding "
                    "ON user_facts USING ivfflat (embedding vector_cosine_ops) "
                    "WITH (lists = 100);"
                ))
            self._initialized = True

    async def aclose(self) -> None:
        if self._engine is not None:
            await self._engine.dispose()

    # ---- Write ----

    async def add_facts(
        self,
        user_id: str,
        facts: list[str],
        tenant_id: Optional[str] = None,
    ) -> int:
        if not facts:
            return 0
        await self.initialize()
        assert self._session_maker is not None
        try:
            embeddings = await self.clients.embeddings.aembed_documents(facts)
        except Exception as exc:
            logger.warning("Embedding facts failed: %s", exc)
            return 0

        rows = [
            self.UserFact(
                user_id=user_id,
                tenant_id=tenant_id,
                fact_text=fact,
                embedding=emb,
            )
            for fact, emb in zip(facts, embeddings)
        ]
        try:
            async with self._session_maker() as session:
                session.add_all(rows)
                await session.commit()
            return len(rows)
        except Exception as exc:
            logger.warning("add_facts failed for user %s: %s", user_id, exc)
            return 0

    # ---- Read ----

    async def retrieve(
        self,
        user_id: str,
        question: str,
        top_k: int,
        min_relevance: float,
        tenant_id: Optional[str] = None,
    ) -> list[RetrievedFact]:
        await self.initialize()
        assert self._session_maker is not None
        try:
            qvec = await self.clients.embeddings.aembed_query(question)
        except Exception as exc:
            logger.warning("Embedding query failed: %s", exc)
            return []

        UF = self.UserFact
        distance_expr = UF.embedding.cosine_distance(qvec)
        stmt = (
            select(UF.fact_text, distance_expr.label("distance"))
            .where(UF.user_id == user_id)
        )
        if tenant_id is not None:
            stmt = stmt.where(UF.tenant_id == tenant_id)
        stmt = stmt.order_by(distance_expr).limit(top_k)

        try:
            async with self._session_maker() as session:
                result = await session.execute(stmt)
                rows = result.all()
        except Exception as exc:
            logger.warning("Retrieve facts failed: %s", exc)
            return []

        out: list[RetrievedFact] = []
        for fact_text, distance in rows:
            relevance = 1.0 - float(distance)  # cosine distance -> similarity
            if relevance >= min_relevance:
                out.append(RetrievedFact(text=fact_text, score=relevance))
        return out

    async def clear_user(self, user_id: str, tenant_id: Optional[str] = None) -> int:
        await self.initialize()
        assert self._session_maker is not None
        UF = self.UserFact
        stmt = delete(UF).where(UF.user_id == user_id)
        if tenant_id is not None:
            stmt = stmt.where(UF.tenant_id == tenant_id)
        try:
            async with self._session_maker() as session:
                result = await session.execute(stmt)
                await session.commit()
                return int(result.rowcount or 0)
        except Exception as exc:
            logger.warning("clear_user failed: %s", exc)
            return 0


# ============================================================================
# Extractor
# ============================================================================

class UserFactExtractor:
    def __init__(self, cfg: CRAGConfig, clients: LLMClients):
        self.cfg = cfg
        self._chain = (
            USER_FACT_EXTRACTOR_PROMPT
            | clients.fast.with_structured_output(ExtractedUserFacts)
        )

    async def extract(
        self, user_message: str, assistant_message: str
    ) -> list[str]:
        try:
            result = await self._chain.ainvoke({
                "user_message": user_message,
                "assistant_message": assistant_message,
            })
        except Exception as exc:
            logger.warning("Fact extraction failed: %s", exc)
            return []

        cleaned: list[str] = []
        for fact in result.facts:
            stripped = fact.strip().rstrip(".")
            if not stripped or len(stripped.split()) < 3 or stripped.endswith("?"):
                continue
            cleaned.append(stripped)
        return cleaned


# ============================================================================
# Manager
# ============================================================================

class UserMemoryManager:
    """
    Coordinates extraction + storage + retrieval. Owned by the pipeline.
    Read path is awaited inline. Write path is fire-and-forget via a tracked
    task set on the pipeline so we can drain it on shutdown.
    """

    def __init__(
        self,
        cfg: CRAGConfig,
        clients: LLMClients,
        store: PostgresUserMemoryStore,
        background_tasks: set[asyncio.Task],
    ):
        self.cfg = cfg
        self.store = store
        self.extractor = UserFactExtractor(cfg, clients)
        self._background_tasks = background_tasks

    @property
    def enabled(self) -> bool:
        return self.cfg.user_memory_enabled

    async def retrieve_for_question(
        self,
        user_id: Optional[str],
        question: str,
        telemetry: TelemetryCollector,
        tenant_id: Optional[str] = None,
    ) -> list[RetrievedFact]:
        if not self.enabled or not user_id:
            return []
        async with telemetry.atime_stage("user_memory_retrieve"):
            facts = await self.store.retrieve(
                user_id=user_id,
                question=question,
                top_k=self.cfg.user_memory_top_k,
                min_relevance=self.cfg.user_memory_min_relevance,
                tenant_id=tenant_id,
            )
        telemetry.stage("user_memory_retrieve").notes["count"] = len(facts)
        return facts

    def schedule_extraction(
        self,
        user_id: Optional[str],
        user_message: str,
        assistant_message: str,
        tenant_id: Optional[str] = None,
    ) -> None:
        if not self.enabled or not user_id:
            return
        if not user_message or not assistant_message:
            return

        async def _run() -> None:
            try:
                facts = await self.extractor.extract(user_message, assistant_message)
                if facts:
                    added = await self.store.add_facts(user_id, facts, tenant_id)
                    if added:
                        logger.info("[user_memory] Stored %d fact(s).", added)
            except Exception as exc:
                logger.warning("[user_memory] Background extraction failed: %s", exc)

        task = asyncio.create_task(_run())
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    @staticmethod
    def format_for_prompt(facts: list[RetrievedFact]) -> str:
        if not facts:
            return ""
        bullets = "\n".join(f"- {f.text}" for f in facts)
        return f"{USER_MEMORY_CONTEXT_HEADER}\n{bullets}\n"

    async def clear_user(
        self, user_id: str, tenant_id: Optional[str] = None
    ) -> None:
        n = await self.store.clear_user(user_id, tenant_id)
        logger.info("[user_memory] Cleared %d fact(s) for user.", n)
