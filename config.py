"""
Centralized, validated configuration. Frozen dataclass loaded once at startup.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    val = os.environ.get(key, default)
    if required and not val:
        raise EnvironmentError(f"Required env var '{key}' is not set.")
    return val or ""


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return int(raw) if raw else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    return float(raw) if raw else default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class CRAGConfig:
    # ---- GCP / Vertex ----
    gcp_project: str
    gcp_location: str = "us-central1"

    # ---- Models ----
    main_model: str = "gemini-2.5-pro"
    fast_model: str = "gemini-2.5-flash"
    embedding_model: str = "text-embedding-005"
    embedding_dim: int = 768
    sparse_model: str = "Qdrant/bm25"

    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 2048
    llm_fast_max_tokens: int = 512

    # ---- Qdrant ----
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection: str = "crag_docs"
    qdrant_prefer_grpc: bool = True

    # ---- Postgres / pgvector (user memory) ----
    postgres_dsn: str = "postgresql+asyncpg://crag:crag@localhost:5432/crag"
    postgres_pool_size: int = 5
    postgres_max_overflow: int = 10

    # ---- Redis cache ----
    redis_url: str = "redis://localhost:6379/0"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    cache_namespace: str = "crag:retr"

    # ---- Retrieval ----
    vector_k: int = 8
    vector_fetch_k: int = 20
    sparse_k: int = 8
    rrf_k: int = 60
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_top_n: int = 5
    max_context_tokens: int = 8000
    parent_child_expansion: bool = True

    # ---- HyDE ----
    hyde_enabled: bool = True
    hyde_classifier_cache_size: int = 256

    # ---- Grading ----
    grader_confidence_threshold: float = 0.5
    min_relevant_for_local_only: int = 3
    min_local_confidence: float = 0.7   # was hardcoded 0.7

    # ---- Web search ----
    web_search_max_results: int = 5

    # ---- Faithfulness ----
    claims_per_batch: int = 8
    max_claims_verified: int = 15
    min_claim_words: int = 3
    faithfulness_pass_threshold: float = 0.85
    faithfulness_retry_threshold: float = 0.50
    faithfulness_recheck_threshold: float = 0.85

    # ---- Concurrency ----
    max_chunk_concurrency: int = 8
    max_claim_concurrency: int = 4
    max_memory_concurrency: int = 2

    # ---- Retry ----
    llm_retry_attempts: int = 3
    llm_retry_min_wait_s: float = 0.5
    llm_retry_max_wait_s: float = 8.0

    # ---- Observability ----
    log_truncate_chars: int = 90

    # ---- Fallbacks ----
    hallucination_fallback: str = (
        "I could not produce a sufficiently grounded answer from the available sources. "
        "Please rephrase the question or consult the source documents directly."
    )

    # ---- Memory: conversational ----
    conversation_enabled: bool = True
    conversation_history_turns: int = 5
    conversation_session_ttl_s: int = 3600

    # ---- Memory: user ----
    user_memory_enabled: bool = False
    user_memory_top_k: int = 3
    user_memory_min_relevance: float = 0.5

    # ---- Eval ----
    eval_recall_drop_threshold: float = 0.02
    eval_mrr_drop_threshold: float = 0.02
    eval_ndcg_drop_threshold: float = 0.02

    # ---- Ingestion ----
    ingest_chunk_size: int = 800
    ingest_chunk_overlap: int = 150
    ingest_parent_chunk_size: int = 2400
    ingest_embedding_batch_size: int = 64
    ingest_upsert_batch_size: int = 256
    ingest_max_concurrency: int = 4

    def __post_init__(self) -> None:
        if not 0.0 <= self.grader_confidence_threshold <= 1.0:
            raise ValueError("grader_confidence_threshold must be in [0,1]")
        if not 0.0 <= self.min_local_confidence <= 1.0:
            raise ValueError("min_local_confidence must be in [0,1]")
        if not 0.0 <= self.faithfulness_pass_threshold <= 1.0:
            raise ValueError("faithfulness_pass_threshold must be in [0,1]")
        if self.faithfulness_retry_threshold > self.faithfulness_pass_threshold:
            raise ValueError("retry_threshold must be <= pass_threshold")
        if self.claims_per_batch < 1:
            raise ValueError("claims_per_batch must be >= 1")
        if self.max_claims_verified < self.claims_per_batch:
            raise ValueError("max_claims_verified must be >= claims_per_batch")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")

    @classmethod
    def from_env(cls) -> "CRAGConfig":
        return cls(
            gcp_project=_env("GOOGLE_CLOUD_PROJECT", required=True),
            gcp_location=_env("GCP_LOCATION", "us-central1"),
            main_model=_env("CRAG_MAIN_MODEL", "gemini-2.5-pro"),
            fast_model=_env("CRAG_FAST_MODEL", "gemini-2.5-flash"),
            embedding_model=_env("CRAG_EMBEDDING_MODEL", "text-embedding-005"),
            embedding_dim=_env_int("CRAG_EMBEDDING_DIM", 768),
            qdrant_url=_env("CRAG_QDRANT_URL", "http://localhost:6333"),
            qdrant_api_key=_env("CRAG_QDRANT_API_KEY") or None,
            qdrant_collection=_env("CRAG_QDRANT_COLLECTION", "crag_docs"),
            qdrant_prefer_grpc=_env_bool("CRAG_QDRANT_GRPC", True),
            postgres_dsn=_env(
                "CRAG_POSTGRES_DSN",
                "postgresql+asyncpg://crag:crag@localhost:5432/crag",
            ),
            redis_url=_env("CRAG_REDIS_URL", "redis://localhost:6379/0"),
            cache_enabled=_env_bool("CRAG_CACHE_ENABLED", True),
            cache_ttl_seconds=_env_int("CRAG_CACHE_TTL_S", 300),
            llm_temperature=_env_float("CRAG_LLM_TEMPERATURE", 0.1),
            hyde_enabled=_env_bool("CRAG_HYDE_ENABLED", True),
            user_memory_enabled=_env_bool("CRAG_USER_MEMORY_ENABLED", False),
            parent_child_expansion=_env_bool("CRAG_PARENT_CHILD_EXPANSION", True),
        )
