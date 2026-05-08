"""
Centralized, validated configuration for the CRAG pipeline.

Design choices:
  - One dataclass, frozen, validated in __post_init__.
  - Loaded once at process start via CRAGConfig.from_env().
  - Passed explicitly to subsystems — no module-level globals reaching into config.
  - Magic numbers from the original (0.5 grader threshold, 90-char log truncation,
    RRF k=60, etc.) live here as named fields with defaults.

Why not pydantic-settings: one engineer, one process, one env file. The dataclass
+ os.environ dance is 30 lines and has no extra dependency. Add pydantic-settings
when you have multiple deploy environments with overlapping overrides.
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


@dataclass(frozen=True)
class CRAGConfig:
    # ---- GCP / Vertex ----
    gcp_project: str
    gcp_location: str = "us-central1"

    # ---- Models ----
    main_model: str = "gemini-2.5-pro"
    fast_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"

    # ---- LLM params ----
    llm_temperature: float = 0.1
    llm_max_output_tokens: int = 2048
    llm_fast_max_tokens: int = 512

    # ---- Storage ----
    chroma_dir: str = "./chroma_db"

    # ---- Retrieval ----
    vector_k: int = 8
    vector_fetch_k: int = 20
    vector_lambda: float = 0.5
    bm25_k: int = 8
    rrf_k: int = 60                       # standard RRF constant
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    cross_encoder_top_n: int = 5
    max_context_tokens: int = 8000        # cap on labelled_context to control cost

    # ---- HyDE ----
    hyde_enabled: bool = True
    hyde_classifier_cache_size: int = 256

    # ---- Grading ----
    grader_confidence_threshold: float = 0.5
    min_relevant_for_local_only: int = 3   # graded-router threshold

    # ---- Web search ----
    web_search_max_results: int = 5

    # ---- Faithfulness verification ----
    claims_per_batch: int = 8              # batch claims into single LLM calls
    max_claims_verified: int = 15          # hard cap; sample beyond this
    min_claim_words: int = 3
    faithfulness_pass_threshold: float = 0.85
    faithfulness_retry_threshold: float = 0.50
    faithfulness_recheck_threshold: float = 0.85

    # ---- Concurrency ----
    max_chunk_workers: int = 8
    max_claim_workers: int = 4

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

    # ---- Memory: conversational (short-term, session-scoped) ----
    conversation_enabled: bool = True
    conversation_history_turns: int = 5         # turns passed to the rewriter
    conversation_session_ttl_s: int = 3600      # in-memory store eviction TTL

    # ---- Memory: user (long-term, cross-session) ----
    user_memory_enabled: bool = False           # opt-in — most apps don't need it
    user_memory_dir: str = "./user_memory_db"
    user_memory_top_k: int = 3                  # facts retrieved per query
    user_memory_min_relevance: float = 0.5      # below this, don't inject
    user_memory_extract_async: bool = True      # extract in background thread

    def __post_init__(self) -> None:
        # Validation — fail fast on misconfig rather than 200 queries later.
        if not 0.0 <= self.grader_confidence_threshold <= 1.0:
            raise ValueError("grader_confidence_threshold must be in [0,1]")
        if not 0.0 <= self.faithfulness_pass_threshold <= 1.0:
            raise ValueError("faithfulness_pass_threshold must be in [0,1]")
        if self.faithfulness_retry_threshold > self.faithfulness_pass_threshold:
            raise ValueError("retry_threshold must be <= pass_threshold")
        if self.claims_per_batch < 1:
            raise ValueError("claims_per_batch must be >= 1")
        if self.max_claims_verified < self.claims_per_batch:
            raise ValueError("max_claims_verified must be >= claims_per_batch")

    @classmethod
    def from_env(cls) -> "CRAGConfig":
        """Load config from environment, applying defaults."""
        return cls(
            gcp_project=_env("GOOGLE_CLOUD_PROJECT", required=True),
            gcp_location=_env("GCP_LOCATION", "us-central1"),
            main_model=_env("CRAG_MAIN_MODEL", "gemini-2.5-pro"),
            fast_model=_env("CRAG_FAST_MODEL", "gemini-2.5-flash"),
            chroma_dir=_env("CRAG_CHROMA_DIR", "./chroma_db"),
            llm_temperature=_env_float("CRAG_LLM_TEMPERATURE", 0.1),
            hyde_enabled=_env("CRAG_HYDE_ENABLED", "true").lower() == "true",
            user_memory_enabled=_env("CRAG_USER_MEMORY_ENABLED", "false").lower() == "true",
            user_memory_dir=_env("CRAG_USER_MEMORY_DIR", "./user_memory_db"),
        )
