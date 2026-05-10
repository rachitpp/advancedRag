"""
LLM client factory. Vertex AI native, with retry wired into the runnable graph.

Three clients by purpose:
  - main:       answer generation (streaming, non-zero temp)
  - structured: claim extraction, self-correction (temp=0)
  - fast:       grading, classification, claim verification (temp=0, low max_tokens)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from google.api_core.exceptions import (
    DeadlineExceeded,
    InternalServerError,
    ResourceExhausted,
    ServiceUnavailable,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

from .config import CRAGConfig

logger = logging.getLogger(__name__)


# Errors that warrant a retry. We want transient infra failures, not bugs.
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
    DeadlineExceeded,
    ServiceUnavailable,
    InternalServerError,
    ResourceExhausted,
)


def attach_retry(runnable: Runnable, cfg: CRAGConfig) -> Runnable:
    """
    Wrap a runnable with config-driven retry.

    Uses LangChain's RunnableRetry rather than tenacity because (a) it composes
    cleanly with .ainvoke/.astream, (b) it supports per-runnable configuration,
    and (c) it propagates retry context into LangSmith traces automatically.
    """
    return runnable.with_retry(
        retry_if_exception_type=RETRYABLE_EXCEPTIONS,
        wait_exponential_jitter=True,
        stop_after_attempt=cfg.llm_retry_attempts,
    )


@dataclass
class LLMClients:
    """
    Bundle of LLM clients constructed once and passed around explicitly.

    Each underlying client is wrapped with retry. All chains built by the
    pipeline inherit retry through composition (PROMPT | client | parser).
    """
    main: Runnable
    structured: Runnable
    fast: Runnable
    embeddings: VertexAIEmbeddings
    cfg: CRAGConfig

    @classmethod
    def from_config(cls, cfg: CRAGConfig) -> "LLMClients":
        common = dict(project=cfg.gcp_project, location=cfg.gcp_location)

        main = ChatVertexAI(
            model_name=cfg.main_model,
            temperature=cfg.llm_temperature,
            max_output_tokens=cfg.llm_max_output_tokens,
            streaming=True,
            **common,
        )
        structured = ChatVertexAI(
            model_name=cfg.main_model,
            temperature=0.0,
            max_output_tokens=cfg.llm_max_output_tokens,
            streaming=False,
            **common,
        )
        fast = ChatVertexAI(
            model_name=cfg.fast_model,
            temperature=0.0,
            max_output_tokens=cfg.llm_fast_max_tokens,
            streaming=False,
            **common,
        )
        embeddings = VertexAIEmbeddings(model_name=cfg.embedding_model, **common)

        return cls(
            main=attach_retry(main, cfg),
            structured=attach_retry(structured, cfg),
            fast=attach_retry(fast, cfg),
            embeddings=embeddings,
            cfg=cfg,
        )

    async def aclose(self) -> None:
        """Best-effort close hook. Vertex clients hold no persistent connection."""
        return None
