"""
Async Redis-backed retrieval cache.

Key shape: {namespace}:{sha256(question_normalized | filters_json)}
Stores: serialized list of (page_content, metadata) tuples.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

import redis.asyncio as redis
from langchain_core.documents import Document

from .config import CRAGConfig

logger = logging.getLogger(__name__)


def _normalize_question(q: str) -> str:
    return " ".join(q.lower().strip().split())


def _filter_signature(filters: Optional[dict[str, Any]]) -> str:
    if not filters:
        return ""
    return json.dumps(filters, sort_keys=True, default=str)


def cache_key(namespace: str, question: str, filters: Optional[dict[str, Any]]) -> str:
    payload = f"{_normalize_question(question)}||{_filter_signature(filters)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


@dataclass
class RetrievalCache:
    """
    Thin async wrapper over Redis. Disabled cleanly when cfg.cache_enabled=False
    or when the Redis connection cannot be established (degrades to no-op).
    """
    cfg: CRAGConfig
    _client: Optional[redis.Redis] = None
    _disabled: bool = False

    @classmethod
    def from_config(cls, cfg: CRAGConfig) -> "RetrievalCache":
        if not cfg.cache_enabled:
            return cls(cfg=cfg, _disabled=True)
        try:
            client = redis.from_url(cfg.redis_url, decode_responses=False)
            return cls(cfg=cfg, _client=client)
        except Exception as exc:
            logger.warning("Redis cache disabled: %s", exc)
            return cls(cfg=cfg, _disabled=True)

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass

    async def get(
        self, question: str, filters: Optional[dict[str, Any]] = None
    ) -> Optional[list[Document]]:
        if self._disabled or self._client is None:
            return None
        key = cache_key(self.cfg.cache_namespace, question, filters)
        try:
            raw = await self._client.get(key)
        except Exception as exc:
            logger.warning("Cache GET failed: %s", exc)
            return None
        if raw is None:
            return None
        try:
            payload = json.loads(raw)
            return [
                Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
                for d in payload
            ]
        except Exception as exc:
            logger.warning("Cache decode failed: %s", exc)
            return None

    async def set(
        self,
        question: str,
        docs: list[Document],
        filters: Optional[dict[str, Any]] = None,
    ) -> None:
        if self._disabled or self._client is None:
            return
        key = cache_key(self.cfg.cache_namespace, question, filters)
        payload = json.dumps([
            {"page_content": d.page_content, "metadata": d.metadata} for d in docs
        ])
        try:
            await self._client.setex(key, self.cfg.cache_ttl_seconds, payload)
        except Exception as exc:
            logger.warning("Cache SET failed: %s", exc)
