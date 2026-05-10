"""
Retrieval subsystem — async-first, Qdrant-backed, native sparse+dense hybrid.

Architecture:
  - QdrantRetrieverIndex   : owns the Qdrant client, sparse+dense embedders,
                             cross-encoder. Lazy collection check on first use.
  - HybridRetriever        : pure algorithm. Calls native Qdrant Query API
                             with prefetch + RRF fusion + metadata filters,
                             then optional cross-encoder rerank, optional
                             parent-child expansion, then token budget cap.
  - QueryAnalyzer          : query classification + HyDE generation.
  - WebSearcher            : Tavily web search with query rewriting.

Hybrid search is delegated to Qdrant (Query API with FusionQuery.RRF)
instead of being computed in Python — fewer round-trips, native scoring.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

from fastembed import SparseTextEmbedding
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from .cache import RetrievalCache
from .config import CRAGConfig
from .llm import LLMClients
from .observability import TelemetryCollector
from .prompts import HYDE_PROMPT, QUERY_CLASSIFIER_PROMPT, QUERY_REWRITE_PROMPT

logger = logging.getLogger(__name__)


# ============================================================================
# Pure helpers
# ============================================================================

def deduplicate_docs(docs: list[Document]) -> list[Document]:
    """Remove docs with duplicate (chunk_id, page_content) keys, preserving order."""
    seen: set[str] = set()
    out: list[Document] = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id") or ""
        key = f"{chunk_id}::{doc.page_content.strip()}"
        if key and key not in seen:
            seen.add(key)
            out.append(doc)
    return out


def _approx_tokens(text: str) -> int:
    return len(text) // 4


def truncate_to_token_budget(
    docs: list[Document], max_tokens: int
) -> tuple[list[Document], bool]:
    kept: list[Document] = []
    used = 0
    truncated = False
    for doc in docs:
        cost = _approx_tokens(doc.page_content)
        if used + cost > max_tokens:
            truncated = True
            break
        kept.append(doc)
        used += cost
    return kept, truncated


def build_qdrant_filter(
    filters: Optional[dict[str, Any]],
    tenant_id: Optional[str],
    children_only: bool,
) -> Optional[qm.Filter]:
    """Compose a Qdrant Filter from user-facing filter dict + tenant + structural flags."""
    must: list[qm.FieldCondition] = []

    if tenant_id is not None:
        must.append(qm.FieldCondition(
            key="tenant_id", match=qm.MatchValue(value=tenant_id)
        ))

    if children_only:
        must.append(qm.FieldCondition(
            key="is_parent", match=qm.MatchValue(value=False)
        ))

    if filters:
        for k, v in filters.items():
            if isinstance(v, list):
                must.append(qm.FieldCondition(key=k, match=qm.MatchAny(any=v)))
            elif isinstance(v, dict) and ("gte" in v or "lte" in v):
                must.append(qm.FieldCondition(
                    key=k,
                    range=qm.Range(gte=v.get("gte"), lte=v.get("lte")),
                ))
            else:
                must.append(qm.FieldCondition(key=k, match=qm.MatchValue(value=v)))

    return qm.Filter(must=must) if must else None


def _point_to_document(point: Any) -> Document:
    payload = point.payload or {}
    content = payload.pop("content", "")
    metadata = dict(payload)
    metadata["chunk_id"] = str(point.id)
    metadata["score"] = float(getattr(point, "score", 0.0) or 0.0)
    return Document(page_content=content, metadata=metadata)


# ============================================================================
# QdrantRetrieverIndex
# ============================================================================

class QdrantRetrieverIndex:
    """
    Owns Qdrant async client, dense embeddings, sparse model, cross-encoder.

    Single responsibility: keep search infrastructure alive, ensure the target
    collection exists, expose typed search primitives. Knows nothing about
    ranking algorithms or pipeline policy.
    """

    def __init__(self, cfg: CRAGConfig, clients: LLMClients):
        self.cfg = cfg
        self.clients = clients
        self._client: Optional[AsyncQdrantClient] = None
        self._sparse: Optional[SparseTextEmbedding] = None
        self._cross_encoder: Optional[HuggingFaceCrossEncoder] = None
        self._init_lock = asyncio.Lock()
        self._collection_ready = False

    # ---- Lifecycle ----------------------------------------------------

    async def ensure_ready(self) -> None:
        if self._collection_ready and self._client is not None:
            return
        async with self._init_lock:
            if self._collection_ready and self._client is not None:
                return
            self._client = AsyncQdrantClient(
                url=self.cfg.qdrant_url,
                api_key=self.cfg.qdrant_api_key,
                prefer_grpc=self.cfg.qdrant_prefer_grpc,
            )
            await self._ensure_collection()
            if self._sparse is None:
                self._sparse = SparseTextEmbedding(model_name=self.cfg.sparse_model)
            if self._cross_encoder is None:
                self._cross_encoder = HuggingFaceCrossEncoder(
                    model_name=self.cfg.cross_encoder_model
                )
            self._collection_ready = True

    async def _ensure_collection(self) -> None:
        assert self._client is not None
        exists = await self._client.collection_exists(self.cfg.qdrant_collection)
        if exists:
            return
        logger.info(
            "[index] Creating Qdrant collection '%s'", self.cfg.qdrant_collection
        )
        await self._client.create_collection(
            collection_name=self.cfg.qdrant_collection,
            vectors_config={
                "dense": qm.VectorParams(
                    size=self.cfg.embedding_dim, distance=qm.Distance.COSINE
                ),
            },
            sparse_vectors_config={
                "sparse": qm.SparseVectorParams(
                    index=qm.SparseIndexParams(on_disk=False)
                ),
            },
        )
        # Indexes for filtering
        for field, schema in [
            ("tenant_id", qm.PayloadSchemaType.KEYWORD),
            ("source", qm.PayloadSchemaType.KEYWORD),
            ("parent_id", qm.PayloadSchemaType.KEYWORD),
            ("is_parent", qm.PayloadSchemaType.BOOL),
        ]:
            try:
                await self._client.create_payload_index(
                    collection_name=self.cfg.qdrant_collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as exc:
                logger.debug("Payload index %s exists or skipped: %s", field, exc)

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass

    # ---- Embedders ----------------------------------------------------

    async def embed_dense(self, text: str) -> list[float]:
        return await self.clients.embeddings.aembed_query(text)

    def embed_sparse(self, text: str) -> qm.SparseVector:
        assert self._sparse is not None
        result = next(iter(self._sparse.embed([text])))
        return qm.SparseVector(
            indices=result.indices.tolist(), values=result.values.tolist()
        )

    # ---- Search primitives -------------------------------------------

    async def hybrid_search(
        self,
        question: str,
        vector_query_text: str,
        limit: int,
        prefetch_dense_k: int,
        prefetch_sparse_k: int,
        rrf_k: int,
        query_filter: Optional[qm.Filter],
    ) -> list[Document]:
        """
        Native Qdrant hybrid: dense prefetch + sparse prefetch + RRF fusion.
        BM25-equivalent without keeping a Python BM25 index in process RAM.
        """
        await self.ensure_ready()
        assert self._client is not None

        dense_vec = await self.embed_dense(vector_query_text)
        sparse_vec = self.embed_sparse(question)

        response = await self._client.query_points(
            collection_name=self.cfg.qdrant_collection,
            prefetch=[
                qm.Prefetch(
                    query=dense_vec, using="dense", limit=prefetch_dense_k,
                    filter=query_filter,
                ),
                qm.Prefetch(
                    query=sparse_vec, using="sparse", limit=prefetch_sparse_k,
                    filter=query_filter,
                ),
            ],
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return [_point_to_document(p) for p in response.points]

    async def fetch_by_ids(self, ids: list[str]) -> list[Document]:
        await self.ensure_ready()
        assert self._client is not None
        if not ids:
            return []
        result = await self._client.retrieve(
            collection_name=self.cfg.qdrant_collection,
            ids=ids,
            with_payload=True,
        )
        return [_point_to_document(p) for p in result]

    def cross_encoder_score(
        self, question: str, docs: list[Document]
    ) -> list[float]:
        if not docs or self._cross_encoder is None:
            return []
        pairs = [(question, d.page_content) for d in docs]
        return list(self._cross_encoder.score(pairs))


# ============================================================================
# QueryAnalyzer
# ============================================================================

@dataclass
class QueryShape:
    category: str
    use_hyde: bool
    hyde_passage: Optional[str] = None


class QueryAnalyzer:
    def __init__(self, cfg: CRAGConfig, clients: LLMClients):
        self.cfg = cfg
        self.clients = clients
        self._classify_chain = QUERY_CLASSIFIER_PROMPT | clients.fast | StrOutputParser()
        self._hyde_chain = HYDE_PROMPT | clients.fast | StrOutputParser()
        self._classify_cached = lru_cache(maxsize=cfg.hyde_classifier_cache_size)(
            self._classify_sync_marker
        )

    def _classify_sync_marker(self, question: str) -> str:  # only used as cache key
        return question

    async def _classify(self, question: str) -> str:
        try:
            raw = await self._classify_chain.ainvoke({"question": question})
            raw = (raw or "").strip().lower()
            if raw in ("factual", "conceptual", "procedural"):
                return raw
        except Exception as exc:
            logger.warning("Classifier failed: %s — defaulting to factual.", exc)
        return "factual"

    async def analyze(
        self, question: str, telemetry: TelemetryCollector
    ) -> QueryShape:
        if not self.cfg.hyde_enabled:
            return QueryShape(category="factual", use_hyde=False)

        async with telemetry.atime_stage("classify"):
            category = await self._classify(question)
            telemetry.record_llm_call("classify")

        use_hyde = category in ("conceptual", "procedural")
        passage: Optional[str] = None
        if use_hyde:
            async with telemetry.atime_stage("hyde"):
                try:
                    passage = (await self._hyde_chain.ainvoke({"question": question})).strip()
                    telemetry.record_llm_call("hyde")
                    if not passage:
                        use_hyde = False
                except Exception as exc:
                    logger.warning("HyDE failed: %s", exc)
                    use_hyde = False

        telemetry.stage("classify").notes["category"] = category
        telemetry.stage("classify").notes["use_hyde"] = use_hyde
        return QueryShape(category=category, use_hyde=use_hyde, hyde_passage=passage)


# ============================================================================
# HybridRetriever
# ============================================================================

class HybridRetriever:
    def __init__(
        self,
        cfg: CRAGConfig,
        index: QdrantRetrieverIndex,
        analyzer: QueryAnalyzer,
        cache: RetrievalCache,
    ):
        self.cfg = cfg
        self.index = index
        self.analyzer = analyzer
        self.cache = cache

    async def _rerank(
        self, question: str, docs: list[Document]
    ) -> list[Document]:
        if not docs:
            return []
        # cross-encoder is sync/CPU; offload to thread to avoid blocking loop
        scores = await asyncio.to_thread(
            self.index.cross_encoder_score, question, docs
        )
        if not scores:
            return docs[: self.cfg.cross_encoder_top_n]
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [d for d, _ in ranked[: self.cfg.cross_encoder_top_n]]

    async def _expand_to_parents(
        self, child_docs: list[Document]
    ) -> list[Document]:
        """Replace child docs with their parents where parent_id is set."""
        if not self.cfg.parent_child_expansion:
            return child_docs

        parent_ids: list[str] = []
        seen: set[str] = set()
        for doc in child_docs:
            pid = doc.metadata.get("parent_id")
            if pid and pid not in seen:
                parent_ids.append(pid)
                seen.add(pid)

        if not parent_ids:
            return child_docs

        parents = await self.index.fetch_by_ids(parent_ids)
        # Preserve ranking order: emit each child's parent the first time we see it.
        parent_by_id = {p.metadata.get("chunk_id"): p for p in parents}
        out: list[Document] = []
        emitted: set[str] = set()
        for doc in child_docs:
            pid = doc.metadata.get("parent_id")
            if pid and pid in parent_by_id and pid not in emitted:
                out.append(parent_by_id[pid])
                emitted.add(pid)
            elif not pid:
                out.append(doc)
        return out

    async def retrieve(
        self,
        question: str,
        telemetry: TelemetryCollector,
        filters: Optional[dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> list[Document]:
        # ---- Cache lookup
        cache_filters = {"filters": filters, "tenant_id": tenant_id}
        cached = await self.cache.get(question, cache_filters)
        if cached is not None:
            telemetry.record_cache("retrieve_total", hit=True)
            telemetry.stage("retrieve_total").notes["cached"] = True
            return cached
        telemetry.record_cache("retrieve_total", hit=False)

        # ---- Analyze
        shape = await self.analyzer.analyze(question, telemetry)
        vector_query = (
            shape.hyde_passage if (shape.use_hyde and shape.hyde_passage) else question
        )

        qfilter = build_qdrant_filter(filters, tenant_id, children_only=True)

        # ---- Hybrid retrieval (native RRF in Qdrant)
        async with telemetry.atime_stage("hybrid_search"):
            fetch_k = max(
                self.cfg.vector_fetch_k + self.cfg.sparse_k,
                self.cfg.cross_encoder_top_n * 4,
            )
            fused = await self.index.hybrid_search(
                question=question,
                vector_query_text=vector_query,
                limit=fetch_k,
                prefetch_dense_k=self.cfg.vector_fetch_k,
                prefetch_sparse_k=self.cfg.sparse_k,
                rrf_k=self.cfg.rrf_k,
                query_filter=qfilter,
            )
        telemetry.stage("hybrid_search").notes["count"] = len(fused)
        telemetry.stage("hybrid_search").notes["hyde_used"] = shape.use_hyde

        # ---- Rerank
        async with telemetry.atime_stage("rerank"):
            reranked = await self._rerank(question, fused)
        telemetry.stage("rerank").notes["count"] = len(reranked)

        # ---- Parent-child expansion
        if self.cfg.parent_child_expansion:
            async with telemetry.atime_stage("parent_expand"):
                expanded = await self._expand_to_parents(reranked)
            telemetry.stage("parent_expand").notes["count"] = len(expanded)
        else:
            expanded = reranked

        # ---- Token budget
        kept, truncated = truncate_to_token_budget(
            expanded, self.cfg.max_context_tokens
        )
        if truncated:
            telemetry.stage("rerank").notes["truncated_for_budget"] = True

        # ---- Cache store
        await self.cache.set(question, kept, cache_filters)
        return kept


# ============================================================================
# WebSearcher
# ============================================================================

class WebSearcher:
    def __init__(self, cfg: CRAGConfig, clients: LLMClients):
        from langchain_community.tools.tavily_search import TavilySearchResults

        self.cfg = cfg
        self._tool = TavilySearchResults(max_results=cfg.web_search_max_results)
        self._rewrite_chain = QUERY_REWRITE_PROMPT | clients.fast | StrOutputParser()

    async def search(
        self, question: str, telemetry: TelemetryCollector
    ) -> list[Document]:
        async with telemetry.atime_stage("web_rewrite"):
            try:
                rewritten = (
                    await self._rewrite_chain.ainvoke({"question": question})
                ).strip()
                telemetry.record_llm_call("web_rewrite")
            except Exception as exc:
                logger.warning("Query rewriter failed: %s", exc)
                rewritten = ""
            if not rewritten:
                rewritten = question

        telemetry.stage("web_rewrite").notes["query"] = rewritten

        async with telemetry.atime_stage("web_search"):
            try:
                results = await self._tool.ainvoke(rewritten)
            except Exception as exc:
                logger.warning("Web search failed: %s", exc)
                return []

        docs: list[Document] = []
        for r in results:
            content = r.get("content") or r.get("snippet") or ""
            url = r.get("url", "unknown")
            if content:
                docs.append(Document(
                    page_content=content,
                    metadata={"source": url, "page": "web", "provenance": "web"},
                ))
        telemetry.stage("web_search").notes["count"] = len(docs)
        return docs
