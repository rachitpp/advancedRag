"""
Retrieval subsystem.

Architecture (after split):
  - RetrieverIndex      : owns infrastructure (Chroma, BM25, cross-encoder).
                          Lazy-loads, handles corpus drift, threadsafe.
                          Knows nothing about algorithms.
  - HybridRetriever     : pure retrieval algorithm. Takes a RetrieverIndex,
                          runs BM25 + Vector + RRF + rerank + budget cap.
                          Stateless w.r.t. infrastructure.
  - QueryAnalyzer       : query classification + HyDE generation.
  - WebSearcher         : Tavily-backed web search with query rewriting.

Why this split:
  - Index lifecycle (init, drift detection, locking) and retrieval algorithm
    (BM25, RRF, rerank) are unrelated concerns that were tangled in one class.
  - Tests can now stub a fake RetrieverIndex without spinning up Chroma.
  - Future swap: replace RetrieverIndex with an OpenSearch-backed version
    without touching the algorithm.

Pipeline flow (in HybridRetriever.retrieve):
  1. analyze(question)              -> factual | conceptual | procedural
  2. maybe_hyde(question, category) -> embedding query (raw or HyDE passage)
  3. parallel: bm25(question), vector(embedding_query)
  4. rrf_fuse(bm25, vector, k=60)   -> fused ranked list
  5. cross_encoder_rerank(top_n)    -> final ordered docs
  6. truncate_to_token_budget       -> final docs within token budget
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from .config import CRAGConfig
from .llm import LLMClients
from .observability import TelemetryCollector
from .prompts import HYDE_PROMPT, QUERY_CLASSIFIER_PROMPT

logger = logging.getLogger(__name__)


# ============================================================================
# Reciprocal Rank Fusion — pure function
# ============================================================================

def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    k: int = 60,
) -> list[tuple[Document, float, dict[int, int]]]:
    """
    Combine multiple ranked lists via RRF.

    Returns list of (doc, fused_score, {list_index: rank_in_that_list}) tuples,
    sorted by fused_score descending. Per-list ranks are returned for
    observability — you can see which retriever contributed each result.

    Doc identity is determined by page_content (after strip). Metadata is
    taken from the first list a doc appears in.
    """
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    contributions: dict[str, dict[int, int]] = {}

    for list_idx, ranked in enumerate(ranked_lists):
        for rank, doc in enumerate(ranked, start=1):
            key = doc.page_content.strip()
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs.setdefault(key, doc)
            contributions.setdefault(key, {})[list_idx] = rank

    fused = sorted(
        ((docs[key], score, contributions[key]) for key, score in scores.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    return fused


# ============================================================================
# Pure helpers — no state
# ============================================================================

def deduplicate_docs(docs: list[Document]) -> list[Document]:
    """Remove docs with duplicate page_content, preserving order."""
    seen: set[str] = set()
    out: list[Document] = []
    for doc in docs:
        key = doc.page_content.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(doc)
    return out


def _approx_tokens(text: str) -> int:
    """Conservative approximation: 1 token ≈ 4 chars. Avoids tiktoken dependency."""
    return len(text) // 4


def truncate_to_token_budget(
    docs: list[Document], max_tokens: int
) -> tuple[list[Document], bool]:
    """Drop docs from the end until total tokens fit. Returns (kept, truncated)."""
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


# ============================================================================
# RetrieverIndex — owns infrastructure, knows nothing about algorithms
# ============================================================================

class RetrieverIndex:
    """
    Owns the Chroma vector store, the BM25 index, and the cross-encoder model.

    Single responsibility: keep the search infrastructure alive and fresh.
    Provides cheap accessors for the algorithmic layer to use.

    Lifecycle:
      - Lazy init on first access (so importing the module doesn't require
        a live DB).
      - Threadsafe via Lock — concurrent first-call from multiple threads
        won't double-build.
      - Drift detection: if Chroma's corpus size drifts >20% from the build-
        time snapshot, the BM25 index is stale and gets rebuilt.

    Why this is a separate class:
      - Mixing infrastructure setup with algorithms made HybridRetriever
        ~150 lines. Splitting them out leaves both classes < 100L.
      - Tests can stub this class entirely without touching Chroma.
      - Future migration to OpenSearch / managed BM25 only touches this file.
    """

    DRIFT_THRESHOLD = 0.20  # 20% corpus drift triggers rebuild

    def __init__(self, cfg: CRAGConfig, clients: LLMClients):
        self.cfg = cfg
        self.clients = clients
        self._lock = threading.Lock()
        self._vector_store: Optional[Chroma] = None
        self._bm25: Optional[BM25Retriever] = None
        self._cross_encoder: Optional[HuggingFaceCrossEncoder] = None
        self._corpus_size: int = 0

    # ---- Lifecycle ----------------------------------------------------

    def ensure_ready(self) -> None:
        """
        Build the index if not yet built. Rebuild if corpus drifted.
        Cheap to call repeatedly — most calls are no-ops after init.
        """
        if self._is_fresh():
            return
        with self._lock:
            if self._is_fresh():
                return  # another thread did it
            self._build()

    def _is_fresh(self) -> bool:
        if self._bm25 is None or self._vector_store is None:
            return False
        try:
            current = self._vector_store._collection.count()
        except Exception:
            return True   # if we can't check, trust the existing index
        if not self._corpus_size:
            return True
        drift = abs(current - self._corpus_size) / self._corpus_size
        if drift > self.DRIFT_THRESHOLD:
            logger.info(
                "[index] Corpus drifted %d -> %d (%.0f%%) — rebuild required.",
                self._corpus_size, current, drift * 100,
            )
            return False
        return True

    def _build(self) -> None:
        logger.info("[index] Building hybrid retriever index...")
        store = Chroma(
            persist_directory=self.cfg.chroma_dir,
            embedding_function=self.clients.embeddings,
        )
        data = store.get(include=["documents", "metadatas"])
        docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(data["documents"], data["metadatas"])
        ]
        if not docs:
            raise RuntimeError(
                f"ChromaDB at '{self.cfg.chroma_dir}' is empty. "
                "Run your ingest pipeline first."
            )
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = self.cfg.bm25_k

        self._vector_store = store
        self._bm25 = bm25
        self._corpus_size = len(docs)

        if self._cross_encoder is None:
            self._cross_encoder = HuggingFaceCrossEncoder(
                model_name=self.cfg.cross_encoder_model
            )
        logger.info("[index] Built. Corpus size: %d docs.", len(docs))

    # ---- Search primitives --------------------------------------------

    def bm25_search(self, question: str) -> list[Document]:
        self.ensure_ready()
        assert self._bm25 is not None
        return self._bm25.invoke(question)

    def vector_search(self, query_text: str) -> list[Document]:
        self.ensure_ready()
        assert self._vector_store is not None
        retriever = self._vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": self.cfg.vector_k,
                "fetch_k": self.cfg.vector_fetch_k,
                "lambda_mult": self.cfg.vector_lambda,
            },
        )
        return retriever.invoke(query_text)

    def cross_encoder_score(
        self, question: str, docs: list[Document]
    ) -> list[float]:
        """Score (question, doc) pairs. Returns scores in input order."""
        self.ensure_ready()
        if not docs or self._cross_encoder is None:
            return []
        pairs = [(question, doc.page_content) for doc in docs]
        return list(self._cross_encoder.score(pairs))


# ============================================================================
# QueryAnalyzer — query classification + HyDE
# ============================================================================

@dataclass
class QueryShape:
    category: str          # "factual" | "conceptual" | "procedural"
    use_hyde: bool
    hyde_passage: Optional[str] = None


class QueryAnalyzer:
    """
    Classifies questions and (conditionally) generates HyDE passages.
    Caches classifications via LRU; HyDE passages are not cached (generative).
    """

    def __init__(self, cfg: CRAGConfig, clients: LLMClients):
        self.cfg = cfg
        self.clients = clients
        self._classify_chain = QUERY_CLASSIFIER_PROMPT | clients.fast | StrOutputParser()
        self._hyde_chain = HYDE_PROMPT | clients.fast | StrOutputParser()
        self._classify_cached = lru_cache(maxsize=cfg.hyde_classifier_cache_size)(
            self._classify_uncached
        )

    def _classify_uncached(self, question: str) -> str:
        try:
            raw = self._classify_chain.invoke({"question": question}).strip().lower()
            if raw in ("factual", "conceptual", "procedural"):
                return raw
        except Exception as exc:
            logger.warning("Query classifier failed: %s — defaulting to factual.", exc)
        return "factual"

    def analyze(
        self, question: str, telemetry: TelemetryCollector
    ) -> QueryShape:
        if not self.cfg.hyde_enabled:
            return QueryShape(category="factual", use_hyde=False)

        with telemetry.time_stage("classify"):
            category = self._classify_cached(question)
            telemetry.record_llm_call("classify")

        # HyDE helps on conceptual/procedural, neutral-to-harmful on factual.
        use_hyde = category in ("conceptual", "procedural")
        passage: Optional[str] = None
        if use_hyde:
            with telemetry.time_stage("hyde"):
                try:
                    passage = self._hyde_chain.invoke({"question": question}).strip()
                    telemetry.record_llm_call("hyde")
                    if not passage:
                        use_hyde = False
                except Exception as exc:
                    logger.warning("HyDE generation failed: %s", exc)
                    use_hyde = False

        telemetry.stage("classify").notes["category"] = category
        telemetry.stage("classify").notes["use_hyde"] = use_hyde
        return QueryShape(category=category, use_hyde=use_hyde, hyde_passage=passage)


# ============================================================================
# HybridRetriever — pure algorithm, no infrastructure ownership
# ============================================================================

class HybridRetriever:
    """
    Hybrid retrieval algorithm.

    Composition:
      RetrieverIndex (infra) + QueryAnalyzer (HyDE)
      -> BM25 + Vector -> RRF -> Cross-encoder rerank -> Token budget cap

    This class holds NO infrastructure state. The index is the source of truth
    for Chroma/BM25/cross-encoder; this class just orchestrates calls to it.
    """

    def __init__(
        self,
        cfg: CRAGConfig,
        index: RetrieverIndex,
        analyzer: QueryAnalyzer,
    ):
        self.cfg = cfg
        self.index = index
        self.analyzer = analyzer

    def _rerank(
        self, question: str, docs: list[Document]
    ) -> list[Document]:
        """Cross-encoder rerank, top-N. Returns input order if scoring fails."""
        if not docs:
            return []
        scores = self.index.cross_encoder_score(question, docs)
        if not scores:
            return docs[: self.cfg.cross_encoder_top_n]
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[: self.cfg.cross_encoder_top_n]]

    def retrieve(
        self,
        question: str,
        telemetry: TelemetryCollector,
    ) -> list[Document]:
        """
        Full retrieval flow. Returns final ranked docs, capped at
        cross_encoder_top_n and within max_context_tokens budget.
        """
        # 1. Analyze query — pick HyDE strategy.
        shape = self.analyzer.analyze(question, telemetry)
        vector_query = (
            shape.hyde_passage if (shape.use_hyde and shape.hyde_passage) else question
        )

        # 2. Run BM25 + vector retrieval.
        with telemetry.time_stage("retrieve_bm25"):
            bm25_results = self.index.bm25_search(question)
        with telemetry.time_stage("retrieve_vector"):
            vector_results = self.index.vector_search(vector_query)

        telemetry.stage("retrieve_bm25").notes["count"] = len(bm25_results)
        telemetry.stage("retrieve_vector").notes["count"] = len(vector_results)
        telemetry.stage("retrieve_vector").notes["hyde_used"] = shape.use_hyde

        # 3. RRF fusion.
        with telemetry.time_stage("rrf_fusion"):
            fused = reciprocal_rank_fusion(
                [bm25_results, vector_results], k=self.cfg.rrf_k
            )
            fused_docs = [doc for doc, _, _ in fused]
        telemetry.stage("rrf_fusion").notes["fused_count"] = len(fused_docs)

        # 4. Cross-encoder rerank on the top of the fused list.
        rerank_input = fused_docs[: self.cfg.vector_fetch_k + self.cfg.bm25_k]
        with telemetry.time_stage("rerank"):
            reranked = self._rerank(question, rerank_input)
        telemetry.stage("rerank").notes["count"] = len(reranked)

        # 5. Token budget.
        kept, truncated = truncate_to_token_budget(
            reranked, self.cfg.max_context_tokens
        )
        if truncated:
            telemetry.stage("rerank").notes["truncated_for_budget"] = True
            logger.info(
                "[retrieval] Context truncated for token budget: kept %d/%d docs.",
                len(kept), len(reranked),
            )
        return kept


# ============================================================================
# WebSearcher — Tavily + query rewriting
# ============================================================================

class WebSearcher:
    """Tavily-backed web search with query rewriting."""

    def __init__(self, cfg: CRAGConfig, clients: LLMClients):
        from langchain_community.tools.tavily_search import TavilySearchResults
        from .prompts import QUERY_REWRITE_PROMPT

        self.cfg = cfg
        self._tool = TavilySearchResults(max_results=cfg.web_search_max_results)
        self._rewrite_chain = QUERY_REWRITE_PROMPT | clients.fast | StrOutputParser()

    def search(
        self, question: str, telemetry: TelemetryCollector
    ) -> list[Document]:
        with telemetry.time_stage("web_rewrite"):
            try:
                rewritten = self._rewrite_chain.invoke({"question": question}).strip()
                telemetry.record_llm_call("web_rewrite")
            except Exception as exc:
                logger.warning("Query rewriter failed: %s", exc)
                rewritten = ""
            if not rewritten:
                rewritten = question

        telemetry.stage("web_rewrite").notes["query"] = rewritten
        logger.info("[retrieval] Web search query: '%s'", rewritten)

        with telemetry.time_stage("web_search"):
            try:
                results = self._tool.invoke(rewritten)
            except Exception as exc:
                logger.warning("[retrieval] Web search failed: %s", exc)
                return []

        docs: list[Document] = []
        for r in results:
            content = r.get("content") or r.get("snippet") or ""
            url = r.get("url", "unknown")
            if content:
                docs.append(
                    Document(
                        page_content=content,
                        metadata={"source": url, "page": "web", "provenance": "web"},
                    )
                )
        if not docs:
            logger.warning(
                "[retrieval] Web search returned no usable results for: '%s'",
                rewritten,
            )
        telemetry.stage("web_search").notes["count"] = len(docs)
        return docs