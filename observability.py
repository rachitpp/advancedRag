"""
Lightweight observability primitives: stage timing, token accounting,
fallback enums, eval-friendly summaries.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Iterator

logger = logging.getLogger(__name__)


class FallbackReason(str, Enum):
    NONE = "none"
    EMPTY_RETRIEVAL = "empty_retrieval"
    WEB_SEARCH_FAILED = "web_search_failed"
    OFF_TOPIC = "off_topic"
    FAITHFULNESS_FAILED = "faithfulness_failed"
    SELF_CORRECTION_REGRESSED = "self_correction_regressed"


@dataclass
class StageRecord:
    name: str
    elapsed_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class TelemetryCollector:
    """
    Per-query telemetry. One instance per run_query call. Not threadsafe;
    in the async pipeline, all mutations happen on the event-loop thread.
    Concurrent fan-out (grading, claim verification) records into per-stage
    records via record_llm_call which is additive — last-write race on a
    single int counter is benign and never observed in our codebase
    (cooperative multitasking, not preemptive).
    """
    stages: dict[str, StageRecord] = field(default_factory=dict)

    def stage(self, name: str) -> StageRecord:
        if name not in self.stages:
            self.stages[name] = StageRecord(name=name)
        return self.stages[name]

    @contextmanager
    def time_stage(self, name: str) -> Iterator[StageRecord]:
        rec = self.stage(name)
        start = time.perf_counter()
        try:
            yield rec
        finally:
            rec.elapsed_ms += (time.perf_counter() - start) * 1000.0

    @asynccontextmanager
    async def atime_stage(self, name: str) -> AsyncIterator[StageRecord]:
        rec = self.stage(name)
        start = time.perf_counter()
        try:
            yield rec
        finally:
            rec.elapsed_ms += (time.perf_counter() - start) * 1000.0

    def record_llm_call(
        self, stage_name: str, tokens_in: int = 0, tokens_out: int = 0
    ) -> None:
        rec = self.stage(stage_name)
        rec.llm_calls += 1
        rec.tokens_in += tokens_in
        rec.tokens_out += tokens_out

    def record_cache(self, stage_name: str, hit: bool) -> None:
        rec = self.stage(stage_name)
        if hit:
            rec.cache_hits += 1
        else:
            rec.cache_misses += 1

    def total_tokens(self) -> tuple[int, int]:
        ti = sum(r.tokens_in for r in self.stages.values())
        to = sum(r.tokens_out for r in self.stages.values())
        return ti, to

    def total_ms(self) -> float:
        return sum(r.elapsed_ms for r in self.stages.values())

    def total_llm_calls(self) -> int:
        return sum(r.llm_calls for r in self.stages.values())

    def summary(self) -> dict[str, Any]:
        ti, to = self.total_tokens()
        return {
            "total_ms": round(self.total_ms(), 1),
            "total_tokens_in": ti,
            "total_tokens_out": to,
            "total_llm_calls": self.total_llm_calls(),
            "stages": {
                name: {
                    "ms": round(r.elapsed_ms, 1),
                    "llm_calls": r.llm_calls,
                    "tokens_in": r.tokens_in,
                    "tokens_out": r.tokens_out,
                    **(
                        {"cache_hits": r.cache_hits, "cache_misses": r.cache_misses}
                        if (r.cache_hits or r.cache_misses) else {}
                    ),
                    **({"notes": r.notes} if r.notes else {}),
                }
                for name, r in self.stages.items()
            },
        }


def extract_usage(response: Any) -> tuple[int, int]:
    """Pull (input_tokens, output_tokens) from a Vertex/LangChain response."""
    try:
        meta = getattr(response, "usage_metadata", None) or {}
        if isinstance(meta, dict):
            return (
                int(meta.get("input_tokens", 0) or 0),
                int(meta.get("output_tokens", 0) or 0),
            )
    except Exception:
        pass
    return 0, 0
