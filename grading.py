"""
Async chunk grading. Bounded concurrency via semaphore.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from langchain_core.documents import Document

from .config import CRAGConfig
from .llm import LLMClients
from .observability import TelemetryCollector
from .prompts import GRADER_PROMPT
from .schemas import GradeResult

logger = logging.getLogger(__name__)


@dataclass
class GradeReport:
    relevant_docs: list[Document]
    total: int
    relevant_count: int
    mean_confidence: float

    @property
    def has_any_relevant(self) -> bool:
        return self.relevant_count > 0

    def has_strong_local_signal(self, cfg: CRAGConfig) -> bool:
        return (
            self.relevant_count >= cfg.min_relevant_for_local_only
            and self.mean_confidence >= cfg.min_local_confidence
        )


class ChunkGrader:
    def __init__(self, cfg: CRAGConfig, clients: LLMClients):
        self.cfg = cfg
        self._chain = GRADER_PROMPT | clients.fast.with_structured_output(GradeResult)

    async def _grade_one(
        self, idx: int, doc: Document, question: str
    ) -> tuple[int, Document, GradeResult]:
        try:
            result = await self._chain.ainvoke(
                {"document": doc.page_content, "question": question}
            )
            return idx, doc, result
        except Exception as exc:
            logger.warning("Chunk %d grading failed: %s", idx, exc)
            return idx, doc, GradeResult(
                score="no", confidence=0.0, reason=f"Grading error: {exc}"
            )

    async def grade(
        self,
        docs: list[Document],
        question: str,
        semaphore: asyncio.Semaphore,
        telemetry: TelemetryCollector,
        label: str = "local",
    ) -> GradeReport:
        if not docs:
            return GradeReport(
                relevant_docs=[], total=0, relevant_count=0, mean_confidence=0.0
            )

        stage_name = f"grade_{label}"

        async def _bounded(i: int, d: Document):
            async with semaphore:
                out = await self._grade_one(i, d, question)
                telemetry.record_llm_call(stage_name)
                return out

        async with telemetry.atime_stage(stage_name):
            tasks = [asyncio.create_task(_bounded(i, d)) for i, d in enumerate(docs)]
            graded = await asyncio.gather(*tasks)

        graded.sort(key=lambda x: x[0])
        relevant: list[Document] = []
        confidences: list[float] = []
        threshold = self.cfg.grader_confidence_threshold

        for i, doc, grade in graded:
            is_rel = grade.score == "yes" and grade.confidence >= threshold
            icon = "✓" if is_rel else "✗"
            logger.info(
                "  [%s] Chunk %d %s (conf=%.2f) — %s",
                label, i + 1, icon, grade.confidence,
                grade.reason[: self.cfg.log_truncate_chars],
            )
            if is_rel:
                relevant.append(doc)
                confidences.append(grade.confidence)

        report = GradeReport(
            relevant_docs=relevant,
            total=len(docs),
            relevant_count=len(relevant),
            mean_confidence=(sum(confidences) / len(confidences)) if confidences else 0.0,
        )
        telemetry.stage(stage_name).notes.update({
            "total": report.total,
            "relevant": report.relevant_count,
            "mean_conf": round(report.mean_confidence, 3),
        })
        return report
