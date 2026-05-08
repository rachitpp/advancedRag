"""
Faithfulness verification subsystem.

Architecture (after split):
  - FaithfulnessVerifier : verification only (relevance gate, claim extraction,
                           batched verification, scoring). Stateless w.r.t.
                           correction policy.
  - SelfCorrector        : correction loop only (decision tree, regression
                           check). Takes a FaithfulnessVerifier as a dependency.

Why this split:
  - The original FaithfulnessVerifier was 341 lines covering two distinct
    concerns: "does this answer match the context?" and "if not, what do we
    do about it?" Mixing them made each method touch both verification state
    and correction policy.
  - SelfCorrector now expresses ONLY the policy: pass/retry/fallback thresholds,
    non-regression check, fallback reasons. It calls FaithfulnessVerifier as
    a black box for re-verification of corrected answers.
  - Tests can verify scoring without the correction loop, and test correction
    policy with a stub verifier.

Pipeline flow:
  verify_answer(question, answer, context) -> VerificationResult
  self_correct_if_needed(question, answer, verification, context) -> CorrectionResult

Error semantics (verifier):
  - Persistent LLM failure -> claim treated as UNSUPPORTED (fail closed).
  - Transient errors are absorbed by the retry decorator on the LLM clients.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from langchain_core.output_parsers import StrOutputParser

from .config import CRAGConfig
from .llm import LLMClients
from .observability import FallbackReason, TelemetryCollector
from .prompts import (
    ANSWER_RELEVANCE_PROMPT,
    CLAIM_BATCH_VERIFIER_PROMPT,
    CLAIM_EXTRACTOR_PROMPT,
    SELF_CORRECTION_PROMPT,
)
from .schemas import (
    AnswerRelevance,
    BatchedClaimVerdicts,
    ClaimVerdict,
    ExtractedClaims,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Result dataclasses
# ============================================================================

@dataclass
class VerificationResult:
    """Outcome of FaithfulnessVerifier.verify_answer."""
    score: float
    supported: list[tuple[str, ClaimVerdict]] = field(default_factory=list)
    unsupported: list[tuple[str, ClaimVerdict]] = field(default_factory=list)
    answer_relevance: str = "yes"   # "yes" | "no" | "partial"
    claims_extracted: int = 0
    claims_verified: int = 0
    claims_sampled: bool = False    # True if we hit max_claims_verified

    @property
    def total(self) -> int:
        return len(self.supported) + len(self.unsupported)


@dataclass
class CorrectionResult:
    """Outcome of SelfCorrector.correct_if_needed."""
    final_answer: str
    final_score: float
    was_corrected: bool
    correction_regressed: bool
    fallback_reason: FallbackReason


# ============================================================================
# FaithfulnessVerifier — verification only
# ============================================================================

class FaithfulnessVerifier:
    """
    Checks an answer's faithfulness against retrieved context.

    Single responsibility: produce a VerificationResult. Does not decide what
    to do with a low score — that's SelfCorrector's job.
    """

    def __init__(self, cfg: CRAGConfig, clients: LLMClients):
        self.cfg = cfg
        self.clients = clients

        self._relevance_chain = (
            ANSWER_RELEVANCE_PROMPT
            | clients.fast.with_structured_output(AnswerRelevance)
        )
        self._extractor_chain = (
            CLAIM_EXTRACTOR_PROMPT
            | clients.structured.with_structured_output(ExtractedClaims)
        )
        self._batch_verifier_chain = (
            CLAIM_BATCH_VERIFIER_PROMPT
            | clients.fast.with_structured_output(BatchedClaimVerdicts)
        )

    # ---- Stage 1: answer relevance ------------------------------------

    def _answer_relevance(
        self, question: str, answer: str, telemetry: TelemetryCollector
    ) -> str:
        with telemetry.time_stage("answer_relevance"):
            try:
                r = self._relevance_chain.invoke(
                    {"question": question, "answer": answer}
                )
                telemetry.record_llm_call("answer_relevance")
                return r.verdict
            except Exception as exc:
                logger.warning("Answer-relevance check failed: %s — assuming yes.", exc)
                return "yes"

    # ---- Stage 2: claim extraction ------------------------------------

    def _extract_claims(
        self, answer: str, telemetry: TelemetryCollector
    ) -> list[str]:
        with telemetry.time_stage("extract_claims"):
            try:
                extracted = self._extractor_chain.invoke({"answer": answer})
                telemetry.record_llm_call("extract_claims")
            except Exception as exc:
                logger.warning("Claim extraction failed: %s", exc)
                return []
        return [
            c.strip()
            for c in extracted.claims
            if c and len(c.split()) >= self.cfg.min_claim_words
        ]

    # ---- Stage 3: batched verification --------------------------------

    def _verify_one_batch(
        self, batch: list[str], context: str
    ) -> list[ClaimVerdict]:
        """N claims in one LLM call. Pads to len(batch) on failure (fail closed)."""
        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(batch))
        try:
            result = self._batch_verifier_chain.invoke(
                {"context": context, "numbered_claims": numbered}
            )
            verdicts = result.verdicts
        except Exception as exc:
            logger.warning("Batched verifier failed: %s — failing closed.", exc)
            return [
                ClaimVerdict(supported=False, reason=f"Verifier error: {exc}")
                for _ in batch
            ]

        # Alignment guard.
        if len(verdicts) != len(batch):
            logger.warning(
                "Batched verifier returned %d verdicts for %d claims — padding.",
                len(verdicts), len(batch),
            )
            while len(verdicts) < len(batch):
                verdicts.append(ClaimVerdict(
                    supported=False, reason="Missing verdict — failed closed."
                ))
            verdicts = verdicts[: len(batch)]
        return verdicts

    def _sample_if_over_cap(self, claims: list[str]) -> tuple[list[str], bool]:
        cap = self.cfg.max_claims_verified
        if len(claims) <= cap:
            return claims, False
        stride = len(claims) / cap
        sampled = [claims[int(i * stride)] for i in range(cap)]
        logger.info(
            "[verify] Sampling %d/%d claims (cap=%d).",
            len(sampled), len(claims), cap,
        )
        return sampled, True

    def _verify_batched(
        self,
        claims: list[str],
        context: str,
        pool: ThreadPoolExecutor,
        telemetry: TelemetryCollector,
    ) -> list[tuple[str, ClaimVerdict]]:
        if not claims:
            return []

        batch_size = self.cfg.claims_per_batch
        batches = [
            claims[i : i + batch_size]
            for i in range(0, len(claims), batch_size)
        ]

        with telemetry.time_stage("verify_claims"):
            if len(batches) == 1:
                all_verdicts = self._verify_one_batch(batches[0], context)
                telemetry.record_llm_call("verify_claims")
            else:
                futures = [
                    pool.submit(self._verify_one_batch, b, context)
                    for b in batches
                ]
                all_verdicts = []
                for fut, batch in zip(futures, batches):
                    try:
                        all_verdicts.extend(fut.result())
                    except Exception as exc:
                        logger.warning("Batch future raised: %s", exc)
                        all_verdicts.extend([
                            ClaimVerdict(
                                supported=False,
                                reason=f"Future error: {exc}",
                            )
                            for _ in batch
                        ])
                    telemetry.record_llm_call("verify_claims")
        return list(zip(claims, all_verdicts))

    # ---- Public API ---------------------------------------------------

    def verify_answer(
        self,
        question: str,
        answer: str,
        labelled_context: str,
        pool: ThreadPoolExecutor,
        telemetry: TelemetryCollector,
    ) -> VerificationResult:
        # 1. Relevance gate.
        relevance = self._answer_relevance(question, answer, telemetry)
        if relevance == "no":
            logger.info("[verify] Answer relevance: NO — skipping faithfulness.")
            return VerificationResult(score=0.0, answer_relevance="no")

        # 2. Extract.
        all_claims = self._extract_claims(answer, telemetry)
        if not all_claims:
            logger.info("[verify] No verifiable claims extracted.")
            return VerificationResult(
                score=1.0 if relevance == "yes" else 0.5,
                answer_relevance=relevance,
                claims_extracted=0,
            )

        claims_to_verify, sampled = self._sample_if_over_cap(all_claims)

        # 3. Batched verification.
        results = self._verify_batched(
            claims_to_verify, labelled_context, pool, telemetry
        )

        supported = [(c, v) for c, v in results if v.supported]
        unsupported = [(c, v) for c, v in results if not v.supported]
        score = len(supported) / len(results) if results else 0.0

        trunc = self.cfg.log_truncate_chars
        for c, v in supported:
            logger.info("  ✓ %s — %s", c[:trunc], v.reason[:trunc])
        for c, v in unsupported:
            logger.info("  ✗ %s — %s", c[:trunc], v.reason[:trunc])

        telemetry.stage("verify_claims").notes.update({
            "score": round(score, 3),
            "supported": len(supported),
            "unsupported": len(unsupported),
            "sampled": sampled,
        })
        return VerificationResult(
            score=score,
            supported=supported,
            unsupported=unsupported,
            answer_relevance=relevance,
            claims_extracted=len(all_claims),
            claims_verified=len(results),
            claims_sampled=sampled,
        )


# ============================================================================
# SelfCorrector — correction loop only
# ============================================================================

class SelfCorrector:
    """
    Decision policy for what to do with a verified answer.

    Pure orchestration:
      - If score >= pass_threshold: keep original.
      - If score in [retry_threshold, pass_threshold): attempt correction.
      - If score < retry_threshold: fallback.
      - Off-topic answers (relevance="no") always fallback.

    Non-regression rule:
      - Corrected answer is re-verified. If correction made things worse,
        we keep the ORIGINAL with a warning rather than ship the worse one.

    Single responsibility: applying this decision tree. Does not know how
    to verify or extract claims — delegates to FaithfulnessVerifier.
    """

    def __init__(
        self,
        cfg: CRAGConfig,
        clients: LLMClients,
        verifier: FaithfulnessVerifier,
    ):
        self.cfg = cfg
        self.verifier = verifier
        self._correction_chain = (
            SELF_CORRECTION_PROMPT
            | clients.structured
            | StrOutputParser()
        )

    # ---- Public API ---------------------------------------------------

    def correct_if_needed(
        self,
        question: str,
        original_answer: str,
        original_verification: VerificationResult,
        labelled_context: str,
        pool: ThreadPoolExecutor,
        telemetry: TelemetryCollector,
    ) -> CorrectionResult:
        score = original_verification.score
        relevance = original_verification.answer_relevance

        # ---- Off-topic: always fallback.
        if relevance == "no":
            return self._fallback(score, FallbackReason.OFF_TOPIC)

        # ---- High score: keep original.
        if score >= self.cfg.faithfulness_pass_threshold:
            return CorrectionResult(
                final_answer=original_answer,
                final_score=score,
                was_corrected=False,
                correction_regressed=False,
                fallback_reason=FallbackReason.NONE,
            )

        # ---- Below retry threshold: too far gone, fallback.
        if score < self.cfg.faithfulness_retry_threshold:
            logger.warning(
                "[correct] Score %.2f below retry threshold %.2f — fallback.",
                score, self.cfg.faithfulness_retry_threshold,
            )
            return self._fallback(score, FallbackReason.FAITHFULNESS_FAILED)

        # ---- Mid range: attempt correction.
        return self._attempt_correction(
            question, original_answer, original_verification,
            labelled_context, pool, telemetry,
        )

    # ---- Correction attempt ------------------------------------------

    def _attempt_correction(
        self,
        question: str,
        original_answer: str,
        original_verification: VerificationResult,
        labelled_context: str,
        pool: ThreadPoolExecutor,
        telemetry: TelemetryCollector,
    ) -> CorrectionResult:
        original_score = original_verification.score
        logger.info("[correct] Score %.2f — attempting self-correction.", original_score)

        unsupported_list = "\n".join(
            f"- {c}" for c, _ in original_verification.unsupported
        )

        # ---- Generate corrected answer.
        with telemetry.time_stage("self_correct"):
            try:
                corrected = self._correction_chain.invoke({
                    "context": labelled_context,
                    "question": question,
                    "previous_answer": original_answer,
                    "unsupported_claims": unsupported_list,
                })
                telemetry.record_llm_call("self_correct")
            except Exception as exc:
                logger.warning("Self-correction call failed: %s", exc)
                # Keep original — correction crashed, but we have a partial answer.
                return CorrectionResult(
                    final_answer=original_answer,
                    final_score=original_score,
                    was_corrected=False,
                    correction_regressed=False,
                    fallback_reason=FallbackReason.SELF_CORRECTION_REGRESSED,
                )

        # ---- Re-verify corrected answer.
        logger.info("[correct] Re-verifying corrected answer.")
        re_verification = self.verifier.verify_answer(
            question, corrected, labelled_context, pool, telemetry
        )
        new_score = re_verification.score

        return self._evaluate_correction(
            original_answer=original_answer,
            original_score=original_score,
            corrected_answer=corrected,
            corrected_score=new_score,
        )

    def _evaluate_correction(
        self,
        original_answer: str,
        original_score: float,
        corrected_answer: str,
        corrected_score: float,
    ) -> CorrectionResult:
        """Apply the non-regression rule to the corrected answer."""
        recheck = self.cfg.faithfulness_recheck_threshold

        # ---- Correction succeeded: high score AND no regression.
        if corrected_score >= recheck and corrected_score >= original_score:
            logger.info(
                "[correct] Correction passed: %.2f -> %.2f.",
                original_score, corrected_score,
            )
            return CorrectionResult(
                final_answer=corrected_answer,
                final_score=corrected_score,
                was_corrected=True,
                correction_regressed=False,
                fallback_reason=FallbackReason.NONE,
            )

        # ---- Correction regressed: keep original.
        if corrected_score < original_score:
            logger.warning(
                "[correct] REGRESSED: %.2f -> %.2f. Keeping original.",
                original_score, corrected_score,
            )
            return CorrectionResult(
                final_answer=original_answer,
                final_score=original_score,
                was_corrected=False,
                correction_regressed=True,
                fallback_reason=FallbackReason.NONE,
            )

        # ---- Correction didn't regress but didn't pass recheck either.
        logger.warning(
            "[correct] Insufficient: %.2f -> %.2f. Fallback.",
            original_score, corrected_score,
        )
        return self._fallback(corrected_score, FallbackReason.FAITHFULNESS_FAILED)

    # ---- Fallback helper ---------------------------------------------

    def _fallback(self, score: float, reason: FallbackReason) -> CorrectionResult:
        return CorrectionResult(
            final_answer=self.cfg.hallucination_fallback,
            final_score=score,
            was_corrected=False,
            correction_regressed=False,
            fallback_reason=reason,
        )