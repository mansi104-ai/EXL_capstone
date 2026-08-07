"""Confidence-threshold guardrail.

Blocks a recommendation from being surfaced (or forces review) when the model's
confidence is below the configured threshold — low-confidence guidance in a
regulated setting should not reach a handler unqualified.
"""
from __future__ import annotations

from ..utils.types import GuardrailResult
from .base import Guardrail, GuardrailContext


class ConfidenceGuardrail(Guardrail):
    name = "confidence_threshold"
    severity = "warn"

    def check(self, ctx: GuardrailContext) -> GuardrailResult:
        if ctx.recommendation is None:
            return self._ok("No recommendation to score.")
        conf = ctx.recommendation.confidence
        if conf >= ctx.confidence_threshold:
            return self._ok(f"Confidence {conf:.2f} ≥ threshold {ctx.confidence_threshold:.2f}.",
                            confidence=conf)
        return self._fail(
            f"Confidence {conf:.2f} < threshold {ctx.confidence_threshold:.2f}. "
            "Route to review before surfacing.",
            severity="warn",
            confidence=conf,
        )
