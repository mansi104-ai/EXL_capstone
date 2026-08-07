"""Hallucination check against retrieved policy.

Ensures every policy reference the recommendation cites was actually retrieved
from the knowledge base for this turn. A citation that doesn't appear in the
retrieved set is treated as ungrounded (a hallucinated reference) and blocks the
recommendation — guidance in a regulated context must be traceable to policy.
"""
from __future__ import annotations

from ..utils.types import GuardrailResult
from .base import Guardrail, GuardrailContext


class HallucinationGuardrail(Guardrail):
    name = "hallucination_grounding"
    severity = "block"

    def check(self, ctx: GuardrailContext) -> GuardrailResult:
        if ctx.recommendation is None:
            return self._ok("No recommendation to verify.")
        retrieved_refs = {c.policy_reference for c in ctx.retrieved}
        cited = ctx.recommendation.citations
        if not cited:
            return self._fail(
                "Recommendation cites no policy reference — ungrounded.",
                severity="block",
            )
        ungrounded = [c for c in cited if c not in retrieved_refs]
        if ungrounded:
            return self._fail(
                f"Ungrounded citation(s) not in retrieved policy: {', '.join(ungrounded)}.",
                severity="block",
                ungrounded=ungrounded,
                retrieved=sorted(retrieved_refs),
            )
        return self._ok(f"All citations grounded in retrieved policy ({', '.join(cited)}).",
                        cited=cited)
