"""Mandatory human-approval guardrail.

High-risk recommendations must not be surfaced to the customer without a human
sign-off. This guardrail decides whether a turn requires approval based on the
assessed risk level; the actual approve/reject happens in the Human Approval
Queue and is written to the evidence log.
"""
from __future__ import annotations

from ..utils.types import GuardrailResult, RiskLevel
from .base import Guardrail, GuardrailContext


class ApprovalGuardrail(Guardrail):
    name = "human_approval"
    severity = "block"

    def __init__(self, require_high_risk: bool = True):
        self.require_high_risk = require_high_risk

    def check(self, ctx: GuardrailContext) -> GuardrailResult:
        rec = ctx.recommendation
        risk = rec.risk_level if rec else RiskLevel.LOW
        if self.require_high_risk and risk == RiskLevel.HIGH:
            return self._fail(
                "High-risk recommendation requires human approval before it is surfaced.",
                severity="block",
                requires_approval=True,
            )
        return self._ok("No mandatory approval required at this risk level.",
                        requires_approval=False)
