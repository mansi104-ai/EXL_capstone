"""Guardrail base class + shared context.

Each guardrail is a small, single-responsibility check that returns a
GuardrailResult. Guardrails are registered with the GuardrailManager, which runs
them and aggregates a GuardrailReport. New guardrails only need to subclass
Guardrail and be added to the manager — the architecture is extension-first.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Optional

from ..finance.taxonomy import Journey
from ..utils.types import GuardrailResult, PolicyChunk, Recommendation


@dataclass
class GuardrailContext:
    """Everything a guardrail might need to inspect for one turn."""

    text: str
    speaker: str = "customer"
    recommendation: Optional[Recommendation] = None
    retrieved: list[PolicyChunk] = field(default_factory=list)
    confidence_threshold: float = 0.55
    # The banking journey this turn was classified into. Lets a guardrail apply
    # the rule that belongs to the situation rather than one blanket rule.
    journey: Optional[Journey] = None
    # The draft reply the customer would actually hear. Distinct from `text`
    # (what the customer said) and from `recommendation` (what the handler is
    # advised to do) — only the clarity guardrail reads it.
    reply: str = ""


class Guardrail(abc.ABC):
    name: str = "guardrail"
    severity: str = "warn"  # info | warn | block

    @abc.abstractmethod
    def check(self, ctx: GuardrailContext) -> GuardrailResult:
        raise NotImplementedError

    def _ok(self, detail: str = "", **data) -> GuardrailResult:
        return GuardrailResult(name=self.name, passed=True, detail=detail,
                               severity="info", data=data)

    def _fail(self, detail: str, severity: Optional[str] = None, **data) -> GuardrailResult:
        return GuardrailResult(name=self.name, passed=False, detail=detail,
                               severity=severity or self.severity, data=data)
