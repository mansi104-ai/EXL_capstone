"""Guardrail manager — runs the registered guardrails and aggregates a report.

Three stages, because there are three different things to protect. Input
guardrails (PII, injection, toxicity) run on the raw utterance. Output guardrails
(confidence, grounding, prohibited actions, approval) run on the recommendation —
they protect the firm. **Reply guardrails** run on the draft the customer would
actually hear, and protect the customer: correct advice phrased in policy
language is still advice they cannot act on.

The manager exposes the masked text and a single GuardrailReport the Supervisor
agent uses to route the case.

The output set includes a finance-specific check, `prohibited_action`, which
blocks advice that would cause harm regardless of how well grounded it is —
offering credit to a customer disclosing gambling harm, for instance. The reply
set includes `data_disclosure`, which blocks any identifier from the account book
appearing in what the customer is told.

Extending: add a Guardrail subclass and register it in `default_input_guardrails`
or `default_output_guardrails`. Optionally, a NeMo Guardrails config can be
layered in front via `nemo_rails` (loaded only if the package is present).
"""
from __future__ import annotations

from typing import Optional

from config.settings import get_settings

from ..finance.taxonomy import Journey
from ..utils.logging import get_logger
from ..utils.types import GuardrailReport, PolicyChunk, Recommendation
from .approval import ApprovalGuardrail
from .base import Guardrail, GuardrailContext
from .clarity import ClarityGuardrail
from .disclosure import DisclosureGuardrail
from .confidence import ConfidenceGuardrail
from .hallucination import HallucinationGuardrail
from .injection import InjectionGuardrail
from .pii import PIIGuardrail
from .prohibited_action import ProhibitedActionGuardrail
from .toxicity import ToxicityGuardrail

log = get_logger("guardrails.manager")


def default_input_guardrails() -> list[Guardrail]:
    return [PIIGuardrail(), InjectionGuardrail(), ToxicityGuardrail()]


def default_reply_guardrails() -> list[Guardrail]:
    """Guardrails on the customer-facing draft.

    Disclosure runs first and blocks; clarity runs second and warns. A reply that
    leaks an account number is not improved by being easy to understand.
    """
    return [DisclosureGuardrail(), ClarityGuardrail()]


def default_output_guardrails() -> list[Guardrail]:
    settings = get_settings()
    return [
        ConfidenceGuardrail(),
        HallucinationGuardrail(),
        ProhibitedActionGuardrail(),
        ApprovalGuardrail(require_high_risk=settings.guardiancx_high_risk_approval),
    ]


class GuardrailManager:
    def __init__(
        self,
        input_guardrails: Optional[list[Guardrail]] = None,
        output_guardrails: Optional[list[Guardrail]] = None,
        reply_guardrails: Optional[list[Guardrail]] = None,
    ):
        self.input_guardrails = input_guardrails or default_input_guardrails()
        self.output_guardrails = output_guardrails or default_output_guardrails()
        self.reply_guardrails = reply_guardrails or default_reply_guardrails()
        self._nemo = self._maybe_load_nemo()

    def _maybe_load_nemo(self):
        try:
            import nemoguardrails  # noqa: F401

            log.info("NeMo Guardrails available (custom validators still primary).")
            return True
        except Exception:  # noqa: BLE001
            return None

    def run_input(self, text: str, speaker: str = "customer") -> tuple[str, GuardrailReport]:
        """Run input guardrails; return (masked_text, report)."""
        ctx = GuardrailContext(text=text, speaker=speaker)
        report = GuardrailReport()
        masked = text
        for g in self.input_guardrails:
            res = g.check(ctx)
            report.results.append(res)
            if res.name == "pii_masking":
                masked = res.data.get("masked_text", masked)
        return masked, report

    def run_output(
        self,
        text: str,
        recommendation: Optional[Recommendation],
        retrieved: list[PolicyChunk],
        journey: Optional[Journey] = None,
    ) -> GuardrailReport:
        settings = get_settings()
        ctx = GuardrailContext(
            text=text,
            recommendation=recommendation,
            retrieved=retrieved,
            confidence_threshold=settings.guardiancx_confidence_threshold,
            journey=journey,
        )
        report = GuardrailReport()
        for g in self.output_guardrails:
            report.results.append(g.check(ctx))
        return report

    def run_reply(self, reply: str, journey=None,
                  account_refused: bool = False) -> GuardrailReport:
        """Check the draft the customer would hear.

        Runs after the reply is composed rather than inside the graph, because
        the reply is written from the recommendation and does not exist yet when
        the output guardrails run.
        """
        ctx = GuardrailContext(text="", reply=reply, journey=journey,
                               account_refused=account_refused)
        report = GuardrailReport()
        for g in self.reply_guardrails:
            report.results.append(g.check(ctx))
        return report


_MANAGER: GuardrailManager | None = None


def get_guardrail_manager() -> GuardrailManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = GuardrailManager()
    return _MANAGER
