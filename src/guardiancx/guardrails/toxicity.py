"""Toxicity filter guardrail.

Lightweight lexicon-based detector for abusive/toxic language. In production
this would call a hosted classifier; the interface is identical so swapping the
backend is a one-line change.
"""
from __future__ import annotations

import re

from ..utils.types import GuardrailResult
from .base import Guardrail, GuardrailContext

_TOXIC = {
    "useless", "idiot", "stupid", "hate you", "shut up", "moron",
    "worthless", "pathetic", "damn bot", "useless bot",
}


class ToxicityGuardrail(Guardrail):
    name = "toxicity"
    severity = "warn"

    def check(self, ctx: GuardrailContext) -> GuardrailResult:
        low = ctx.text.lower()
        hits = [w for w in _TOXIC if re.search(r"\b" + re.escape(w) + r"\b", low)]
        if not hits:
            return self._ok("No toxic language detected.")
        return self._fail(
            f"Toxic language detected ({', '.join(hits)}). Flag for tone-aware handling.",
            severity="warn",
            terms=hits,
        )
