"""PII masking guardrail.

Detects and masks common UK banking PII (emails, phone numbers, card numbers,
sort codes, account numbers, IBANs) before text is logged or sent to the LLM.
Returns the masked text and a count of redactions in `data`.
"""
from __future__ import annotations

import re

from ..utils.types import GuardrailResult
from .base import Guardrail, GuardrailContext

_PATTERNS = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("sort_code", re.compile(r"\b\d{2}[- ]?\d{2}[- ]?\d{2}\b")),
    ("account", re.compile(r"\b\d{8}\b")),
    ("phone", re.compile(r"\b(?:\+?44|0)\d{9,10}\b")),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
]


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}
    masked = text
    for label, pattern in _PATTERNS:
        found = pattern.findall(masked)
        if found:
            counts[label] = counts.get(label, 0) + len(found)
            masked = pattern.sub(f"[{label.upper()}_REDACTED]", masked)
    return masked, counts


class PIIGuardrail(Guardrail):
    name = "pii_masking"
    severity = "warn"

    def check(self, ctx: GuardrailContext) -> GuardrailResult:
        masked, counts = mask_pii(ctx.text)
        total = sum(counts.values())
        if total == 0:
            return self._ok("No PII detected.", masked_text=ctx.text, redactions={})
        return self._fail(
            f"Masked {total} PII item(s): {', '.join(f'{k}×{v}' for k, v in counts.items())}.",
            severity="warn",
            masked_text=masked,
            redactions=counts,
        )
