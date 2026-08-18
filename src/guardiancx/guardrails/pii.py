"""PII masking guardrail.

Detects and masks the personal data a UK banking conversation actually carries —
emails, phone numbers, card numbers, sort codes, account numbers, IBANs, National
Insurance numbers, postcodes and dates of birth — before text is sent to the LLM,
written to the evidence store, or rendered on screen.

The pattern set lives here and is shared with the live-call redactor
(`voice.live_pii`), so a disclosure is treated identically whether the customer
typed it or said it out loud. Order matters: the most specific patterns run first,
so a sort code is not consumed by the looser card-number pattern.
"""
from __future__ import annotations

import re

from ..utils.types import GuardrailResult
from .base import Guardrail, GuardrailContext

_PATTERNS = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # National Insurance number: two prefix letters (excluding the unused set),
    # six digits, one suffix letter A-D.
    ("ni_number", re.compile(
        r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b", re.IGNORECASE)),
    ("iban", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("sort_code", re.compile(r"\b\d{2}[- ]?\d{2}[- ]?\d{2}\b")),
    ("account", re.compile(r"\b\d{8}\b")),
    ("phone", re.compile(r"\b(?:\+?44|0)\d{9,10}\b")),
    # Dates of birth, numeric and written. Bounded to plausible birth years so
    # ordinary dates in conversation ("on the 3rd of June") survive.
    ("date_of_birth", re.compile(
        r"\b(?:0?[1-9]|[12]\d|3[01])[/\-.](?:0?[1-9]|1[0-2])[/\-.](?:19|20)\d{2}\b")),
    ("date_of_birth", re.compile(
        r"\b(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+(?:of\s+)?"
        r"(?:january|february|march|april|may|june|july|august|september|october|"
        r"november|december)\s+(?:19|20)\d{2}\b", re.IGNORECASE)),
    # UK postcode, outward + inward.
    ("postcode", re.compile(
        r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.IGNORECASE)),
]


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """Replace every recognised identifier with a labelled placeholder.

    Returns the masked text and a per-category count of what was redacted, which
    is what the evidence record stores — the fact of the redaction, never the
    value.
    """
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
