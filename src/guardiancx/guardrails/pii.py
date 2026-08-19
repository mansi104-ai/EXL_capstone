"""PII masking guardrail.

Detects and masks the personal data an Indian banking conversation actually
carries — Aadhaar and PAN, account numbers, IFSC codes, UPI IDs, mobile numbers,
card numbers, PIN codes, dates of birth and email — before text is sent to the
model, written to the evidence store, or rendered on screen.

**Aadhaar is treated as the most sensitive item here, and deliberately so.** The
Aadhaar Act restricts its storage and display, UIDAI requires that only the last
four digits ever be shown, and it is the one identifier in the set that unlocks
identity elsewhere rather than merely one account. It is matched first, on the
widest pattern, and its own detection never records the value.

The pattern set lives here and is shared with the live-call redactor
(`voice.live_pii`), so a disclosure is treated identically whether the customer
typed it or said it out loud. Order matters: the most specific patterns run
first, so an IFSC code is not consumed by a looser alphanumeric match and a PAN
is not swallowed by the card-number rule.
"""
from __future__ import annotations

import re

from ..utils.types import GuardrailResult
from .base import Guardrail, GuardrailContext

# Ordered longest-and-most-specific first. Two orderings here were bugs before
# they were rules:
#
# * **Card before Aadhaar.** Aadhaar is exactly twelve digits and a card is
#   thirteen to sixteen, so on a sixteen-digit card the Aadhaar rule matched
#   twelve of them, masked those, and left the remaining four digits of the card
#   sitting in the clear — a leak created by the redactor itself. The longer,
#   more specific pattern has to consume the run first.
# * **Mobile before account.** An Indian mobile is ten digits starting 6-9,
#   which the 9-18 digit account rule would otherwise swallow. Both get masked
#   either way, but the evidence record should say which it was.
_PATTERNS = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # IFSC: four-letter bank code, a zero, then six branch characters.
    ("ifsc", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE)),
    # PAN: five letters, four digits, one letter.
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)),
    # UPI handle: name@bank. After email so real addresses win.
    ("upi_id", re.compile(r"\b[\w.-]{3,}@(?:okhdfcbank|oksbi|okaxis|okicici|paytm|"
                          r"ybl|ibl|axl|upi|pib)\b", re.IGNORECASE)),
    ("card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    # Aadhaar: exactly twelve digits, written in groups of four, never starting
    # 0 or 1. The lookarounds keep it from biting a chunk out of a longer run.
    ("aadhaar", re.compile(r"(?<!\d)(?<!\d[\s-])[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?![\s-]?\d)")),
    ("mobile", re.compile(r"(?:\+?91[\s-]?)?\b[6-9]\d{4}[\s-]?\d{5}\b")),
    # Bank account numbers in India run 9-18 digits.
    ("account", re.compile(r"\b\d{9,18}\b")),
    ("date_of_birth", re.compile(
        r"\b(?:0?[1-9]|[12]\d|3[01])[/\-.](?:0?[1-9]|1[0-2])[/\-.](?:19|20)\d{2}\b")),
    ("date_of_birth", re.compile(
        r"\b(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+(?:of\s+)?"
        r"(?:january|february|march|april|may|june|july|august|september|october|"
        r"november|december)\s+(?:19|20)\d{2}\b", re.IGNORECASE)),
    # Indian PIN code. Six digits not starting zero, and only when it is being
    # given as an address — a bare six-digit number is far too common to mask.
    ("pin_code", re.compile(
        r"\b(?:pin\s*code|pincode|pin)\b\D{0,10}([1-9]\d{5})\b", re.IGNORECASE)),
]

# The categories where even the guardrail's own reporting must not echo a value.
SENSITIVE = {"aadhaar", "pan"}


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
            return self._ok("No personal data detected.", masked_text=ctx.text,
                            redactions={})

        detail = ", ".join(f"{k.replace('_', ' ')}×{v}" for k, v in counts.items())
        # Aadhaar and PAN escalate the finding: they identify the person
        # everywhere, not just to this bank.
        severity = "block" if (SENSITIVE & set(counts)) else "warn"
        note = ("" if severity == "warn" else
                " Aadhaar/PAN must never be captured on a recorded line.")
        return self._fail(
            f"Masked {total} item(s): {detail}.{note}",
            severity=severity,
            masked_text=masked,
            redactions=counts,
        )
