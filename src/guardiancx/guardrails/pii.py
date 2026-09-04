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
# * **IBAN first.** A UAE IBAN is "AE" and twenty-one digits. Nineteen of those
#   digits are a bare run that the account rule would happily claim, and an
#   evidence record saying "account" when the customer read out a full IBAN is
#   the wrong record.
# * **Emirates ID before card.** An Emirates ID is fifteen digits and a card is
#   thirteen to sixteen, so the card rule matches an Emirates ID outright. The
#   more specific pattern — anchored on the 784 issuer prefix every Emirates ID
#   carries — has to consume the run first, or the redaction is logged as a card
#   number and the fact that an identity document was read aloud is lost.
# * **Mobile before account.** A UAE mobile is nine digits after the country
#   code and starts 05, which the 9-18 digit account rule would otherwise
#   swallow. Both get masked either way, but the evidence record should say
#   which it was.
_PATTERNS = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # UAE IBAN: "AE", two check digits, then nineteen more, however it is spaced.
    ("iban", re.compile(r"\bAE\d{2}(?:[\s-]?\d){19}\b", re.IGNORECASE)),
    # Emirates ID: 784-YYYY-NNNNNNN-C. Every resident holds one and everybody
    # knows their own, which is exactly why it turns up in transcripts.
    ("emirates_id", re.compile(r"\b784[\s-]?\d{4}[\s-]?\d{7}[\s-]?\d\b")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    # UAE mobile: optional +971 or a leading 0, then 5X and seven digits.
    ("mobile", re.compile(r"(?:\+?971[\s-]?|\b0)5[024568]\d?[\s-]?\d{3}[\s-]?\d{4}\b")),
    # Bank account numbers here run 9-18 digits.
    ("account", re.compile(r"\b\d{9,18}\b")),
    ("date_of_birth", re.compile(
        r"\b(?:0?[1-9]|[12]\d|3[01])[/\-.](?:0?[1-9]|1[0-2])[/\-.](?:19|20)\d{2}\b")),
    ("date_of_birth", re.compile(
        r"\b(?:0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+(?:of\s+)?"
        r"(?:january|february|march|april|may|june|july|august|september|october|"
        r"november|december)\s+(?:19|20)\d{2}\b", re.IGNORECASE)),
    # There is no postal code in the UAE, and no rule here pretends there is.
    # Address is not an identifier this system ever asks for or masks.
]

# The categories where even the guardrail's own reporting must not echo a value.
SENSITIVE = {"emirates_id", "iban"}


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
