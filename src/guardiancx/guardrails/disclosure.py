"""Disclosure guardrail — no identifier leaves the ledger unmasked.

The Account Access agent decides *entitlement*: whose money is this, and may this
caller be told about it. That decision is a model's, and models can be talked
round. A bereaved caller pressing three times, a caller who says "I've got power
of attorney" without having it, a prompt-injection line buried in a transcript —
each is a plausible route to a model concluding that disclosure is fine.

So entitlement is checked twice, by two different kinds of reasoning. This
guardrail does not reason about who deserves what. It reads the drafted reply and
asks one question: **does this contain a number that exists in the ledger?** If it
does, the reply was read out of the book rather than composed from the masked
view, and it is blocked — regardless of how good the reason was.

That is a deliberately dumb check, and its dumbness is the point: it cannot be
argued with, and it fails closed. It catches:

* a third party's account number or sort code;
* the caller's *own* full account number, which is never spoken either — a
  handler confirms an account by its last four digits;
* a date of birth, Emirates ID, phone number or email read back from the file.

Severity is `block`. Unlike clarity, this is not a matter of the customer being
inconvenienced by an awkward sentence: a disclosure cannot be taken back once it
has been said aloud on a recorded line.
"""
from __future__ import annotations

import re

from ..finance.accounts import all_identifiers
from ..utils.types import GuardrailResult
from .base import Guardrail, GuardrailContext


def _normalise(text: str) -> str:
    """Digits-only view, so 44710021 is found in '4471 0021' and '4471-0021'."""
    return re.sub(r"[\s\-]", "", text)


# Spoken digit sequences are caught by the live-PII layer on the way in; here we
# are looking at what the *system* is about to say, which is written text.
_MASKED_FORM = re.compile(r"ending\s+\d{4}|\*{2}-\*{2}-\d{2}", re.IGNORECASE)


class DisclosureGuardrail(Guardrail):
    """Blocks a reply containing any identifier from the account book."""

    name = "data_disclosure"
    severity = "block"

    def check(self, ctx: GuardrailContext) -> GuardrailResult:
        reply = (ctx.reply or "").strip()
        if not reply:
            return self._ok("No customer-facing reply to check.")

        haystack = _normalise(reply)
        lowered = reply.lower()
        found: list[str] = []

        for identifier in all_identifiers():
            if not identifier:
                continue
            # Emails read naturally; numbers need the normalised form.
            if "@" in identifier or " " in identifier.strip():
                if identifier.lower() in lowered:
                    found.append(identifier)
            elif _normalise(identifier) and _normalise(identifier) in haystack:
                found.append(identifier)

        if found:
            # Report the kind of thing leaked, never the value itself — this
            # detail is written to the evidence log.
            kinds = sorted({("email" if "@" in f else
                             "IBAN" if re.match(r"^AE\d{2}", f, re.IGNORECASE) else
                             "Emirates ID" if re.match(r"^784-", f) else
                             "date of birth" if re.match(r"^\d{4}-\d{2}-\d{2}$", f) else
                             "phone number" if f.startswith("+971") else
                             "account number") for f in found})
            return self._fail(
                "Reply contains customer data that must not be read out "
                f"({', '.join(kinds)}). Confirm an account by its last four digits instead.",
                severity="block",
                leaked_kinds=kinds,
                count=len(found),
            )

        # A refusal that was overridden: the agent said no, the reply said yes.
        if ctx.account_refused and not self._reads_as_refusal(lowered):
            return self._fail(
                "The account request was refused as third-party data, but the reply does "
                "not decline it. A sympathetic request is still a refusal.",
                severity="block",
                overridden_refusal=True,
            )

        detail = "No account identifiers disclosed."
        if _MASKED_FORM.search(reply):
            detail = "Account referred to by its last four digits only."
        return self._ok(detail)

    @staticmethod
    def _reads_as_refusal(lowered: str) -> bool:
        return any(phrase in lowered for phrase in (
            "not able to", "can't go through", "cannot go through", "i'm not able",
            "not allowed", "can't share", "cannot share", "can't give you",
            "cannot give you", "can't discuss", "cannot discuss", "in someone else's name",
            "isn't in your name", "not in your name",
        ))
