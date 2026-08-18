"""Clarity guardrail — does the reply say something the customer can act on?

Every other guardrail in this system protects the firm: that the advice is
grounded, that it is not harmful, that a human signed it off. This one protects
the customer's *understanding*, and it exists because that is also a rule. The
Consumer Duty's consumer understanding outcome (PRIN 2A.5) requires
communications a customer can act on, tailored where the customer is vulnerable —
and the people who parse a jargon-heavy sentence worst are precisely the ones
this system exists to protect.

The failure it catches is specific and, left alone, constant: policy is written
for handlers, so a drafted reply drifts into policy voice. It talks *about* "the
customer" instead of *to* you. It says "forbearance" and "signposting". It
recites a clause instead of making an offer. None of that is caught by grounding
or by the prohibited-action check — the advice can be perfectly correct and
still be unusable by the person it is for.

Five checks, each mapping to a way a reply fails a listener:

| Check | Why it matters on a call |
|---|---|
| Third-person voice | "the customer" means the handler is reading a file aloud |
| Unexplained jargon | industry terms and rulebook codes stop the listener dead |
| A concrete offer | "we have options available" gives the customer nothing to say yes to |
| Sentence length | this is *spoken*; a 40-word sentence cannot be followed by ear |
| Overall length | a reply longer than a breath is a reply that loses the point |

Severity is `warn`, not `block`. The reply is a draft a human handler sends or
edits, and a handler is perfectly capable of fixing a clumsy sentence — what they
cannot do is notice a problem nobody showed them. Blocking would also leave the
customer with silence, which is worse than an awkward sentence.
"""
from __future__ import annotations

import re

from ..agents.reply import THIRD_PERSON_MARKERS, UNEXPLAINED_JARGON
from ..utils.types import GuardrailResult
from .base import Guardrail, GuardrailContext

# A spoken sentence longer than this is hard to follow by ear, especially for
# someone distressed or with a capability-related need.
MAX_SENTENCE_WORDS = 28
# A whole reply longer than this stops being an answer and becomes a briefing.
MAX_REPLY_WORDS = 90

# Policy reference codes (VP-J1, VP-H4). These are filing labels; to a customer
# they are noise, and quoting one signals the handler is reading from a script.
_POLICY_CODE = re.compile(r"\bVP-[A-Z]\d+\b")

# "We may be able to look at some options" — hedging with nothing behind it.
_EMPTY_HEDGE = re.compile(
    r"\b(?:some options|various options|a number of options|certain measures|"
    r"appropriate support|relevant support)\b", re.IGNORECASE)

# Evidence that the reply actually offers something the customer can accept.
_CONCRETE_OFFER = re.compile(
    r"\b(?:I can|I'll|I will|we can|would you like|shall I|let me)\b", re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


class ClarityGuardrail(Guardrail):
    """Checks a customer-facing reply for comprehensibility."""

    name = "customer_clarity"
    severity = "warn"

    def check(self, ctx: GuardrailContext) -> GuardrailResult:
        reply = (ctx.reply or "").strip()
        if not reply:
            return self._ok("No customer-facing reply to check.")

        lowered = reply.lower()
        issues: list[str] = []
        details: dict[str, list[str]] = {}

        # 1. Is the handler talking about the customer rather than to them?
        third_person = [m for m in THIRD_PERSON_MARKERS if m in lowered]
        if third_person:
            issues.append("speaks about the customer instead of to them "
                          f"({', '.join(repr(m) for m in third_person[:3])})")
            details["third_person"] = third_person

        # 2. Industry terms and rulebook codes.
        jargon = [term for term in UNEXPLAINED_JARGON
                  if re.search(r"\b" + re.escape(term) + r"\b", lowered)]
        codes = _POLICY_CODE.findall(reply)
        if jargon or codes:
            found = jargon + codes
            issues.append(f"uses terms the customer will not know ({', '.join(found[:4])})")
            details["jargon"] = found

        # 3. Does it actually offer anything?
        if not _CONCRETE_OFFER.search(reply):
            issues.append("makes no concrete offer — nothing for the customer to accept")
            details["no_offer"] = [reply[:80]]
        elif _EMPTY_HEDGE.search(reply):
            issues.append("offers 'options' without naming any")
            details["vague"] = _EMPTY_HEDGE.findall(reply)

        # 4. Sentences too long to follow by ear.
        long_sentences = [s for s in _sentences(reply) if len(s.split()) > MAX_SENTENCE_WORDS]
        if long_sentences:
            issues.append(f"{len(long_sentences)} sentence(s) too long to follow when spoken")
            details["long_sentences"] = [s[:70] + "…" for s in long_sentences[:2]]

        # 5. Overall length.
        word_count = len(reply.split())
        if word_count > MAX_REPLY_WORDS:
            issues.append(f"too long for one turn ({word_count} words)")
            details["word_count"] = [str(word_count)]

        if issues:
            return self._fail(
                "Reply may be hard for the customer to act on: " + "; ".join(issues) + ".",
                severity="warn",
                issues=issues,
                **details,
            )
        return self._ok(
            f"Reply is direct, concrete and plain ({word_count} words).",
            word_count=word_count,
        )
