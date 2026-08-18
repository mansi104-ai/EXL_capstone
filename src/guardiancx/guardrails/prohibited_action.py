"""Prohibited-action guardrail — the finance-specific output check.

The guidance prompt tells the model what it must not propose on each journey.
That is a request. This is the guarantee.

Some proposals are wrong in ways that no amount of confidence or grounding
redeems, because the harm comes from the action itself rather than from the
reasoning behind it:

* offering credit, a limit increase, or a consolidation loan to a customer who
  has just disclosed gambling harm;
* pressing for a payment that would leave essential bills unpaid;
* asking a customer reporting a scam to move money to a "safe account" — the
  exact instruction the scammer gave them;
* asking for a full card number, PIN or password.

Each is a deterministic pattern check against the drafted advice, scoped to the
journey the Financial Context agent identified. A match blocks the recommendation
and routes it to a human. The check is cheap, explainable, and — unlike the model
— cannot be talked out of its position.
"""
from __future__ import annotations

import re

from ..finance.taxonomy import Journey
from ..utils.types import GuardrailResult
from .base import Guardrail, GuardrailContext

# (journey or None for "any", compiled pattern, what it means)
_RULES: list[tuple[object, re.Pattern, str]] = [
    (Journey.GAMBLING_HARM,
     re.compile(r"\b(?:additional|further|more|new)\s+(?:credit|borrowing|lending)\b"
                r"|\blimit\s+increase\b|\bincrease\s+(?:the\s+|their\s+)?(?:credit\s+)?limit\b"
                r"|\bconsolidation\s+loan\b|\boffer\s+(?:them\s+)?a\s+(?:new\s+)?"
                r"(?:credit\s+card|loan|overdraft)\b", re.IGNORECASE),
     "offers credit to a customer disclosing gambling harm"),

    (Journey.AFFORDABILITY_SHOCK,
     re.compile(r"\bconsolidation\s+loan\b|\btake\s+out\s+(?:a\s+)?(?:further|another)\s+loan\b"
                r"|\bincrease\s+(?:the\s+|their\s+)?(?:credit\s+)?limit\b", re.IGNORECASE),
     "offers further borrowing as a remedy for unaffordability"),

    (Journey.ARREARS_COLLECTIONS,
     re.compile(r"\b(?:threaten|warn\s+(?:them\s+)?(?:of|about)|proceed\s+(?:to|with))\s+"
                r"(?:legal\s+action|enforcement|default|court|a\s+ccj)\b"
                r"|\binsist\s+on\s+(?:the\s+)?full\s+payment\b"
                r"|\bdemand\s+(?:the\s+)?(?:full\s+)?(?:payment|balance)\b", re.IGNORECASE),
     "threatens enforcement or demands full payment during a forbearance conversation"),

    (Journey.FRAUD_SCAM,
     re.compile(r"\bsafe\s+account\b|\bmove\s+(?:the\s+|their\s+)?(?:money|funds)\s+to\b",
                re.IGNORECASE),
     "repeats the scammer's own instruction to move money"),

    (Journey.BEREAVEMENT_ESTATE,
     re.compile(r"\b(?:ask|require|get)\s+(?:them|the\s+customer)\s+to\s+"
                r"(?:repeat|explain\s+again|call\s+back\s+and\s+explain)\b", re.IGNORECASE),
     "makes a bereaved customer repeat the disclosure"),

    # Applies on every journey.
    (None,
     re.compile(r"\b(?:full\s+card\s+number|card\s+number\s+in\s+full|pin\b|password\b"
                r"|security\s+password|full\s+long\s+number)\b", re.IGNORECASE),
     "asks for credentials that must never be requested"),
]


# Cues that invert the meaning of a match. The policy corpus is full of clauses
# that name a prohibited action precisely in order to forbid it — "avoid offering
# additional credit", "never ask the customer to move money to a safe account" —
# and guidance that quotes one is complying, not breaching. Without this, the
# guardrail blocks the correct advice and passes the wrong advice that happens to
# use different words.
_NEGATION = re.compile(
    r"\b(?:avoid|never|not|n't|no|don't|do not|must not|cannot|can't|refrain|"
    r"without|rather than|instead of|stop|decline|resist|discourage|"
    r"should not|shouldn't|won't|refuse)\b",
    re.IGNORECASE,
)
# How far back to look for a negation cue. Wide enough to catch "avoid offering
# any further credit", short enough not to be neutralised by an unrelated "not"
# earlier in the sentence.
_NEGATION_WINDOW = 60


def _is_negated(advice: str, start: int) -> bool:
    window = advice[max(0, start - _NEGATION_WINDOW):start]
    # Only look within the current clause — a negation before a full stop or
    # semicolon belongs to a different instruction.
    boundary = max(window.rfind("."), window.rfind(";"), window.rfind("!"))
    if boundary != -1:
        window = window[boundary + 1:]
    return bool(_NEGATION.search(window))


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _quoted_from_policy(segment: str, retrieved) -> bool:
    """Is this advice segment a verbatim quotation of a retrieved clause?

    The fallback guidance path — used whenever no LLM is configured — composes
    its adaptations by quoting the retrieved clauses directly. Those clauses
    routinely *name* a prohibited action in order to forbid it, or describe it as
    a scam pattern to watch for ("a 'safe account' request"). Pattern-matching a
    quotation of the policy against the policy's own prohibitions flags the
    correct advice as a breach.

    So a segment that appears verbatim in a clause the retrieval step returned is
    treated as quoted, not proposed. It is grounded by definition, and the policy
    — not this guardrail — is the authority on its own wording.
    """
    body = _normalise(segment)
    # Drop the "Clause title: " prefix the fallback prepends.
    if ":" in segment:
        body = _normalise(segment.split(":", 1)[1])
    if len(body) < 40:  # too short to attribute confidently
        return False
    return any(body in _normalise(chunk.text) for chunk in retrieved)


class ProhibitedActionGuardrail(Guardrail):
    name = "prohibited_action"
    severity = "block"

    def check(self, ctx: GuardrailContext) -> GuardrailResult:
        if ctx.recommendation is None:
            return self._ok("No recommendation to check.")

        rec = ctx.recommendation
        # Check only what the system proposed in its own words; verbatim policy
        # quotations are excluded. Segments are joined on a full stop so a
        # negation in one cannot reach across into the next.
        segments = [
            part for part in [rec.summary, *rec.adaptations]
            if part and not _quoted_from_policy(part, ctx.retrieved)
        ]
        advice = ". ".join(segments)
        journey = ctx.journey

        if not advice.strip():
            return self._ok("Guidance quotes retrieved policy verbatim — nothing proposed "
                            "beyond it.")

        breaches: list[str] = []
        for scope, pattern, meaning in _RULES:
            if scope is not None and scope is not journey:
                continue
            match = next(
                (m for m in pattern.finditer(advice) if not _is_negated(advice, m.start())),
                None,
            )
            if match:
                breaches.append(f"{meaning} ({match.group(0).strip()!r})")

        if breaches:
            return self._fail(
                "Recommendation proposes a prohibited action: " + "; ".join(breaches),
                severity="block",
                breaches=breaches,
                journey=journey.value if journey else "",
            )
        return self._ok(
            "No prohibited action proposed"
            + (f" for {journey.value}." if journey else "."),
            journey=journey.value if journey else "",
        )
