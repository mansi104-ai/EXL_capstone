"""Wording-fidelity guardrail — "approved wording only" made checkable.

A treatment strategy that says the agent uses approved wording without deviation
is making a claim that is either enforced or decorative, and prompts alone make
it decorative. A model told to use only the wording it is given will do so almost
always, and the exception is not random: it drifts when the customer pushes, when
the situation is unusual, and when the conversation is emotionally difficult —
which is to say, on exactly the calls a reviewer will read.

So this checks the sentence rather than trusting the instruction. On a call
running under a strategy with `approved_wording_only`, every **commitment** the
reply makes must be approved wording from a retrieved clause, verbatim.

**Why commitments and not every sentence.** A reply is not only offers. It
acknowledges what the customer said, it states figures from their own account,
and it hands the turn back. Requiring every one of those to be pre-authored would
either freeze the call into an IVR script or force the strategy to authorise so
much boilerplate that the rule stops meaning anything. What actually binds the
bank — and what a customer acts on — is the sentence that says *I can do this
for you*. That is the sentence that must be approved, and it is identifiable
across languages because approved offers are written in exactly that form.

**Composition is allowed; invention is not.** The reply composer joins two
approved offers into one sentence and drops a repeated "I can" from the second,
because two sentences each opening "I can" read as a list rather than a choice.
That is still approved wording, so the check consumes a commitment sentence
fragment by fragment against the approved set rather than demanding a single
whole-string match. What it will not accept is a fragment that came from
somewhere else — which is the case that matters, because a model that has
invented an offer typically keeps the approved ones around it.

The check is scoped: no strategy, or a strategy that does not require approved
wording only, and it passes without comment. It is not a general style rule, and
applying it to an inbound call would block replies that were never meant to be
scripted.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from ..utils.language import normalise as normalise_language
from ..utils.types import GuardrailResult
from .base import Guardrail, GuardrailContext

# What a commitment looks like, per language. These are the openings the approved
# `Offer:` lines are themselves written in, which is what makes the test
# symmetrical: anything shaped like an offer is held to the offer standard.
_COMMITMENT: dict[str, re.Pattern] = {
    "en": re.compile(r"\b(?:i|we)\s+(?:can|could|will|'ll|shall)\b|\bi'?ll\b|\bwe'?ll\b",
                     re.IGNORECASE),
    # "yumkinuni" — I can / it is possible for me. Also "sa-" future forms.
    "ar": re.compile(r"يمكنني|يمكننا|بإمكاني|سأقوم|سوف\s+أ"),
    # "…kar sakti/sakta hoon" — I can do. Also "main …" with a modal.
    "ur": re.compile(r"سکتی\s+ہوں|سکتا\s+ہوں|سکتے\s+ہیں|کر\s+دوں\s+گی"),
    "hi": re.compile(r"सकती\s+हूँ|सकता\s+हूँ|कर\s+दूँगी"),
}

# Sentence terminators across the scripts in play. Urdu ends a sentence with the
# danda "۔"; Arabic uses the Latin full stop but its own question mark "؟".
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?۔؟])\s+")

# Connectives that may sit between two approved fragments in a composed
# sentence. Anything else between them means text was introduced.
_CONNECTIVE = re.compile(r"^(?:\s*(?:,|، |and|و|اور|، اور|, and)\s*)+", re.IGNORECASE)

# Characters stripped before comparison. Punctuation differs between the policy
# file and the spoken line — a trailing full stop is not a deviation.
_STRIP = re.compile(r"[.,!?;:۔،؟\"'“”‘’()\[\]{}\-—–…]+")


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", _STRIP.sub(" ", (text or "").lower())).strip()


def _variants(offers: Iterable[str]) -> list[str]:
    """Every acceptable normalised form of the approved offers, longest first.

    Longest first matters: the greedy consumer must try "i can freeze the
    interest and charges" before "i can freeze the interest", or it stops halfway
    through an approved sentence and reports the remainder as invented.
    """
    out: set[str] = set()
    for offer in offers:
        norm = _normalise(offer)
        if not norm:
            continue
        out.add(norm)
        # The composer drops a repeated "I can" from a second joined offer.
        for lead in ("i can ", "i could ", "i will ", "we can "):
            if norm.startswith(lead):
                out.add(norm[len(lead):])
    return sorted(out, key=len, reverse=True)


def _consumed_by_approved(sentence: str, variants: list[str]) -> bool:
    """Can this sentence be built entirely out of approved fragments?"""
    remaining = _normalise(sentence)
    if not remaining:
        return True
    guard = 0
    while remaining and guard < 12:
        guard += 1
        connective = _CONNECTIVE.match(remaining)
        if connective:
            remaining = remaining[connective.end():].strip()
            continue
        for variant in variants:
            if remaining.startswith(variant):
                remaining = remaining[len(variant):].strip()
                break
        else:
            return False
    return not remaining


def _is_commitment(sentence: str, language: str) -> bool:
    code = normalise_language(language)
    pattern = _COMMITMENT.get(code)
    if pattern is not None and pattern.search(sentence):
        return True
    # English commitments turn up inside calls held in other languages — a
    # fallback reply, or a customer who switched. Always check English too.
    return bool(_COMMITMENT["en"].search(sentence))


def _approved_for(ctx: GuardrailContext) -> list[str]:
    offers: list[str] = []
    for chunk in ctx.retrieved or []:
        getter = getattr(chunk, "offers_in", None)
        if callable(getter):
            found, _ = getter(ctx.language)
        else:
            found = list(getattr(chunk, "offers", []) or [])
        offers.extend(found)
    return offers


class ApprovedWordingGuardrail(Guardrail):
    """Blocks a commitment the strategy never authorised the agent to make."""

    name = "approved_wording"
    severity = "block"

    def check(self, ctx: GuardrailContext) -> GuardrailResult:
        strategy = _resolve_strategy(ctx.strategy)
        if strategy is None:
            return self._ok("No treatment strategy on this call.")
        if not strategy.approved_wording_only:
            return self._ok(f"{strategy.label()} does not require approved wording only.")

        reply = (ctx.reply or "").strip()
        if not reply:
            return self._ok("No reply to check.")

        approved = _approved_for(ctx)
        variants = _variants(approved)

        commitments = [s for s in _SENTENCE_SPLIT.split(reply)
                       if s.strip() and _is_commitment(s, ctx.language)]
        if not commitments:
            return self._ok("The reply makes no commitment to the customer.")

        if not variants:
            # A commitment with nothing approved behind it. This is the case
            # where a model has offered something the retrieval step never
            # supplied, and it is the most important one to catch.
            return self._fail(
                f"The reply commits the bank to something, but no approved wording "
                f"was retrieved for this call. {strategy.label()} permits approved "
                f"wording only.",
                strategy=strategy.name, unapproved=commitments[:3],
            )

        unapproved = [s.strip() for s in commitments
                      if not _consumed_by_approved(s, variants)]
        if unapproved:
            return self._fail(
                f"The reply makes {len(unapproved)} commitment(s) that are not "
                f"approved wording. {strategy.label()} permits approved wording "
                f"only, without deviation.",
                strategy=strategy.name,
                unapproved=unapproved[:3],
                approved_available=len(approved),
            )

        return self._ok(
            f"All {len(commitments)} commitment(s) are approved wording "
            f"under {strategy.label()}.",
            commitments=len(commitments),
        )


def _resolve_strategy(strategy):
    """Accept a strategy name, a strategy object, or nothing."""
    if strategy is None or strategy == "":
        return None
    if hasattr(strategy, "approved_wording_only"):
        return strategy
    from ..outbound.strategy import get_strategy

    return get_strategy(str(strategy))
