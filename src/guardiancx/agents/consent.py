"""Consent to record — the first thing said on every call.

A call cannot be recorded, transcribed or analysed until the customer has been
told it will be and has agreed. That is not a nicety: recording a call without
consent processes personal data with no lawful basis, and on this system the
recording feeds a vulnerability assessment that gets written to an evidence
store — a special-category inference the customer never agreed to.

So the call opens with the request, and nothing enters the pipeline until it is
answered. The customer's answer is classified by an agent rather than matched
against a word list, because people do not answer yes-or-no questions with yes or
no. Real answers include "go on then", "I'd rather you didn't", "do you have to?",
"I suppose so", "what for?" — and getting "I'd rather you didn't" wrong is not a
small error.

Three outcomes:

* **granted** — the call proceeds and every turn is assessed as normal.
* **refused** — the call stops. Not "continues without recording": this system's
  entire function is to analyse the conversation, so there is nothing lawful left
  for it to do. The customer is told how to get help another way.
* **unclear** — ask once more, plainly. A customer who asks "what for?" is
  entitled to an answer before deciding, and a system that reads a question as
  consent is helping itself.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from ..services.claude_client import EFFORT_REPLY, get_claude
from .prompts import CONSENT_SYSTEM

# The opening line. Short, specific about what is recorded and why, and it asks a
# real question rather than announcing a decision already taken.
CONSENT_REQUEST = (
    "Hello, you're through to Pan Indian Bank. Before we start — I record and make notes "
    "on our calls so I can get you the right support and you don't have to repeat "
    "yourself later. Is that alright with you?"
)

# What is said when the customer declines. It closes the call without sulking,
# and leaves a route that does not involve being recorded.
CONSENT_DECLINED = (
    "That's absolutely fine, and thank you for telling me. I'll end the call here "
    "rather than record it. You can write to us, or come into any branch, and we'll "
    "help you there — you won't have to explain any of this twice."
)

CONSENT_CLARIFY = (
    "Of course — the notes stay on your account and are only seen by the team "
    "helping you. Would you like me to go ahead?"
)

# After several asks with no answer either way. Not a refusal, and worded so it
# does not read as one — the customer gets the routes that do not need consent.
CONSENT_EXHAUSTED = (
    "I don't want to keep asking. I can't take you through the account without "
    "recording the call, but you can visit any branch or write to us and we'll "
    "help you there. Is there anything else I can do right now?"
)


class ConsentState(str, Enum):
    PENDING = "pending"      # asked, waiting for an answer
    GRANTED = "granted"
    REFUSED = "refused"
    UNCLEAR = "unclear"      # asked again


class ConsentDecision(BaseModel):
    state: ConsentState
    rationale: str = ""
    source: str = "heuristic"

    @property
    def may_proceed(self) -> bool:
        return self.state is ConsentState.GRANTED

    @property
    def call_over(self) -> bool:
        return self.state is ConsentState.REFUSED


_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["granted", "refused", "unclear"]},
        "rationale": {"type": "string"},
    },
    "required": ["answer", "rationale"],
    "additionalProperties": False,
}

# The fallback. Refusal patterns are checked first and drawn wider than
# agreement, because the cost of the two errors is not symmetric: treating a
# refusal as consent records someone against their wishes, while treating
# consent as unclear merely asks again.
_REFUSAL = re.compile(
    # Every alternative here has to be *about the recording*. A bare "don't" was
    # in this pattern once, and it read "I don't have the money to repay" as a
    # refusal — so a customer explaining she could not pay her home loan was told
    # the call was being ended. Generic negation is not a refusal; the sentence
    # has to decline the thing that was asked.
    #
    # "no" keeps its lookahead because the commonest ways to *agree* in English
    # contain it: "sure, no problem", "no worries", "no bother".
    r"\b(?:no(?!\s+(?:problem|worries|bother|trouble|issue|objection))|nope|nah)\b"
    r"|(?:please\s+)?don'?t\s+(?:record|tape|do\s+that|make\s+notes)"
    r"|rather\s+(?:you\s+)?(?:didn'?t|not\b)"
    r"|i'?d\s+rather\s+not"
    r"|not\s+(?:happy|comfortable|okay|ok)\s+(?:with|about)"
    r"|\brefuse\b|\bi\s+object\b"
    r"|without\s+(?:the\s+)?record"
    r"|stop\s+recording|turn\s+(?:it|that)\s+off",
    re.IGNORECASE,
)
_AGREEMENT = re.compile(
    r"\b(?:yes|yeah|yep|yup|sure|okay|ok|fine|alright|all right|go ahead|go on|"
    r"that'?s fine|of course|carry on|please do|i suppose|if you must|"
    r"do what you need|"
    # The "no"-shaped agreements, listed explicitly so they are read as consent
    # rather than falling through to "unclear" and asking the customer twice.
    r"no problem|no worries|no bother|no trouble|no objection)\b",
    re.IGNORECASE,
)
_QUESTION = re.compile(r"\b(?:why|what for|who|how long|do you have to|is it necessary)\b|\?",
                       re.IGNORECASE)


def _heuristic(text: str) -> ConsentDecision:
    if _REFUSAL.search(text):
        return ConsentDecision(state=ConsentState.REFUSED,
                               rationale="Declined in the answer.")
    if _QUESTION.search(text) and not _AGREEMENT.search(text):
        return ConsentDecision(state=ConsentState.UNCLEAR,
                               rationale="The customer asked a question rather than answering.")
    if _AGREEMENT.search(text):
        return ConsentDecision(state=ConsentState.GRANTED, rationale="Agreed in the answer.")
    return ConsentDecision(state=ConsentState.UNCLEAR,
                           rationale="No clear yes or no in the answer.")


def classify(text: str) -> ConsentDecision:
    """Read the customer's answer to the recording request."""
    if not text or not text.strip():
        return ConsentDecision(state=ConsentState.UNCLEAR, rationale="Nothing said.")

    llm = get_claude()
    result = llm.structured(CONSENT_SYSTEM, f"The customer replied: {text!r}",
                            _SCHEMA, max_tokens=200, effort=EFFORT_REPLY)
    if not result:
        return _heuristic(text)

    try:
        state = ConsentState(str(result.get("answer", "unclear")))
    except ValueError:
        return _heuristic(text)

    decision = ConsentDecision(state=state, rationale=str(result.get("rationale", "")),
                               source=llm.provider)
    # The stricter reading wins. If the wording refuses and the model read consent,
    # the refusal stands — this is the one classification where being wrong in the
    # permissive direction is not recoverable.
    if state is ConsentState.GRANTED and _REFUSAL.search(text) and not _AGREEMENT.search(text):
        return ConsentDecision(state=ConsentState.REFUSED, source=decision.source,
                               rationale="Wording declines; the stricter reading is taken.")
    return decision


# How many times the question may be re-asked before the call is closed. Only
# an explicit refusal ends it sooner.
MAX_CONSENT_ASKS = 3


def response_for(decision: ConsentDecision, asks_so_far: int = 0) -> Optional[str]:
    """What the handler says next, given the answer.

    A customer who ignores the question and starts talking about their problem
    has **not** refused. Treating a second non-answer as a refusal hung up on a
    woman explaining she could not repay her home loan — she was engaging with
    the call, just not with the procedural question in front of it. Only an
    explicit "no" ends the call early; everything else gets asked again, and the
    asking acknowledges whatever she actually said.
    """
    if decision.state is ConsentState.REFUSED:
        return CONSENT_DECLINED
    if decision.state is ConsentState.GRANTED:
        return None      # the conversation proper begins
    if asks_so_far + 1 >= MAX_CONSENT_ASKS:
        return CONSENT_EXHAUSTED
    return CONSENT_CLARIFY
