"""End-of-utterance (EOU) detection — deciding when the customer has finished.

Voice assistants normally end a turn on silence: stop hearing sound for N
milliseconds, assume the speaker is done, reply. On a bank's vulnerability line
that is actively harmful. The customer saying

    "my husband passed away in March and I…"

pauses there because they are trying to keep their composure, not because they
have finished. A silence-triggered agent talks over them. A customer reading out
a reference number pauses between digit groups. A customer with a cognitive
impairment pauses constantly. Silence is the worst available proxy for "finished
speaking" in exactly the population this system exists to protect.

So endpointing here is **semantic**: the decision is made on whether the words so
far form a finished thought, with silence as a secondary input rather than the
trigger. Three layers, cheapest first:

1. **Syntactic completeness** — a scored model over the tail of the transcript.
   Trailing conjunctions, prepositions, articles, fillers, dangling numbers and
   an absent sentence-final stop all push the score down; a clean sentence ending
   or a complete short answer pushes it up. Sub-millisecond, no dependencies.
2. **Adaptive silence** — the silence needed to commit the turn is derived from
   that score. A confidently complete sentence commits almost immediately; an
   obviously mid-sentence pause is given several seconds.
3. **A distress grace period** — when the caller is showing acoustic distress,
   every threshold is extended. A crying customer gets more room, not less.

An optional LLM adjudicator (`ENDPOINT_SYSTEM`) can be consulted for the genuinely
ambiguous middle band, but the loop never depends on it — a live call cannot wait
on a network round trip to decide whether to keep listening.
"""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel

from ..services.claude_client import EFFORT_REPLY, get_claude
from ..utils.logging import get_logger

log = get_logger("voice.endpointing")

# Base silence (ms) before a *clearly complete* utterance is committed.
MIN_SILENCE_MS = 450
# Silence before an utterance that looks unfinished is committed anyway — the
# ceiling that stops a hung turn.
MAX_SILENCE_MS = 4500
# Extra grace granted when the caller sounds distressed.
DISTRESS_GRACE_MS = 1500


class EndpointDecision(BaseModel):
    complete: bool
    completeness: float          # 0-1, how finished the thought looks
    required_silence_ms: int     # silence needed before committing this turn
    reason: str
    source: str = "semantic"     # "semantic" | "llm" | "timeout"

    @property
    def label(self) -> str:
        return "turn complete" if self.complete else "still speaking"


# --------------------------------------------------------------------------- #
# Lexical features
# --------------------------------------------------------------------------- #
# Ending on any of these means a clause is still open.
_DANGLING = {
    # conjunctions
    "and", "but", "or", "so", "because", "although", "though", "while", "whereas",
    "since", "unless", "until", "if", "when", "whether", "plus",
    # prepositions
    "about", "above", "across", "after", "against", "at", "before", "behind",
    "between", "by", "for", "from", "in", "into", "like", "of", "off", "on",
    "onto", "over", "than", "through", "to", "toward", "towards", "under", "with",
    "within", "without",
    # articles / determiners / possessives
    "a", "an", "the", "my", "your", "our", "their", "his", "her", "its", "this",
    "that", "these", "those", "some", "any", "no", "every", "each", "another",
    # auxiliaries and copulas left hanging
    "is", "are", "was", "were", "am", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "going",
    # pronouns that cannot end a sentence naturally
    "i", "we", "they", "he", "she", "it", "you",
}

# Fillers and hesitation markers — a turn ending on one is a thinking pause.
_FILLERS = {"um", "uh", "erm", "er", "ah", "hmm", "mm", "like", "well", "just",
            "sort", "kind", "you know", "i mean"}

# Short utterances that are genuinely complete on their own.
_COMPLETE_SHORT = {
    "yes", "yeah", "yep", "no", "nope", "okay", "ok", "right", "sure", "thanks",
    "thank you", "please", "correct", "that's right", "that's it", "go on",
    "sorry", "pardon", "hello", "goodbye", "bye", "i don't know", "i'm not sure",
}

# Phrases that promise more to come — a value is about to be dictated.
_ANNOUNCES_VALUE = re.compile(
    r"\b(?:my|the|our)\s+"
    r"(?:sort\s*code|account\s*number|card\s*number|reference|postcode|"
    r"date\s+of\s+birth|number|address|name|email)\s*"
    r"(?:is|are|was|would\s+be)?\s*$"
)

# The tail looks like a number still being read out in groups.
_TRAILING_DIGITS = re.compile(r"(?:\b\d{1,4}[\s-]+){1,}\d{0,3}$")

_SENTENCE_END = re.compile(r"[.!?…]\s*$")
_CLAUSE_END = re.compile(r"[,;:]\s*$")
_WORD = re.compile(r"[a-z']+")


def _tail_words(text: str, n: int = 3) -> list[str]:
    return _WORD.findall(text.lower())[-n:]


def score_completeness(transcript: str) -> tuple[float, str]:
    """Score how finished the transcript looks, 0 (mid-sentence) to 1 (done).

    Returns the score and the single most decisive reason, which is what the
    live console displays so the behaviour is legible rather than magical.
    """
    text = (transcript or "").strip()
    if not text:
        return 0.0, "nothing said yet"

    low = text.lower()
    words = _WORD.findall(low)
    if not words:
        return 0.0, "no words recognised"

    # A complete short answer is complete, whatever its length.
    stripped = re.sub(r"[^a-z' ]", "", low).strip()
    if stripped in _COMPLETE_SHORT:
        return 0.92, "complete short answer"

    # Strongest negative signals first — these are near-certainties.
    if _ANNOUNCES_VALUE.search(low):
        return 0.04, "about to dictate a value"
    if _TRAILING_DIGITS.search(text.strip()):
        return 0.10, "digits still being read out"

    last = words[-1]
    if last in _FILLERS or " ".join(words[-2:]) in _FILLERS:
        return 0.12, f"trailing filler ({last!r})"
    if last in _DANGLING:
        return 0.08, f"clause left open on {last!r}"
    if _CLAUSE_END.search(text):
        return 0.20, "ends mid-clause on punctuation"

    # Positive signal: an actual sentence-final stop.
    score = 0.80 if _SENTENCE_END.search(text) else 0.50
    reason = "sentence-final punctuation" if _SENTENCE_END.search(text) else "no terminator"

    # Length: a two-word fragment with no terminator is rarely a finished thought;
    # a long stretch of speech usually is, terminator or not.
    if len(words) <= 2 and score < 0.8:
        score -= 0.25
        reason = "too short to be a finished thought"
    elif len(words) >= 12:
        score += 0.12
        reason = reason if _SENTENCE_END.search(text) else "long, self-contained utterance"

    # A question is a finished thought that explicitly hands the turn over.
    if text.rstrip().endswith("?") or words[0] in {
        "can", "could", "would", "will", "do", "does", "did", "is", "are", "am",
        "what", "why", "how", "when", "where", "who", "which", "should",
    }:
        score += 0.10
        if text.rstrip().endswith("?"):
            reason = "question — turn handed over"

    return max(0.0, min(1.0, score)), reason


def required_silence_ms(completeness: float, distress: float = 0.0) -> int:
    """How long to keep listening, given how finished the utterance looks.

    The curve is quadratic rather than linear, and that matters. A linear map
    makes a plainly finished sentence wait over a second before the agent
    replies, which reads as a slow, unresponsive system; squaring the remaining
    uncertainty collapses the wait toward the floor as soon as the model is
    confident, while keeping the long patient window for anything ambiguous.
    Decisive when it is sure, patient when it is not.

    Grace for a distressed caller is added on top — the whole point of doing this
    semantically is that the people who pause most are the people who most need
    the room.
    """
    certainty = max(0.0, min(1.0, completeness))
    span = MAX_SILENCE_MS - MIN_SILENCE_MS
    base = MIN_SILENCE_MS + int(span * (1.0 - certainty) ** 2)
    return base + int(DISTRESS_GRACE_MS * max(0.0, min(1.0, distress)))


def detect(transcript: str, silence_ms: int = 0, distress: float = 0.0,
           use_llm: bool = False) -> EndpointDecision:
    """Decide whether this turn is finished.

    `silence_ms` is how long the speaker has been quiet. Pass 0 when there is no
    timing information (the text channel, or a push-to-talk recording the user
    has explicitly ended) — the decision then rests on semantics alone.
    """
    completeness, reason = score_completeness(transcript)

    # The ambiguous middle band is where an LLM adjudicator earns its latency —
    # and only there.
    if use_llm and 0.30 <= completeness <= 0.70:
        adjudicated = _llm_adjudicate(transcript)
        if adjudicated is not None:
            completeness, reason = adjudicated
            needed = required_silence_ms(completeness, distress)
            return EndpointDecision(
                complete=completeness >= 0.5 and silence_ms >= needed,
                completeness=round(completeness, 3),
                required_silence_ms=needed,
                reason=reason,
                source="llm",
            )

    needed = required_silence_ms(completeness, distress)

    # No timing information: semantics alone decides.
    if silence_ms <= 0:
        return EndpointDecision(
            complete=completeness >= 0.35,
            completeness=round(completeness, 3),
            required_silence_ms=needed,
            reason=reason,
            source="semantic",
        )

    if silence_ms >= MAX_SILENCE_MS + int(DISTRESS_GRACE_MS * distress):
        return EndpointDecision(
            complete=True,
            completeness=round(completeness, 3),
            required_silence_ms=needed,
            reason="held the floor in silence — committing the turn",
            source="timeout",
        )

    return EndpointDecision(
        complete=silence_ms >= needed,
        completeness=round(completeness, 3),
        required_silence_ms=needed,
        reason=reason,
        source="semantic",
    )


# --------------------------------------------------------------------------- #
# Optional LLM adjudication
# --------------------------------------------------------------------------- #
_SCHEMA = {
    "type": "object",
    "properties": {
        "complete": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["complete", "confidence", "reason"],
    "additionalProperties": False,
}


def _llm_adjudicate(transcript: str) -> Optional[tuple[float, str]]:
    from ..agents.prompts import ENDPOINT_SYSTEM

    llm = get_claude()
    if not llm.available:
        return None
    result = llm.structured(
        ENDPOINT_SYSTEM, f"Partial transcript: {transcript!r}", _SCHEMA,
        max_tokens=200, effort=EFFORT_REPLY,
    )
    if not result:
        return None
    confidence = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
    complete = bool(result.get("complete"))
    # Map the model's boolean+confidence onto the same 0-1 completeness scale the
    # rest of the module speaks in.
    score = 0.5 + 0.5 * confidence if complete else 0.5 - 0.5 * confidence
    return score, str(result.get("reason", "adjudicated by model"))
