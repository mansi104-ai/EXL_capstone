"""Live PII redaction — catching personal data *during* the call.

Written PII is easy: a sort code looks like `09-01-22` and a regex finds it. On a
call it looks like

    "my sort code is oh nine, oh one, double two"

which no written-form pattern will ever match. Speech-to-text engines render
spoken figures inconsistently — sometimes as digits, sometimes as words, usually
as a mixture — and the customer says them *before* anyone can stop them. By the
time a batch redactor sees the finished transcript, the raw value has already been
sent to the recogniser, held in a partial result, and potentially rendered on a
handler's screen.

This module closes that gap in two ways:

* **Spoken-form detection.** Runs of number-words ("oh nine oh one double two")
  are normalised to digits and matched against the same identifier rules the
  written path uses, then redacted *in place* in the original wording so the
  transcript still reads naturally.
* **Pre-emptive redaction.** When the transcript ends on a phrase that announces
  a value — "my account number is" — the redactor raises `expecting`, and the
  live console warns the handler and holds the next fragment before it is
  displayed or sent anywhere. The guard fires on the announcement, not on the
  disclosure.

Everything shares the pattern set in `guardrails.pii`, so a value is treated the
same whether it was typed or spoken.
"""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from ..guardrails.pii import mask_pii

# --------------------------------------------------------------------------- #
# Spoken numbers
# --------------------------------------------------------------------------- #
_DIGIT_WORDS: dict[str, str] = {
    "zero": "0", "oh": "0", "o": "0", "nought": "0", "naught": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90",
}
_REPEATERS = {"double": 2, "triple": 3, "treble": 3}

# A token that can form part of a spoken figure.
_NUM_TOKEN = re.compile(
    r"\b(?:" + "|".join(sorted(_DIGIT_WORDS, key=len, reverse=True))
    + r"|double|triple|treble|\d+)\b",
    re.IGNORECASE,
)

# Phrases that announce a value is coming next. Matched at the tail of a partial
# transcript, which is what makes the guard pre-emptive rather than reactive.
_ANNOUNCERS: list[tuple[str, re.Pattern]] = [
    ("sort_code", re.compile(r"\b(?:sort\s*code)\b[^.?!]{0,20}$", re.IGNORECASE)),
    ("account", re.compile(r"\b(?:account\s*(?:number|no\.?))\b[^.?!]{0,20}$", re.IGNORECASE)),
    ("card", re.compile(
        r"\b(?:card\s*(?:number|no\.?)|long\s*number|the\s+card\s+ends?)\b[^.?!]{0,20}$",
        re.IGNORECASE)),
    ("emirates_id", re.compile(
        r"\b(?:emirates\s+i\.?d\.?|\beid\b|identity\s+card|resident\s+id)\b"
        r"[^.?!]{0,20}$", re.IGNORECASE)),
    ("iban", re.compile(r"\b(?:iban|i\.b\.a\.n\.?)\b[^.?!]{0,20}$", re.IGNORECASE)),
    ("date_of_birth", re.compile(
        r"\b(?:date\s+of\s+birth|d\.?o\.?b\.?|born\s+on)\b[^.?!]{0,20}$", re.IGNORECASE)),
    ("security", re.compile(
        r"\b(?:pin|passcode|password|security\s+(?:code|number|answer)|cvv|"
        r"three\s+digits?\s+on\s+the\s+back)\b[^.?!]{0,20}$", re.IGNORECASE)),
    ("phone", re.compile(
        r"\b(?:mobile|phone|telephone)\s*(?:number|no\.?)\b[^.?!]{0,20}$", re.IGNORECASE)),
]

# How many digits a spoken run must reach before it is treated as an identifier
# on its own, with no announcing phrase. Six covers sort codes and dates; below
# that, ordinary conversation ("I've got two or three cards") gets caught.
_UNPROMPTED_DIGIT_FLOOR = 6
# With an announcing phrase in the same utterance, a much shorter run is enough.
_PROMPTED_DIGIT_FLOOR = 3


class LiveRedaction(BaseModel):
    """The result of feeding one transcript fragment through the redactor."""

    text: str                                    # safe to display, log and send
    redactions: dict[str, int] = Field(default_factory=dict)
    spoken_runs: int = 0                         # spoken figures caught
    expecting: Optional[str] = None              # a value is about to be spoken
    security_request: bool = False               # PIN/password territory

    @property
    def total(self) -> int:
        return sum(self.redactions.values())

    @property
    def clean(self) -> bool:
        return self.total == 0 and self.expecting is None

    def describe(self) -> str:
        if self.expecting:
            return f"holding — caller is about to give their {self.expecting.replace('_', ' ')}"
        if not self.total:
            return "no personal data in this turn"
        return "redacted " + ", ".join(f"{k.replace('_', ' ')}×{v}"
                                       for k, v in self.redactions.items())


def _expand_run(tokens: list[str]) -> str:
    """Turn spoken number tokens into the digit string they represent."""
    digits: list[str] = []
    repeat = 1
    for token in tokens:
        low = token.lower()
        if low in _REPEATERS:
            repeat = _REPEATERS[low]
            continue
        value = _DIGIT_WORDS.get(low, token if token.isdigit() else "")
        if not value:
            repeat = 1
            continue
        digits.append(value * repeat)
        repeat = 1
    return "".join(digits)


def _spoken_runs(text: str) -> list[tuple[int, int, str]]:
    """Find maximal runs of spoken-number tokens.

    Returns (start, end, digits) spans over the original text, so redaction can
    replace the words the customer actually used rather than a normalised form.
    """
    runs: list[tuple[int, int, str]] = []
    tokens = list(_NUM_TOKEN.finditer(text))
    if not tokens:
        return runs

    start_idx = 0
    while start_idx < len(tokens):
        end_idx = start_idx
        # Extend while the gap to the next token is only separators — spaces,
        # commas, dashes — which is how people punctuate a dictated figure.
        while end_idx + 1 < len(tokens):
            gap = text[tokens[end_idx].end():tokens[end_idx + 1].start()]
            if re.fullmatch(r"[\s,\-–—.]*", gap or ""):
                end_idx += 1
            else:
                break
        span = tokens[start_idx:end_idx + 1]
        digits = _expand_run([m.group(0) for m in span])
        if digits:
            runs.append((span[0].start(), span[-1].end(), digits))
        start_idx = end_idx + 1
    return runs


def _announced(text: str) -> Optional[str]:
    """The kind of value the tail of `text` announces, if any."""
    for label, pattern in _ANNOUNCERS:
        if pattern.search(text):
            return label
    return None


def redact(text: str, prompted: bool = False) -> LiveRedaction:
    """Redact one transcript fragment, written and spoken forms alike.

    `prompted` carries the announcement forward from the previous fragment, so a
    figure dictated across two partial results is still caught.
    """
    if not text or not text.strip():
        return LiveRedaction(text=text or "")

    announced = _announced(text)
    # An announcement anywhere in the fragment (not only at the tail) lowers the
    # bar for what counts as an identifier in the rest of it.
    has_context = prompted or announced is not None or bool(
        re.search(r"\b(sort\s*code|account\s*number|card\s*number|emirates\s+id|"
                  r"\beid\b|iban|date\s+of\s+birth|reference)\b", text, re.IGNORECASE)
    )
    floor = _PROMPTED_DIGIT_FLOOR if has_context else _UNPROMPTED_DIGIT_FLOOR

    # 1. Written forms first, using the shared pattern set. Order matters: the
    #    spoken-run detector sees only digit *sequences*, so run first it would
    #    swallow the "12 34 56" inside a National Insurance number or the digits
    #    of a date of birth and destroy the structure the specific patterns match
    #    on. Redacting the precise identifiers first leaves the generic sweep to
    #    catch what is genuinely just a dictated figure — and the evidence record
    #    then says "ni_number" rather than "some numbers".
    masked, counts = mask_pii(text)

    # 2. Spoken figures over what remains, right to left so earlier offsets stay
    #    valid as each run is replaced.
    spoken = 0
    for start, end, digits in reversed(_spoken_runs(masked)):
        if len(digits) < floor:
            continue
        masked = masked[:start] + "[SPOKEN_NUMBER_REDACTED]" + masked[end:]
        spoken += 1
    if spoken:
        counts["spoken_number"] = counts.get("spoken_number", 0) + spoken

    return LiveRedaction(
        text=masked,
        redactions=counts,
        spoken_runs=spoken,
        expecting=announced,
        security_request=announced == "security",
    )


class LivePIIRedactor:
    """Stateful redactor for a streaming transcript.

    Holds one bit of state — whether the previous fragment ended on an
    announcement — which is what lets a figure spoken across a partial-result
    boundary still be caught.
    """

    def __init__(self) -> None:
        self._expecting: Optional[str] = None
        self.total_redactions: dict[str, int] = {}

    @property
    def expecting(self) -> Optional[str]:
        return self._expecting

    def feed(self, fragment: str) -> LiveRedaction:
        result = redact(fragment, prompted=self._expecting is not None)
        self._expecting = result.expecting
        for key, count in result.redactions.items():
            self.total_redactions[key] = self.total_redactions.get(key, 0) + count
        return result

    def reset(self) -> None:
        self._expecting = None
        self.total_redactions = {}
