"""The shape of a real call.

A support call is not a chat window that happens to have a bank behind it. It has
a fixed spine, and every step of that spine exists because something goes wrong
without it:

    greeting → consent to record → who am I speaking to → are you who you say
    → how can I help → (the conversation) → close

The console used to start at "how can I help", with the caller already chosen
from a dropdown. That skipped the two steps where a real handler spends the first
ninety seconds of every call, and it meant the system knew who it was talking to
before anyone had said a word.

This module is that spine. Three things about it are worth stating plainly.

**Identification is not verification.** A caller giving a name is a claim. Until
they answer something only the account holder should know, they are a claim with
a name attached, and no account data may be discussed. The two stages are
separate because conflating them is exactly how social engineering works.

**Verification answers are never stored.** The customer will say their date of
birth out loud, and the live redactor will mask it out of the transcript — as it
should. Verification therefore happens on the raw utterance, before redaction,
and the only thing that survives is a boolean. The system records *that* the
caller verified, never *what* they said to do it.

**The pipeline runs from the moment consent is given, not from the moment
serving starts.** Customers routinely disclose the important thing while you are
still taking their details — "sorry, I'm all over the place since my husband
died" arrives during verification, not after it. A state machine that waits for
the serving stage to start listening misses the disclosure that matters most.
"""
from __future__ import annotations

import re
from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from ..finance.accounts import Customer, get_customer, list_customers
from ..services.claude_client import EFFORT_REPLY, get_claude
from ..utils.logging import get_logger
from .prompts import IDENTITY_SYSTEM

log = get_logger("agents.call_flow")

# How many times a caller may fail a check before the call stops trying. Real
# firms use two or three; the point of a limit is that an unlimited number of
# guesses is not a security question, it is a quiz.
MAX_ATTEMPTS = 3


class CallStage(str, Enum):
    CONSENT = "consent"          # asked to record, waiting for an answer
    IDENTIFY = "identify"        # taking the caller's name
    VERIFY = "verify"            # confirming they are that person
    SERVING = "serving"          # the conversation proper
    UNVERIFIED = "unverified"    # could not confirm — general help only
    CLOSED = "closed"            # consent refused, or the call ended


# What a handler must confirm before discussing an account. Ordered by how
# naturally they come up on a call.
#
# Both of these are *knowledge* checks against data the bank already holds, and
# neither is a credential. That distinction is the whole point of the list: a
# bank may ask a customer to confirm something it knows about them, and must
# never ask them to reveal something that grants access. There is no PIN here,
# no password, no OTP and no CVV, and `guardrails.credential_request` enforces
# that against the drafted line rather than trusting this list to stay clean.
#
# The last four digits of an Emirates ID replace the UK postcode this system
# started with. There is no postcode in the UAE — addresses are unstructured,
# and a "postcode" field in a UAE bank's verification flow is a tell that the
# system was built for somewhere else. The Emirates ID is held by every legal
# resident, the customer knows the number, and four digits is enough to
# corroborate an identity without being enough to impersonate one.
VERIFICATION_FIELDS = ["date_of_birth", "emirates_id_last4"]

FIELD_PROMPTS = {
    "date_of_birth": "their date of birth",
    "emirates_id_last4": "the last four digits of their Emirates ID",
}


class CallState(BaseModel):
    """Where this call has got to, and what is known about the caller."""

    stage: CallStage = CallStage.CONSENT
    customer_id: str = ""            # set once identified
    claimed_name: str = ""           # what they said their name was
    verified: list[str] = Field(default_factory=list)   # fields confirmed
    pending_field: str = ""          # what was just asked
    attempts: int = 0
    notes: list[str] = Field(default_factory=list)      # for the call record

    @property
    def identified(self) -> bool:
        return bool(self.customer_id)

    @property
    def is_verified(self) -> bool:
        """Verified enough to discuss the account."""
        return bool(self.verified)

    @property
    def may_discuss_account(self) -> bool:
        return self.stage is CallStage.SERVING and self.is_verified

    @property
    def open(self) -> bool:
        return self.stage not in (CallStage.CLOSED,)


# --------------------------------------------------------------------------- #
# Identification — turning "it's Margaret Hughes" into a customer record
# --------------------------------------------------------------------------- #
_NAME_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "gave_name": {"type": "boolean"},
    },
    "required": ["name", "gave_name"],
    "additionalProperties": False,
}

# Openings people actually use. The name is whatever follows.
_NAME_LEAD = re.compile(
    r"\b(?:it'?s|this is|my name'?s|my name is|i'?m|speaking to|you'?re speaking to|"
    r"name'?s|call me)\b[:,]?\s*",
    re.IGNORECASE,
)
_TITLES = re.compile(r"\b(?:mr|mrs|miss|ms|dr|prof)\b\.?\s*", re.IGNORECASE)
_NOT_NAMES = {"sorry", "sure", "yes", "no", "okay", "fine", "here", "calling",
              "afraid", "not", "just", "really", "trying", "phoning"}


def extract_name(text: str) -> str:
    """Pull a claimed name out of free speech.

    People do not answer "can I take your name" with a name. They answer with
    "it's Margaret Hughes", "Mrs Hughes speaking", "yeah, Margaret", or just
    "Margaret Hughes". The LLM handles the long tail; the pattern below handles
    the common cases and is what runs when no model is configured.
    """
    if not text or not text.strip():
        return ""

    llm = get_claude()
    result = llm.structured(IDENTITY_SYSTEM, f"The caller said: {text!r}",
                            _NAME_SCHEMA, max_tokens=150, effort=EFFORT_REPLY)
    if result and result.get("gave_name") and str(result.get("name", "")).strip():
        return str(result["name"]).strip()
    if result and not result.get("gave_name"):
        return ""

    return _extract_name_heuristic(text)


def _extract_name_heuristic(text: str) -> str:
    cleaned = _TITLES.sub("", _NAME_LEAD.sub("", text.strip()))
    cleaned = re.sub(r"\b(?:speaking|here|calling)\b\.?$", "", cleaned, flags=re.IGNORECASE)
    # Capitalised words are the strongest signal, but people type in lower case,
    # so fall back to the leading words once the filler is stripped.
    words = [w for w in re.findall(r"[A-Za-z'\-]+", cleaned)
             if w.lower() not in _NOT_NAMES]
    if not words:
        return ""
    capitalised = [w for w in words if w[:1].isupper()]
    chosen = capitalised[:3] if capitalised else words[:2]
    name = " ".join(chosen).strip(" '-")
    return name if len(name) >= 2 else ""


def match_customer(name: str) -> Optional[Customer]:
    """Find the customer record for a claimed name.

    Deliberately forgiving on the given name and strict on the surname: callers
    say "Maggie" for Margaret and "Bob" for Robert, but a surname mismatch means
    a different household. An ambiguous match returns nothing rather than a
    guess — picking one of two candidates is how a caller ends up looking at
    someone else's account.
    """
    if not name or not name.strip():
        return None

    wanted = [w for w in re.findall(r"[a-z']+", name.lower()) if len(w) > 1]
    if not wanted:
        return None

    matches = []
    for customer in list_customers():
        parts = [w for w in re.findall(r"[a-z']+", customer.name.lower())]
        surname = parts[-1] if parts else ""
        if surname in wanted:
            matches.append(customer)
            continue
        # A full-name match without the surname landing (e.g. only a first name
        # given) counts only if it is unique across the book.
        if len(wanted) == 1 and wanted[0] == parts[0]:
            matches.append(customer)

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Try to disambiguate on the given name before giving up.
        narrowed = [c for c in matches
                    if c.name.split()[0].lower() in wanted]
        if len(narrowed) == 1:
            return narrowed[0]
    return None


# --------------------------------------------------------------------------- #
# Verification — checking the claim, without keeping the answer
# --------------------------------------------------------------------------- #
_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}


def _parse_dates(text: str) -> list[date]:
    """Every date the caller might have just said, in any of the usual forms."""
    found: list[date] = []
    low = text.lower()

    # 12/03/1951, 12-3-1951, 12.03.51
    for match in re.finditer(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b", low):
        day, month, year = (int(g) for g in match.groups())
        found.extend(_as_dates(day, month, year))

    # 12th of March 1951 / March 12 1951
    for match in re.finditer(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]+)\s+(\d{2,4})\b", low
    ):
        day, month_name, year = match.group(1), match.group(2), match.group(3)
        if month_name in _MONTHS:
            found.extend(_as_dates(int(day), _MONTHS[month_name], int(year)))
    for match in re.finditer(r"\b([a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{2,4})\b", low):
        month_name, day, year = match.group(1), match.group(2), match.group(3)
        if month_name in _MONTHS:
            found.extend(_as_dates(int(day), _MONTHS[month_name], int(year)))
    return found


def _as_dates(day: int, month: int, year: int) -> list[date]:
    if year < 100:
        year += 1900 if year > 25 else 2000
    out = []
    for d, m in ((day, month), (month, day)):   # tolerate US-order input
        try:
            out.append(date(year, m, d))
        except ValueError:
            continue
    return out


# Digits as people say them out loud. A caller reading four digits off a card
# says "seven, four, double two" at least as often as "seven four two two", and a
# check that only understands numerals fails honest customers on the phone.
_SPOKEN_DIGITS = {
    "zero": "0", "oh": "0", "o": "0", "nought": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_MULTIPLIERS = {"double": 2, "triple": 3, "treble": 3}


def spoken_digits(text: str) -> str:
    """Every digit in the utterance, in order, numerals and words alike.

    "double two" becomes "22"; "seven four oh nine" becomes "7409". Order is
    preserved because that is the only thing that makes the result comparable to
    a stored number.
    """
    out: list[str] = []
    pending = 1
    for token in re.findall(r"[a-z]+|\d", (text or "").lower()):
        if token.isdigit():
            out.append(token * pending)
            pending = 1
        elif token in _MULTIPLIERS:
            pending = _MULTIPLIERS[token]
        elif token in _SPOKEN_DIGITS:
            out.append(_SPOKEN_DIGITS[token] * pending)
            pending = 1
        else:
            pending = 1
    return "".join(out)


def check_answer(customer: Customer, field: str, raw_text: str) -> bool:
    """Does the caller's answer match the record?

    Runs on the **raw** utterance, before redaction, because the redactor
    correctly masks dates of birth and identity numbers out of the transcript.
    Only the result of this call is ever kept — the answer itself is not
    returned, logged or stored anywhere.
    """
    if not raw_text or not raw_text.strip():
        return False

    if field == "date_of_birth":
        try:
            expected = date.fromisoformat(customer.date_of_birth)
        except ValueError:
            return False
        return expected in _parse_dates(raw_text)

    if field == "emirates_id_last4":
        wanted = customer.emirates_id_last4
        if not wanted:
            return False
        said = spoken_digits(raw_text)
        # A window scan rather than an endswith: a caller may read the whole
        # number, or say "it ends four one two three, I think". Either contains
        # the four digits in order, and neither ends with them.
        return any(said[i:i + 4] == wanted for i in range(max(len(said) - 3, 0)))

    return False


def next_verification_field(state: CallState) -> str:
    for field in VERIFICATION_FIELDS:
        if field not in state.verified:
            return field
    return ""


# --------------------------------------------------------------------------- #
# Advancing the call
# --------------------------------------------------------------------------- #
class TurnOutcome(BaseModel):
    """What this turn did to the call, and what the handler must do next."""

    state: CallState
    objective: str = ""          # what the handler's next line must achieve
    note: str = ""               # for the call record and the trace
    failed_check: bool = False


def advance(state: CallState, raw_text: str) -> TurnOutcome:
    """Move the call on by one customer turn.

    Consent is handled by `agents.consent` before this is reached; `advance` owns
    everything from "who am I speaking to" onwards.
    """
    stage = state.stage

    if stage is CallStage.IDENTIFY:
        return _identify(state, raw_text)
    if stage is CallStage.VERIFY:
        return _verify(state, raw_text)

    # Serving and unverified stages pass straight through — the pipeline, not
    # the state machine, decides what happens with the words.
    return TurnOutcome(state=state, objective="respond")


def _identify(state: CallState, raw_text: str) -> TurnOutcome:
    name = extract_name(raw_text)
    state.attempts += 1

    if not name:
        if state.attempts >= MAX_ATTEMPTS:
            state.stage = CallStage.UNVERIFIED
            state.attempts = 0
            return TurnOutcome(
                state=state, objective="proceed without the account",
                note="Caller did not give a usable name.")
        return TurnOutcome(state=state, objective="ask for their name again",
                           note="No name given.")

    state.claimed_name = name
    customer = match_customer(name)
    if customer is None:
        if state.attempts >= MAX_ATTEMPTS:
            state.stage = CallStage.UNVERIFIED
            state.attempts = 0
            return TurnOutcome(
                state=state, objective="proceed without the account",
                note=f"No record found for {name!r}.")
        return TurnOutcome(
            state=state, objective=f"say you cannot find {name} and ask them to spell it",
            note=f"No record matched {name!r}.")

    state.customer_id = customer.customer_id
    state.attempts = 0
    state.stage = CallStage.VERIFY
    state.pending_field = next_verification_field(state)
    state.notes.append(f"Identified as {customer.name}.")
    return TurnOutcome(
        state=state,
        objective=f"thank them by name and ask for {FIELD_PROMPTS[state.pending_field]}",
        note=f"Matched {customer.name}.")


def _verify(state: CallState, raw_text: str) -> TurnOutcome:
    customer = get_customer(state.customer_id)
    if customer is None:                       # defensive; identification set it
        state.stage = CallStage.UNVERIFIED
        return TurnOutcome(state=state, objective="proceed without the account")

    field = state.pending_field or next_verification_field(state)
    state.attempts += 1

    if check_answer(customer, field, raw_text):
        state.verified.append(field)
        state.attempts = 0
        state.pending_field = ""
        state.stage = CallStage.SERVING
        state.notes.append(f"Verified on {field.replace('_', ' ')}.")
        return TurnOutcome(
            state=state, objective="confirm they are verified and ask how you can help",
            note=f"Verified on {field.replace('_', ' ')}.")

    # Wrong answer. Offer the other field before giving up — a customer who has
    # just been bereaved may genuinely fumble a date, and refusing them on one
    # wrong answer is its own kind of harm.
    remaining = [f for f in VERIFICATION_FIELDS
                 if f not in state.verified and f != field]
    if state.attempts < MAX_ATTEMPTS and remaining:
        state.pending_field = remaining[0]
        return TurnOutcome(
            state=state, failed_check=True,
            objective=f"say that did not match, kindly, and ask instead for "
                      f"{FIELD_PROMPTS[state.pending_field]}",
            note="Check failed.")
    if state.attempts < MAX_ATTEMPTS:
        return TurnOutcome(
            state=state, failed_check=True,
            objective=f"say that did not match and ask again for {FIELD_PROMPTS[field]}",
            note="Check failed.")

    state.stage = CallStage.UNVERIFIED
    state.attempts = 0
    state.notes.append("Could not be verified.")
    return TurnOutcome(
        state=state, failed_check=True,
        objective="explain you cannot go through the account, and offer branch or post",
        note="Verification exhausted.")


def opening_objective(state: CallState) -> str:
    """What the handler must achieve on the turn that starts each stage."""
    return {
        CallStage.IDENTIFY: "ask who you are speaking to",
        CallStage.VERIFY: (f"ask for {FIELD_PROMPTS.get(state.pending_field, 'a security detail')}"
                           if state.pending_field else "ask a security question"),
        CallStage.SERVING: "ask how you can help",
        CallStage.UNVERIFIED: "offer general help without the account",
    }.get(state.stage, "respond")


# --------------------------------------------------------------------------- #
# The customer ending the call
# --------------------------------------------------------------------------- #
# A customer can hang up at any point, and they say so in ordinary words. Without
# this the console kept answering "let me bring up your account" to someone who
# had already said goodbye — the most obviously broken thing a phone system can
# do.
_GOODBYE = re.compile(
    r"^\W*(?:end\s+(?:the\s+)?call|hang\s+up|goodbye|good\s?bye|bye(?:\s+bye)?|"
    r"that'?s\s+(?:all|it|everything)|nothing\s+else|i'?m\s+done|"
    r"thanks?(?:\s+you)?[,.!\s]*(?:bye|goodbye|that'?s\s+all)|"
    r"no\s+thanks?,?\s*(?:that'?s\s+all|bye)?)\W*$",
    re.IGNORECASE,
)

# The handler's closing line, when the customer has ended it.
CALL_CLOSED = (
    "Of course. Thank you for calling, and please do ring back any time — "
    "everything we've discussed is on your account. Take care."
)


def wants_to_end(text: str) -> bool:
    """Has the customer just ended the call?

    Anchored to the whole utterance on purpose. "That's all I can afford" and
    "I'm done with this bank" both contain a closing phrase and neither is a
    goodbye, so a substring match would hang up on a customer mid-complaint.
    """
    return bool(_GOODBYE.match((text or "").strip()))
