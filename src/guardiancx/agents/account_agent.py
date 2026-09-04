"""Account Access Agent — deciding what this caller may be told.

A vulnerability system that cannot answer "how much do I owe?" is an
observability tool, not a service. This agent connects the conversation to the
ledger: it works out what the customer is asking for, whose money it concerns,
and how much of the answer may be spoken.

The interesting half is the refusal. Consider a caller saying:

    "My husband died last week. I need his credit card number to settle it."

Every instinct in the conversation says help her. She is bereaved, the request is
reasonable-sounding, she is a genuine customer, and she may well end up
administering the estate. She is still not entitled to the number, and a handler
who reads it out has disclosed a third party's data. Firms leak precisely here,
because the request arrives wrapped in sympathy — which is why the decision is
made by an agent that reasons about *entitlement*, and then backstopped by a
deterministic guardrail that reasons about *digits*.

Three outcomes:

* **disclose** — the caller's own or joint account. Even then the figures are
  masked: "the one ending 0021", never the full number.
* **partial** — the account exists and is theirs, but the specific field asked
  for should not be read aloud (a full account number, a sort code).
* **refuse** — the data belongs to someone else. The agent says so plainly and
  offers the route that does exist, because "no" without a next step is what
  makes a grieving customer call back angry.

Primary path: the LLM with a strict schema, so the *reasoning* about entitlement
is a model decision rather than a keyword list. Fallback: a deterministic
resolver, so the refusal still happens with no model configured — this is not a
capability that may degrade quietly.
"""
from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from ..finance.accounts import (
    Account,
    Relationship,
    accounts_for,
    context_summary,
    get_customer,
    aed,
)
from ..finance.taxonomy import Product
from ..services.claude_client import EFFORT_ANALYSIS, get_claude
from ..utils.logging import get_logger
from .prompts import ACCOUNT_ACCESS_SYSTEM
from .state import AgentState

log = get_logger("agents.account")


class AccountRequest(BaseModel):
    """What the customer asked the ledger for, and what may be said back."""

    asked: bool = False                 # did they ask for account data at all?
    field: str = ""                     # balance | arrears | payment | number | transactions …
    subject: str = "self"               # self | joint | third_party | unknown
    decision: str = "none"              # disclose | partial | refuse | unavailable | none
    facts: list[str] = Field(default_factory=list)   # masked facts the handler may state
    refusal_reason: str = ""
    alternative: str = ""               # the route that does exist
    source: str = "heuristic"

    @property
    def refused(self) -> bool:
        return self.decision == "refuse"


_SCHEMA = {
    "type": "object",
    "properties": {
        "asked": {"type": "boolean"},
        "field": {
            "type": "string",
            "enum": ["balance", "limit", "arrears", "payment", "account_number",
                     "sort_code", "transactions", "statement", "other", "none"],
        },
        "subject": {"type": "string", "enum": ["self", "joint", "third_party", "unknown"]},
        "reasoning": {"type": "string"},
    },
    "required": ["asked", "field", "subject", "reasoning"],
    "additionalProperties": False,
}

# Fields that are never read aloud, even to the account's own holder. A handler
# confirms an account by its last four digits; the full number is not needed to
# have the conversation, and saying it puts it into the recording.
_NEVER_SPOKEN = {"account_number", "sort_code"}

# Phrases indicating the request concerns someone else. Used by the fallback
# resolver and as a floor under the model: if these fire and the model said
# "self", the stricter reading wins.
_THIRD_PARTY_CUES = re.compile(
    r"\b(?:my\s+(?:husband|wife|partner|mother|father|mum|dad|son|daughter|"
    r"brother|sister|late\s+\w+)(?:'s)?|his\s+|her\s+|their\s+|"
    r"someone\s+else|the\s+deceased)\b",
    re.IGNORECASE,
)
_SELF_CUES = re.compile(r"\b(?:my own|for me|mine|i owe|i have|do i)\b", re.IGNORECASE)

_FIELD_CUES: list[tuple[str, re.Pattern]] = [
    ("account_number", re.compile(
        r"\b(?:account|card|reference|policy)\s*(?:number|no\.?)\b|\bfull\s+number\b"
        r"|\blong\s+number\b|\bthe\s+numbers?\s+(?:for|on)\b", re.IGNORECASE)),
    ("sort_code", re.compile(r"\bsort\s*code\b", re.IGNORECASE)),
    # Ahead of "balance": "how much is my limit" is a question about the limit,
    # and the balance cue would otherwise claim it on "how much".
    ("limit", re.compile(r"\blimit\b|\bavailable\s+credit\b|\bcredit\s+available\b"
                         r"|\bhow\s+much\s+(?:can|could)\s+i\s+(?:spend|borrow)\b",
                         re.IGNORECASE)),
    ("arrears", re.compile(r"\barrears\b|\bbehind\b|\bmissed\b|\bcatch\s+up\b", re.IGNORECASE)),
    ("payment", re.compile(r"\bpayment\b|\bdirect debit\b|\bwhen is it due\b|\bdue\b",
                           re.IGNORECASE)),
    ("balance", re.compile(r"\bbalance\b|\bhow much\b|\bowe\b|\boutstanding\b|\bleft\b",
                           re.IGNORECASE)),
    ("transactions", re.compile(r"\btransactions?\b|\bwent out\b|\bspent\b|\bpaid out\b",
                                re.IGNORECASE)),
    ("statement", re.compile(r"\bstatement\b", re.IGNORECASE)),
]


# How customers name each product out loud. Nobody says "credit_card"; they say
# "my card", "the credit card", "my home loan". Recognising the product named is
# what lets the resolver tell *which* of the caller's accounts they mean from an
# account they do not hold at all — two very different answers.
#
# Ordered: the specific pattern must win over the general one, so "credit card"
# is not claimed by the bare "card".
_PRODUCT_WORDS: list[tuple[Product, re.Pattern, str]] = [
    (Product.CREDIT_CARD,
     re.compile(r"\bcredit\s*card\b|\bcard\b", re.IGNORECASE), "a credit card"),
    (Product.MORTGAGE,
     re.compile(r"\bmortgage\b|\bhome\s*loan\b|\bhousing\s*loan\b", re.IGNORECASE),
     "a home loan"),
    (Product.PERSONAL_LOAN,
     re.compile(r"\bpersonal\s*loan\b", re.IGNORECASE), "a personal loan"),
    (Product.CAR_FINANCE,
     re.compile(r"\bcar\s*(?:loan|finance)\b|\bauto\s*loan\b", re.IGNORECASE),
     "car finance"),
    (Product.OVERDRAFT,
     re.compile(r"\boverdraft\b", re.IGNORECASE), "an overdraft"),
    (Product.SAVINGS,
     re.compile(r"\bsavings?\b|\bfixed\s*deposit\b|\bfd\b", re.IGNORECASE),
     "a savings account"),
    (Product.CURRENT_ACCOUNT,
     re.compile(r"\bcurrent\s*account\b|\bchecking\b", re.IGNORECASE),
     "a current account"),
    (Product.PENSION,
     re.compile(r"\bpension\b", re.IGNORECASE), "a pension"),
    (Product.INSURANCE,
     re.compile(r"\binsurance\b", re.IGNORECASE), "an insurance policy"),
]


def named_product(text: str) -> tuple[Optional[Product], str]:
    """The product the caller named, and how to say it back to them."""
    for product, pattern, spoken in _PRODUCT_WORDS:
        if pattern.search(text):
            return product, spoken
    return None, ""


def _spoken_list(items: list[str]) -> str:
    """Read a short list the way a person would say it."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _facts_for(account: Account, field: str) -> list[str]:
    """Masked, speakable facts about one account.

    Figures go through `aed`, so a customer hears AED 8,650 rather than a bare
    number in no currency at all, and instalments are instalments.
    """
    plural = "s" if account.arrears_months != 1 else ""
    facts = [f"{account.label}, {account.masked_number}"]
    if field in ("balance", "other", "none", "statement"):
        if account.owed:
            facts.append(f"{aed(account.owed)} outstanding")
        else:
            facts.append(f"{aed(account.balance)} available")
    if field in ("arrears", "balance", "other") and account.arrears_months:
        facts.append(f"{account.arrears_months} instalment{plural} overdue, "
                     f"{aed(account.arrears_amount)} to bring it up to date")
    if field in ("payment", "arrears", "other", "none") and account.monthly_payment:
        facts.append(f"an instalment of {aed(account.monthly_payment)} due on "
                     f"{account.next_payment_date}")
    if field == "limit":
        if account.credit_limit:
            available = max(0.0, account.credit_limit - account.owed)
            facts.append(f"a limit of {aed(account.credit_limit)}")
            facts.append(f"{aed(available)} of that still available")
        else:
            facts.append("no credit limit — it isn't a card or an overdraft")
    if field == "transactions":
        for txn in account.transactions[:3]:
            facts.append(f"{txn.date}: {txn.description} {aed(abs(txn.amount))}")
    return facts


def _relevant_account(accounts: list[Account], text: str,
                      field: str = "other") -> Optional[Account]:
    """Pick the account the customer most likely means, or None if they named
    one they do not hold.

    A caller who names a product means that product. The previous version
    matched on `product.value.replace("_", " ")` — the literal string "credit
    card" — and when that missed it fell through to whichever account was in the
    most trouble. So "can you check my credit card limit", asked by a customer
    with no card, was answered with her joint home loan: three instalments overdue,
    AED 25,950 to bring it up to date. Fluent, confident, and about an account she
    had not mentioned.

    Answering the wrong account is a disclosure, not a near-miss. If the product
    named is not held, this returns None and the caller is told so.
    """
    if not accounts:
        return None

    wanted, _ = named_product(text)
    if wanted is not None:
        held = [a for a in accounts if a.product is wanted]
        if not held:
            return None
        return max(held, key=lambda a: (a.arrears_months, a.owed))

    low = text.lower()
    for account in accounts:
        if account.label.lower() in low:
            return account

    # No product named. For a field only some products can answer, prefer one
    # that can — "what's my limit" should not be answered from a savings account
    # just because it is the one in the most trouble.
    if field == "limit":
        with_limit = [a for a in accounts if a.credit_limit]
        if with_limit:
            return max(with_limit, key=lambda a: a.credit_limit)

    # Otherwise the one in the most trouble — that is what the call is about.
    return max(accounts, key=lambda a: (a.arrears_months, a.owed))


def _resolve(customer_id: str, text: str, field: str, subject: str,
             source: str) -> AccountRequest:
    """Turn a classified request into a disclosure decision. Deterministic."""
    own = accounts_for(customer_id)

    # Third-party is decided here, not by the model, and the stricter reading
    # always wins: if either the model or the cue pattern says someone else's
    # data is in play, it is refused.
    # Both signals — the model's reading and the phrase cues — only refuse when a
    # request is actually being made. "My husband passed away and I've missed two
    # instalments" mentions a third party and asks for nothing; refusing it told a widow
    # she could not discuss her own home loan, which is worse than useless.
    asking = bool(_REQUEST_SHAPE.search(text) and _DATA_NOUN.search(text))
    cues = bool(_THIRD_PARTY_CUES.search(text)) and not _SELF_CUES.search(text)
    if asking and (subject == "third_party" or cues):
        joint = [a for a in own if a.is_joint]
        alternative = (
            f"I can talk about the {joint[0].label.lower()} you hold together"
            if joint else
            "I can send our bereavement pack, which sets out exactly what's needed"
        )
        return AccountRequest(
            asked=True, field=field, subject="third_party", decision="refuse",
            refusal_reason=("That account is in someone else's name, so I'm not able to "
                            "go through it with you — even now."),
            alternative=alternative,
            source=source,
        )

    if not own:
        return AccountRequest(asked=True, field=field, subject="unknown", decision="none",
                              source=source)

    account = _relevant_account(own, text, field)

    # They named a product that is not on their file. Say so, and say what is —
    # a customer told only "I can't see that" concludes the bank has lost the
    # account, and rings back angrier.
    if account is None:
        _, spoken = named_product(text)
        held = _spoken_list([a.label.lower() for a in own])
        return AccountRequest(
            asked=True, field=field, subject=subject, decision="unavailable",
            refusal_reason=f"I can't see {spoken or 'that account'} on your name here.",
            alternative=f"What I do have for you is {held}" if held else "",
            source=source,
        )

    if field in _NEVER_SPOKEN:
        return AccountRequest(
            asked=True, field=field, subject=subject, decision="partial",
            facts=[f"{account.label}, {account.masked_number}"],
            refusal_reason=("I can't read the full number out, but I can confirm the "
                            "last four digits."),
            source=source,
        )

    relationship = Relationship.JOINT if account.is_joint else Relationship.SELF
    return AccountRequest(
        asked=True, field=field, subject=relationship.value, decision="disclose",
        facts=_facts_for(account, field), source=source,
    )


# The shape of a request, independent of what is being requested. Used so that a
# third-party ask is caught even when the field is one we have no cue for.
_REQUEST_SHAPE = re.compile(
    r"\b(?:can|could|would|will)\s+you\b|\b(?:give|tell|read|send|share|confirm|find|"
    r"look\s+up|get)\s+(?:me|us)\b|\bwhat(?:'s| is)\b|\bhow\s+much\b|\bi\s+need\b"
    r"|\bi\s+want\b|\bdo\s+you\s+have\b|\?",
    re.IGNORECASE,
)

# Words that mean "their financial information", however it is phrased.
_DATA_NOUN = re.compile(
    r"\b(?:account|card|balance|number|details?|statement|payments?|arrears|"
    r"transactions?|sort\s*code|pin|password|policy)\b",
    re.IGNORECASE,
)


def looks_like_third_party_request(text: str) -> bool:
    """Is this a request for somebody else's information?

    Deliberately independent of field classification. The original version only
    refused once it had recognised *which* field was being asked for, so "can you
    give me his credit card number" sailed through: "card number" was not in the
    cue list, so no field matched, so no request was registered, so nothing was
    refused. The safety-critical branch must not sit behind a lookup table.
    """
    if not _THIRD_PARTY_CUES.search(text) or _SELF_CUES.search(text):
        return False
    return bool(_REQUEST_SHAPE.search(text) and _DATA_NOUN.search(text))


def _heuristic_field(text: str) -> tuple[bool, str]:
    for name, pattern in _FIELD_CUES:
        if pattern.search(text):
            return True, name
    # No recognised field, but plainly a request for someone's data.
    if _REQUEST_SHAPE.search(text) and _DATA_NOUN.search(text):
        return True, "other"
    return False, "none"


def run(state: AgentState) -> AgentState:
    trace = state.setdefault("trace", [])
    customer_id = state.get("customer_id", "")

    if state.get("speaker") != "customer" or not get_customer(customer_id):
        state["account_request"] = AccountRequest()
        return state

    text = state.get("masked_text") or state.get("text", "")
    llm = get_claude()
    result = llm.structured(
        ACCOUNT_ACCESS_SYSTEM,
        f"Caller on file:\n{context_summary(customer_id)}\n\nThe caller said: {text}",
        _SCHEMA, max_tokens=400, effort=EFFORT_ANALYSIS,
    )

    # The third-party check runs first and independently of the model. If the
    # model is unavailable, rate-limited, or simply says "not a data request",
    # a request for someone else's information is still refused — this branch
    # must never depend on a classifier being reachable.
    if looks_like_third_party_request(text):
        _, field = _heuristic_field(text)
        request = _resolve(customer_id, text, field, "third_party",
                           result and llm.provider or "heuristic")
    elif result and result.get("asked"):
        request = _resolve(customer_id, text, str(result.get("field", "other")),
                           str(result.get("subject", "self")), llm.provider)
    elif result:
        request = AccountRequest(source=llm.provider)
    else:
        asked, field = _heuristic_field(text)
        request = (_resolve(customer_id, text, field, "unknown", "heuristic")
                   if asked else AccountRequest())

    state["account_request"] = request
    if request.asked:
        trace.append({
            "agent": "account_access",
            "summary": f"[{request.source}] {request.field} · {request.subject} · "
                       f"{request.decision}"
                       + (f" — {request.refusal_reason}" if request.refusal_reason else ""),
        })
    return state
