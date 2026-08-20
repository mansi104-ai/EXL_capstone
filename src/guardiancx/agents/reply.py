"""Composing what the customer actually hears.

Policy is written for handlers. Every clause in the knowledge base is an
instruction in the imperative, about a third party:

    "Express condolences and reassure the customer they will not need to
     repeat the bereavement disclosure to another team."

Read that aloud to a widow and you have said nothing to her at all — you have
recited her own file back at her. The same sentence, addressed to the person on
the phone, is:

    "I'm so sorry. You won't have to explain this again — I've noted it."

That transformation is this module's job, and it is not cosmetic. Under the
Consumer Duty's consumer understanding outcome (PRIN 2A.5) a firm must
communicate in a way its customers can act on, and must tailor that
communication where a customer is vulnerable. A reply the customer cannot parse
is a compliance failure, not a style problem — and the customers who parse it
worst are exactly the ones this system exists to protect.

Three things happen here:

* **Person and mood** — imperatives about "the customer" become offers to "you".
  "Offer a payment holiday" becomes "I can arrange a payment holiday for you".
* **Plain English** — the terms the industry uses for itself are replaced with
  the words a customer uses. "Forbearance" is not a word anyone has said to
  their bank unprompted.
* **Precision** — one clear offer and one clear next step, rather than the
  whole clause. A reply that lists five things is a reply that gets none of them
  done.

Where a retrieved clause carries approved `Offer:` wording, that wording is used
verbatim — a sentence a compliance reviewer signed off beats anything derived at
runtime, and the derivation exists for clauses that have not been given one yet.

When the caller has asked about their money, the answer leads: a customer who
asked what they owe should hear the figure before they hear an offer. And when
the Account Access agent has refused the request as somebody else's data, the
refusal leads and is never softened into a maybe — followed immediately by the
route that does exist, because a "no" with no next step is what makes a grieving
customer call back angry.

The composer runs whether or not an LLM is configured: with a model it builds
the prompt and the model writes the reply; without one it composes the reply
itself from the retrieved policy. Both paths are checked by the clarity
guardrail afterwards, because the model can slip into policy voice too.
"""
from __future__ import annotations

import re
from typing import Optional

from ..finance.taxonomy import FinancialContext, Journey
from ..services.claude_client import EFFORT_REPLY, get_llm
from ..utils.types import Driver, SentimentReading

# --------------------------------------------------------------------------- #
# Openers
#
# Chosen by *situation* rather than by whichever driver scored highest. A
# customer reporting a scam who happens to score on capability should not be
# told "I'll go at your pace" — it answers a question they did not ask, and it
# reads as condescension at the exact moment they feel foolish.
# --------------------------------------------------------------------------- #
_JOURNEY_OPENERS: dict[Journey, str] = {
    Journey.BEREAVEMENT_ESTATE: "I'm so sorry for your loss.",
    Journey.FRAUD_SCAM: "Thank you for telling me — you did the right thing by calling.",
    Journey.GAMBLING_HARM: "Thank you for telling me. That took courage, and I can help.",
    Journey.ARREARS_COLLECTIONS: "Thank you for letting me know. We can sort this out together.",
    Journey.FORBEARANCE_REQUEST: "Of course — let's find something that works for you.",
    Journey.AFFORDABILITY_SHOCK: "I'm sorry to hear that. Let's see what we can do to help.",
    Journey.THIRD_PARTY_ACCESS: "Of course, that's no problem at all.",
    Journey.COMPLAINT: "I'm sorry this has been difficult. Let me put it right.",
}

_DRIVER_OPENERS: dict[Driver, str] = {
    Driver.LIFE_EVENTS: "I'm very sorry to hear that.",
    Driver.HEALTH: "Thank you for telling me, and I'm sorry you're going through this.",
    Driver.RESILIENCE: "I understand — let's make this manageable for you.",
    Driver.CAPABILITY: "Of course. I'll keep this simple and go at your pace.",
}

_NEUTRAL_OPENER = "Thanks for letting me know."

# When the customer is in real distress, the opener acknowledges that before
# anything transactional. Nothing else in the reply changes.
_DISTRESS_OPENER = "I can hear this is really hard, and I'm glad you've called."


# --------------------------------------------------------------------------- #
# Plain English
#
# The left-hand column is what the industry calls it. The right-hand column is
# what a customer would say. Ordered longest-first so multi-word terms are
# replaced before their component words.
# --------------------------------------------------------------------------- #
_PLAIN_ENGLISH: list[tuple[str, str]] = [
    ("income-and-expenditure review", "a quick look at what's coming in and going out"),
    ("income and expenditure review", "a quick look at what's coming in and going out"),
    ("authorised push payment", "a scam where you were tricked into paying"),
    ("forbearance options", "ways we can ease your payments"),
    ("hardship and forbearance", "extra support"),
    ("payment concession", "a reduced payment"),
    ("breathing space", "a pause on interest and contact"),
    ("moratorium", "a pause"),
    ("forbearance", "support with your payments"),
    ("arrears activity", "collections contact"),
    ("collections activity", "letters and calls chasing payment"),
    ("collections process", "chasing you for payment"),
    ("affordability review", "a look at what you can afford"),
    ("affordability", "what you can afford"),
    ("signposting", "putting you in touch with"),
    ("signpost", "put you in touch with"),
    ("adaptation", "change"),
    ("adaptations", "changes"),
    ("the disclosure", "what you've told me"),
    ("this disclosure", "what you've told me"),
    ("a health flag", "a note"),
    ("health flag", "a note"),
    ("a vulnerability flag", "a note"),
    ("third-party mandate", "permission for someone to help you"),
    ("lasting power of attorney", "power of attorney"),
    ("specialist support team", "our specialist team"),
    ("specialist safeguarding team", "our specialist support team"),
    ("financial difficulty team", "our support team"),
    ("free debt advice", "free, independent debt advice"),
    ("statutory support", "any benefits you're entitled to"),
    ("income maximisation", "checking you're getting everything you're entitled to"),
    ("non-priority", "less urgent than your essential bills"),
    ("reimbursement process and its timescales", "how and when you get your money back"),
    ("reimbursement process", "how you get your money back"),
    ("reimbursement", "getting your money back"),
    ("recall", "try to get the payment back"),
    ("cooling-off period", "time to think it over"),
    ("gambling block", "block on gambling payments"),
    ("credit reference", "your credit file"),
    # Never name the relationship. A bereaved customer may have lost a wife, a
    # parent or a child, and guessing is worse than saying nothing.
    ("the estate is settled", "everything is sorted out"),
    ("the estate", "their estate"),
    ("the survivor", "you"),
    ("check understanding", "check you're happy with it"),
    ("the practical controls first:", ""),
    ("for this contact", ""),
]

# Longest first, always. Hand-ordering silently breaks when a term is added:
# "forbearance options" would otherwise fire inside "hardship and forbearance
# options" and leave a half-translated phrase.
_PLAIN_ENGLISH.sort(key=lambda pair: len(pair[0]), reverse=True)

# Terms that must never reach a customer, with no clean short substitute. If one
# survives into a draft, the clarity guardrail flags it rather than guessing.
UNEXPLAINED_JARGON = [
    "conc", "bcobs", "mcob", "cobs", "icobs", "disp", "prin",
    "fg21/1", "consumer duty", "app fraud", "app scam", "psr",
    "vulnerability driver", "risk level", "guardrail", "pipeline",
    "advisory", "policy clause", "knowledge base",
]

# Phrases that mean the handler is talking *about* the customer rather than *to*
# them — the single clearest sign that policy voice has leaked into the reply.
THIRD_PERSON_MARKERS = [
    "the customer", "the caller", "the consumer", "the account holder",
    "they will not need", "they need not", "their account",
]


# A replacement that already carries its own article leaves "a a reduced
# payment" behind. Cheaper to clean up afterwards than to make every entry
# article-aware.
_DOUBLE_ARTICLE = re.compile(r"\b(a|an|the)\s+(a|an|the)\b", re.IGNORECASE)
# An article in front of a clause rather than a noun: "explain the how and when
# you get your money back". The article belonged to the term that was replaced.
_ARTICLE_BEFORE_CLAUSE = re.compile(
    r"\b(?:a|an|the)\s+(?=(?:how|what|when|whether|ways|checking|putting|getting|"
    r"trying|any|free)\b)", re.IGNORECASE)
# A split can leave a clause hanging on its conjunction.
_DANGLING_CONJUNCTION = re.compile(r"\s+(?:and|or|but|with|to|for|of)\s*$", re.IGNORECASE)


def tidy_offer(text: str) -> str:
    """Clean the seams left by substitution and splitting.

    Each of these is an artefact of composing a sentence out of pieces rather
    than a wording choice, so they are fixed here once instead of being
    special-cased in every entry of the lexicon.
    """
    out = _DOUBLE_ARTICLE.sub(r"\2", text)
    out = _ARTICLE_BEFORE_CLAUSE.sub("", out)
    out = re.sub(r"\s+", " ", out).strip(" .,;:")
    out = _DANGLING_CONJUNCTION.sub("", out)
    return out.strip(" .,;:")


def plain_english(text: str) -> str:
    """Replace industry terms with the words a customer would use."""
    out = text
    for term, replacement in _PLAIN_ENGLISH:
        out = re.sub(re.escape(term), replacement, out, flags=re.IGNORECASE)
    out = _DOUBLE_ARTICLE.sub(r"\2", out)
    return re.sub(r"\s+", " ", out).strip()


# --------------------------------------------------------------------------- #
# Turning policy instructions into offers
# --------------------------------------------------------------------------- #
# Verbs a clause uses to prescribe an action, mapped to how that action is
# offered to the customer. The order matters: longer, more specific forms first.
_OFFER_VERBS: list[tuple[str, str]] = [
    (r"offer to ", "I can "),
    (r"offer (?:the customer |them )?a ", "I can arrange a "),
    (r"offer (?:the customer |them )?an ", "I can arrange an "),
    (r"offer ", "I can arrange "),
    (r"arrange (?:for )?", "I can arrange "),
    (r"provide (?:the customer |them )?(?:with )?", "I can give you "),
    (r"give (?:the customer |them )?", "I can give you "),
    (r"set up ", "I can set up "),
    (r"apply ", "I can apply "),
    (r"put (?:the customer |them )?in touch with ", "I can put you in touch with "),
    (r"refer (?:the customer |them )?to ", "I can refer you to "),
    (r"signpost (?:the customer |them )?(?:to )?", "I can put you in touch with "),
    (r"suspend ", "I can pause "),
    (r"pause ", "I can pause "),
    (r"freeze ", "I can freeze "),
    (r"stop ", "I can stop "),
    (r"accept ", "I can accept "),
    (r"allow ", "I can give you "),
    (r"explain ", "I can explain "),
    (r"check (?:whether |if )?", "I can check "),
    (r"note ", "I can note "),
    (r"record ", "I can note "),
]

# Reassurances rather than offers — "do not require the customer to repeat it"
# is a promise to the customer, phrased in the negative to staff.
_REASSURANCES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"do not require the customer to repeat|need not explain again|"
                r"never (?:have to |need to )?repeat|not (?:have to |need to )?repeat",
                re.IGNORECASE),
     "you won't have to explain this again"),
    (re.compile(r"do not (?:pressure|press) for payment", re.IGNORECASE),
     "I won't push you for a payment you can't make"),
    (re.compile(r"treat (?:our|the) debt as non-priority|non-priority relative to essentials",
                re.IGNORECASE),
     "your rent, energy and food come before paying us"),
    (re.compile(r"without judgement|avoid judgement|not attribut\w* blame|"
                r"not a customer error|do not blame", re.IGNORECASE),
     "there's no judgement here"),
    (re.compile(r"at (?:the customer's|their) (?:own )?pace|no deadline", re.IGNORECASE),
     "there's no rush — we can go at your pace"),
    (re.compile(r"single point of contact|one named point of contact", re.IGNORECASE),
     "you'll deal with just me rather than being passed around"),
]

# Sentence fragments that are purely internal process and never belong in a reply.
_INTERNAL_ONLY = re.compile(
    r"^\s*(?:record|log|note on the (?:account|file)|assess whether|"
    r"consider whether|follow|treat it as|use a|do not disclose)\b",
    re.IGNORECASE,
)

_CUSTOMER_TO_YOU = [
    (re.compile(r"\bthe customer's\b", re.IGNORECASE), "your"),
    (re.compile(r"\bthe customer\b", re.IGNORECASE), "you"),
    (re.compile(r"\btheir\b", re.IGNORECASE), "your"),
    (re.compile(r"\bthem\b", re.IGNORECASE), "you"),
    (re.compile(r"\bthey are\b", re.IGNORECASE), "you are"),
    (re.compile(r"\bthey\b", re.IGNORECASE), "you"),
]


def _to_second_person(text: str) -> str:
    out = text
    for pattern, replacement in _CUSTOMER_TO_YOU:
        out = pattern.sub(replacement, out)
    return out


def _strip_clause_title(text: str) -> str:
    """Drop a leading 'Clause title: ' prefix, which is a filing label."""
    head, sep, tail = text.partition(":")
    if sep and len(head.split()) <= 8 and not head.strip().lower().startswith("i "):
        return tail.strip()
    return text.strip()


def _split_clauses(text: str) -> list[str]:
    """Break policy prose into the individual instructions it contains."""
    parts: list[str] = []
    for sentence in re.split(r"(?<=[.;:])\s+", text):
        # A single sentence commonly chains several instructions with commas
        # and "and": "offer X, signpost Y, and provide Z".
        for piece in re.split(r",\s+(?=(?:and\s+)?[a-z])|\.\s+", sentence):
            piece = re.sub(r"^\s*and\s+", "", piece).strip(" .;:")
            if piece:
                parts.append(piece)
    return parts


# Trailing clauses that are true but belong to the handler's process, not to the
# offer the customer is being asked to accept.
_OFFER_TAIL = re.compile(
    r"\s+(?:while\b|where\b|so that\b|and confirm\b|and record\b|and note\b|"
    r"for this contact\b|in line with\b|as prescribed\b).*$",
    re.IGNORECASE,
)

# An offer longer than this stops being something a listener can say yes to.
MAX_OFFER_WORDS = 12


def extract_offers(adaptations: list[str], limit: int = 2) -> tuple[list[str], list[str]]:
    """Turn handler instructions into things you can say to the customer.

    Returns (offers, reassurances). Offers are phrased "I can …"; reassurances
    are statements of what will not happen to them. Anything that is purely
    internal process is dropped rather than paraphrased — a customer does not
    need to hear that their vulnerability was recorded.
    """
    offers: list[str] = []
    reassurances: list[str] = []

    for adaptation in adaptations:
        body = _strip_clause_title(adaptation)
        for clause in _split_clauses(body):
            if _INTERNAL_ONLY.match(clause):
                continue

            matched_reassurance = next(
                (phrase for pattern, phrase in _REASSURANCES if pattern.search(clause)), None)
            if matched_reassurance and matched_reassurance not in reassurances:
                reassurances.append(matched_reassurance)

            lowered = clause[0].lower() + clause[1:] if clause else clause
            for pattern, replacement in _OFFER_VERBS:
                match = re.match(pattern, lowered, flags=re.IGNORECASE)
                if not match:
                    continue
                remainder = lowered[match.end():].strip()
                if not remainder:
                    break
                # Cut the process tail before measuring: "a reduced payment while
                # the estate is settled and confirm it in writing" is one short
                # offer wearing a long coat.
                remainder = _OFFER_TAIL.sub("", remainder)
                if len(remainder.split()) > MAX_OFFER_WORDS:
                    break
                offer = plain_english(_to_second_person(remainder))
                offer = tidy_offer(replacement.strip() + " " + offer)
                # Too short to mean anything once the tail was cut.
                if len(offer.split()) >= 4 and offer not in offers:
                    offers.append(offer)
                break

    return offers[:limit], reassurances[:2]


# --------------------------------------------------------------------------- #
# The composed reply
# --------------------------------------------------------------------------- #
def _opener(context: Optional[FinancialContext], decision,
            sentiment: Optional[SentimentReading]) -> str:
    if sentiment is not None and sentiment.distress >= 0.7:
        return _DISTRESS_OPENER
    if context is not None and context.journey in _JOURNEY_OPENERS:
        return _JOURNEY_OPENERS[context.journey]
    if decision is not None and decision.assessment.signals:
        triggered = decision.assessment.triggered
        if triggered:
            top = max((s for s in decision.assessment.signals if s.driver in triggered),
                      key=lambda s: s.score)
            return _DRIVER_OPENERS.get(top.driver, _NEUTRAL_OPENER)
    return _NEUTRAL_OPENER


def _overlaps(phrase: str, offers: list[str]) -> bool:
    """Do these say substantially the same thing?"""
    def keywords(text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z]{4,}", text.lower())}

    words = keywords(phrase)
    if not words:
        return False
    return any(len(words & keywords(offer)) / len(words) >= 0.5 for offer in offers)


def _state_facts(facts: list[str]) -> str:
    """Read the ledger back as a sentence a person can follow.

    The first fact names the account; the rest are the figures. Kept to two
    figures — a customer cannot hold a read-out of their whole file.
    """
    if not facts:
        return ""
    account, figures = facts[0], facts[1:3]
    if not figures:
        return f"That's your {account}."
    if len(figures) == 1:
        return f"On your {account}, {figures[0]}."
    return f"On your {account}, {figures[0]}, and {figures[1]}."


def _join_offers(offers: list[str]) -> str:
    """One offer reads as a promise; two read as a choice. Three read as a menu,
    which is why the composer never returns three."""
    if not offers:
        return ""
    if len(offers) == 1:
        return offers[0].rstrip(".") + "."
    first, second = offers[0].rstrip("."), offers[1].rstrip(".")
    # Two sentences rather than one long one: both are spoken aloud, and joining
    # them with "and" produces exactly the over-long sentence the clarity
    # guardrail exists to catch.
    if len(first.split()) + len(second.split()) > 22:
        return f"{first}. {second}."
    # The second offer drops a repeated "I can" so the joined sentence flows.
    if first.lower().startswith("i can") and second.lower().startswith("i can"):
        second = second[len("I can "):]
    return f"{first}, and {second}."


def approved_offers(retrieved) -> list[str]:
    """The signed-off wording from the retrieved clauses, best-scoring first."""
    out: list[str] = []
    for chunk in retrieved or []:
        for offer in getattr(chunk, "offers", []) or []:
            if offer not in out:
                out.append(offer)
    return out


def offers_for(decision, retrieved=None, limit: int = 2) -> tuple[list[str], list[str]]:
    """The offers to put in front of the customer.

    Approved wording first; derivation only fills the gap when a clause has none.
    """
    approved = approved_offers(retrieved)
    if approved:
        _, reassurances = extract_offers(
            decision.recommendation.adaptations if decision and decision.recommendation else [])
        return approved[:limit], reassurances
    if decision is None or decision.recommendation is None:
        return [], []
    return extract_offers(decision.recommendation.adaptations, limit=limit)


def compose_reply(decision, context: Optional[FinancialContext] = None,
                  sentiment: Optional[SentimentReading] = None,
                  retrieved=None, account=None) -> str:
    """Build the handler's reply without a model.

    This is the path every demo runs on until an API key is configured, so it
    has to be genuinely good rather than a placeholder. Structure is fixed and
    deliberate: acknowledge, then one or two concrete offers, then a question
    that hands the conversation back.
    """
    if decision is None:
        return "Thanks for calling. How can I help you today?"

    opener = _opener(context, decision, sentiment)

    # A refused data request is answered before anything else, and the refusal is
    # stated as a fact rather than hedged.
    if account is not None and getattr(account, "refused", False):
        parts = [opener, account.refusal_reason]
        if account.alternative:
            parts.append(account.alternative.rstrip(".") + ".")
        parts.append("Would that help?")
        return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()

    # A product the caller does not hold. Answered plainly, and paired with what
    # they do hold — the alternative is what stops "I can't see that" landing as
    # "the bank has lost my account".
    if account is not None and getattr(account, "decision", "none") == "unavailable":
        parts = [opener, account.refusal_reason]
        if account.alternative:
            parts.append(account.alternative.rstrip(".") + ".")
            parts.append("Would you like me to go through one of those?")
        else:
            parts.append("Is there something else I can help with?")
        return re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()

    # Account facts are stated before anything else can short-circuit the reply.
    # They used to sit after the no-recommendation early return, so a plain
    # "how much do I owe?" — which triggers no vulnerability and retrieves no
    # policy — fell through to "let me bring up your account" while the figure
    # sat unused in the state. Answering the question the customer asked is not
    # contingent on a policy clause being retrieved.
    facts = ""
    if account is not None and getattr(account, "decision", "none") in ("disclose", "partial"):
        if account.facts:
            facts = _state_facts(account.facts)
        if getattr(account, "refusal_reason", ""):
            facts = (facts + " " + account.refusal_reason).strip()

    recommendation = decision.recommendation

    if recommendation is None or not recommendation.adaptations:
        if facts:
            return re.sub(r"\s+", " ",
                          f"{facts} Is there anything else I can help with?").strip()
        # A distressed customer with no retrieved policy is the one case where
        # saying nothing useful is actively harmful. Answer the person, not the
        # transaction, and hand the turn back rather than moving on.
        if sentiment is not None and sentiment.distress >= 0.5:
            return (f"{opener} There's no rush at all — take your time, "
                    "and tell me what would help most right now.")
        if decision.assessment.triggered:
            return f"{opener} Let me talk you through the support we can offer."
        # Nothing triggered, no policy, no account question. Rather than assert
        # that an account is being brought up — which is what this used to say to
        # anyone asking the time — invite the customer to say what they need.
        return "Of course. What can I help you with today?"

    # A reply carrying figures has already used most of the customer's attention,
    # so it gets one offer rather than two.
    offers, reassurances = offers_for(decision, retrieved, limit=1 if facts else 2)

    parts = [opener]

    # If they asked about their money, answer that first — an offer lands better
    # once the customer knows what it is an offer about.
    if facts:
        parts.append(facts)

    if reassurances:
        # Only if it adds something. The essentials reassurance and the
        # essentials offer are the same sentence twice, and saying it twice
        # sounds like a script rather than a person.
        reassurance = reassurances[0]
        if not _overlaps(reassurance, offers):
            parts.append(reassurance.capitalize().rstrip(".") + ".")
    if offers:
        joined = _join_offers(offers)
        parts.append(joined)
        # Approved wording sometimes ends in a question of its own ("...is that
        # alright?"). Appending a second one produced "is that alright? Would
        # that help?" — two questions, so the customer answers neither.
        if not joined.rstrip().endswith("?"):
            parts.append("Would that help?")
    else:
        parts.append("Let me talk you through the support we can offer.")

    reply = " ".join(part for part in parts if part)
    return re.sub(r"\s+", " ", reply).strip()


# --------------------------------------------------------------------------- #
# The prompt, when a model is available
# --------------------------------------------------------------------------- #
def build_reply_prompt(customer_text: str, decision,
                       context: Optional[FinancialContext] = None,
                       sentiment: Optional[SentimentReading] = None,
                       retrieved=None, account=None, history: str = "") -> str:
    """Assemble the user message for the handler-reply model call.

    The policy is passed through the same plain-English pass the template path
    uses, so the model is reading the customer-facing vocabulary rather than
    reaching for the industry term it just saw.
    """
    recommendation = decision.recommendation if decision else None
    if recommendation:
        offers, reassurances = offers_for(decision, retrieved, limit=3)
        guidance = plain_english(recommendation.summary)
        available = "\n".join(f"- {offer}" for offer in offers) or "- (nothing specific)"
        promises = "\n".join(f"- {r}" for r in reassurances)
    else:
        guidance, available, promises = "(no specific policy retrieved)", "- (nothing specific)", ""

    lines = []
    if history:
        # Without the thread, "Yes, it would help." is an unremarkable four-word
        # utterance that retrieves nothing — and the handler answers it by asking
        # what they need, having just offered them something.
        lines += [history, ""]
    lines.append(f"The customer has just said: {customer_text}")
    if history:
        lines.append(
            "If they are agreeing to something you just offered, confirm you are "
            "doing it and say what happens next. Do not ask again what they need.")

    # Account material comes first in the prompt as well as in the reply: it is
    # the thing most likely to be dropped if it arrives last.
    if account is not None and getattr(account, "asked", False):
        lines.append("")
        if account.refused:
            lines += [
                "THE CALLER HAS ASKED ABOUT SOMEBODY ELSE'S ACCOUNT. You must decline.",
                f"Say, in your own warm words: {account.refusal_reason}",
                "Do not give any figure, balance, number or detail from that account, "
                "and do not say whether it exists. Do not soften the refusal into a "
                "maybe, a 'let me check', or a promise to look into it.",
            ]
            if account.alternative:
                lines.append(f"Then offer what you can do instead: {account.alternative}")
        elif getattr(account, "decision", "none") == "unavailable":
            lines += [
                "THE CALLER HAS ASKED ABOUT A PRODUCT THEY DO NOT HOLD.",
                f"Say plainly: {account.refusal_reason}",
                "Do not read out figures from a different account, and do not "
                "guess at which account they meant.",
            ]
            if account.alternative:
                lines.append(f"Then tell them what is on the file: {account.alternative}, "
                             "and ask which they would like to go through.")
        elif account.facts:
            lines += [
                "Account facts you may state (already masked — use them as written, "
                "and never give a fuller number than this):",
                *[f"- {fact}" for fact in account.facts],
            ]
            if account.refusal_reason:
                lines.append(account.refusal_reason)

    lines += [
        "",
        f"What policy allows you to do: {guidance}",
        "",
        "Specific things you may offer (use at most two; this wording is approved, "
        "so stay close to it):",
        available,
    ]
    if promises:
        lines += ["", "Reassurances you may give:", promises]
    if context is not None and context.source != "skipped":
        lines += ["", f"Situation: {context.summary()}"]
        if context.prohibited:
            lines.append("Never offer: " + "; ".join(context.prohibited))
    if sentiment is not None and sentiment.source != "skipped":
        descriptor = "in real distress" if sentiment.distress >= 0.7 else sentiment.label.lower()
        lines += ["", f"The customer sounds {descriptor}."]
    lines += ["", "Write the handler's reply, speaking directly to the customer:"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Lines for the procedural stages of the call
# --------------------------------------------------------------------------- #
# Approved wording for each objective, used when no model is configured. These
# are floors, not scripts: with a model the handler says the same thing in their
# own words, which is why a caller does not hear the identical sentence twice.
_STAGE_FALLBACK: dict[str, str] = {
    "ask who you are speaking to": "Thank you. Can I take your name, please?",
    "ask for their name again": "Sorry, I didn't catch that — can I take your name?",
    "proceed without the account": (
        "I'm not able to bring up an account without those details, but I can "
        "still help. What's on your mind?"),
    "respond": "I see. Tell me a little more.",
    "ask how you can help": "Thank you, that's all confirmed. How can I help you today?",
    "confirm they are verified and ask how you can help": (
        "Thank you, that's confirmed. How can I help you today?"),
    "offer general help without the account": (
        "I can't go through the account itself, but I can still help. "
        "What did you need?"),
    "explain you cannot go through the account, and offer branch or post": (
        "I'm sorry — I can't confirm those details, so I'm not able to go through "
        "the account on this call. Any branch can help you with ID, or you can "
        "write to us."),
}


# Objectives that embed a value, matched on a cue rather than the whole string.
_OBJECTIVE_TEMPLATES: list[tuple[str, str]] = [
    ("their date of birth", "Thank you. {name}could you confirm your date of birth?"),
    ("the postcode", "Thank you. {name}could you confirm the PIN code on the account?"),
    ("spell it", "Sorry, {name}I can't find that name — could you spell it for me?"),
    ("a security detail", "Thank you. {name}could you confirm a detail on the account?"),
    ("did not match", "That doesn't quite match what I have, I'm afraid. "
                      "Could we try another detail?"),
]


def stage_line(objective: str, *, customer_name: str = "", last_utterance: str = "",
               disclosure: str = "", history: str = "") -> str:
    """The handler's line at a procedural stage of the call.

    The *content* is fixed by `objective` — a bank must take a name and confirm
    identity in a consistent way — but the *wording* is generated, because a
    handler who says the identical sentence to every caller sounds like an IVR,
    and because the line has to bend around whatever the customer just said.

    `disclosure` is the important argument. A customer often says the thing that
    matters while you are still taking their details; when that happens the line
    must acknowledge it before continuing with the procedure.
    """
    llm = get_llm()
    if llm.available:
        parts = [f"Stage objective: {objective}."]
        if customer_name:
            parts.append(f"The customer's name is {customer_name}.")
        if history:
            parts.append(history)
        if last_utterance:
            parts.append(f"They have just said: {last_utterance}")
        if disclosure:
            parts.append(
                f"IMPORTANT — they have just disclosed something difficult: "
                f"{disclosure}. Acknowledge that warmly in your first clause, then "
                f"do what the objective asks.")
        parts.append("Write the handler's next line.")

        from .prompts import HANDLER_TURN_SYSTEM

        drafted = llm.text(HANDLER_TURN_SYSTEM, "\n\n".join(parts),
                           max_tokens=160, effort=EFFORT_REPLY)
        if drafted and drafted.strip():
            return drafted.strip().strip('"')

    line = _STAGE_FALLBACK.get(objective)
    if line:
        if customer_name and "name, please" not in line:
            line = line.replace("Thank you,", f"Thank you, {customer_name.split()[0]},", 1)
        return line

    # Objectives that carry a value in them ("ask for their date of birth") have
    # no fixed entry, so they are turned into a question here. Falling back to
    # speaking the objective itself put "Thank them by name and ask for their
    # date of birth." in the handler's mouth — a stage direction read aloud.
    first = customer_name.split()[0] if customer_name else ""
    for cue, template in _OBJECTIVE_TEMPLATES:
        if cue in objective:
            return template.format(name=(f"{first}, " if first else "")).strip()
    return "Sorry — could you bear with me one moment?"


# --------------------------------------------------------------------------- #
# Following the thread without a model
# --------------------------------------------------------------------------- #
# Word sets rather than one big alternation: this has to be obviously correct at
# a glance, and a regex that silently stops matching "yes" is worse than useless.
_AFFIRMATIVE_OPENERS = {
    "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "please", "definitely",
    "absolutely", "certainly", "alright", "right", "good", "great", "fine",
}
_AFFIRMATIVE_PHRASES = (
    "go ahead", "please do", "that would help", "it would help", "that helps",
    "sounds good", "that's fine", "thats fine", "yes please", "i would",
)
# A "yes" carrying one of these is not a plain agreement — "yes, but I still
# can't pay" has a second half that matters far more than the first.
_CONTRADICTIONS = {"but", "however", "although", "still", "not", "cant",
                   "cannot", "wont", "except"}

# "I can arrange a moratorium" -> "arrange a moratorium"
_OFFER_IN_LINE = re.compile(r"I (?:can|could|will|'ll) ([^.?!]{4,90})", re.IGNORECASE)


def is_affirmation(text: str) -> bool:
    """Is this the customer saying yes to what was just offered?

    Deliberately narrow. It only fires on a short utterance that opens with an
    agreement and carries no contradiction, because the cost of being wrong is
    confirming an action the customer did not actually accept.
    """
    stripped = (text or "").strip().lower()
    if not stripped:
        return False
    words = re.findall(r"[a-z']+", stripped)
    if not words or len(words) > 8:
        return False
    if _CONTRADICTIONS & set(words):
        return False
    if words[0] in _AFFIRMATIVE_OPENERS:
        return True
    return any(phrase in stripped for phrase in _AFFIRMATIVE_PHRASES)


def confirm_offer(previous_handler_line: str) -> str:
    """Confirm the thing the handler last offered.

    Without this, a bare "yes" retrieves no policy, triggers no driver and
    carries no account question — so the composer fell through to asking the
    customer what they needed, one turn after offering it to them. Which is the
    most obviously broken thing a support call can do.
    """
    offers = _OFFER_IN_LINE.findall(previous_handler_line or "")
    if not offers:
        return ("Of course — I'll get that started and confirm it in writing. "
                "Is there anything else I can help with?")
    # Take the offer up to its first comma: the tail is usually a second clause
    # ("…, and confirm that in writing") that this sentence is about to add back.
    action = offers[0].split(",")[0].strip().rstrip(".")
    return (f"Right — I'll {action}, and confirm it in writing. "
            "Is there anything else I can help with?")
