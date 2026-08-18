"""Tests for the customer-facing reply.

These exist because the failure they guard against is invisible to every other
check in the system. A reply can be perfectly grounded in policy, pass the
prohibited-action guardrail, carry the right risk level — and still be a handler
reading staff instructions aloud to a bereaved customer. Correct and unusable are
not the same thing.

The regression at the top of this file is the one that motivated the work: the
composer used to emit "Express condolences and reassure the customer they will
not need to repeat the bereavement disclosure to another team."
"""
from __future__ import annotations

import pytest

from guardiancx.agents.reply import (
    THIRD_PERSON_MARKERS,
    compose_reply,
    extract_offers,
    plain_english,
    tidy_offer,
)
from guardiancx.finance.taxonomy import FinancialContext, Journey
from guardiancx.guardrails.base import GuardrailContext
from guardiancx.guardrails.clarity import ClarityGuardrail
from guardiancx.utils.types import (
    CaseDecision,
    Driver,
    DriverSignal,
    PolicyChunk,
    Recommendation,
    SentimentReading,
    VulnerabilityAssessment,
)


def _decision(adaptations, driver=Driver.LIFE_EVENTS, score=0.9):
    return CaseDecision(
        conversation_id="T", turn_index=0,
        assessment=VulnerabilityAssessment(
            signals=[DriverSignal(driver=driver, score=score)]),
        recommendation=Recommendation(
            summary="Consider the prescribed adaptation.",
            adaptations=adaptations, citations=["VP-L1"], confidence=0.8),
    )


# --------------------------------------------------------------------------- #
# The regression that started this
# --------------------------------------------------------------------------- #
def test_staff_instructions_are_never_read_aloud():
    decision = _decision([
        "Bereavement: Express condolences and reassure the customer they will not "
        "need to repeat the bereavement disclosure to another team. Offer to pause "
        "interest and charges."
    ])
    reply = compose_reply(decision, FinancialContext(journey=Journey.BEREAVEMENT_ESTATE))
    lowered = reply.lower()
    for marker in THIRD_PERSON_MARKERS:
        assert marker not in lowered, f"{marker!r} leaked into: {reply}"
    assert "reassure" not in lowered
    assert "express condolences" not in lowered


def test_the_clause_title_is_not_spoken():
    decision = _decision([
        "Job loss or income shock: When a customer reports losing their job, offer "
        "a review of affordability and offer to pause or reduce payments."
    ])
    reply = compose_reply(decision, FinancialContext(journey=Journey.AFFORDABILITY_SHOCK))
    assert "job loss or income shock:" not in reply.lower()
    assert "when a customer reports" not in reply.lower()


def test_reply_is_not_truncated_mid_word():
    decision = _decision(["Offer " + "a very long policy instruction " * 20])
    reply = compose_reply(decision)
    assert "…" not in reply
    assert not reply.rstrip().endswith(("a", "the", "and", "of", "to"))


# --------------------------------------------------------------------------- #
# Approved wording wins
# --------------------------------------------------------------------------- #
def test_approved_wording_is_used_verbatim():
    """What a vulnerable customer hears should be signed off, not derived."""
    chunk = PolicyChunk(
        policy_reference="VP-J5", title="Bereavement", text="…",
        offers=["I can be your single point of contact, so you only have to tell me once"],
    )
    reply = compose_reply(
        _decision(["Open a bereavement case with one named point of contact."]),
        FinancialContext(journey=Journey.BEREAVEMENT_ESTATE),
        retrieved=[chunk],
    )
    assert "I can be your single point of contact" in reply


def test_derivation_fills_in_when_a_clause_has_no_approved_wording():
    chunk = PolicyChunk(policy_reference="VP-X1", title="Something", text="…", offers=[])
    reply = compose_reply(
        _decision(["Offer a payment holiday for three months."]),
        FinancialContext(journey=Journey.FORBEARANCE_REQUEST),
        retrieved=[chunk],
    )
    assert "payment holiday" in reply.lower()


# --------------------------------------------------------------------------- #
# Openers
# --------------------------------------------------------------------------- #
def test_opener_follows_the_situation_not_the_highest_driver():
    """A scam victim scoring on capability must not be told 'I'll go at your pace'
    — it answers a question they did not ask, at the moment they feel foolish."""
    decision = _decision(["Stop further payments."], driver=Driver.CAPABILITY, score=0.9)
    reply = compose_reply(decision, FinancialContext(journey=Journey.FRAUD_SCAM))
    assert "you did the right thing" in reply.lower()
    assert "go at your pace" not in reply.lower()


def test_distress_changes_the_opening():
    decision = _decision(["Offer to stay on the line."])
    calm = compose_reply(decision, None, SentimentReading(distress=0.1))
    upset = compose_reply(decision, None, SentimentReading(distress=0.9))
    assert calm != upset
    assert "i can hear this is really hard" in upset.lower()


def test_distress_without_retrieved_policy_still_answers_the_person():
    """The failure this guards against: answering a crisis with account servicing."""
    decision = CaseDecision(
        conversation_id="T", turn_index=0,
        assessment=VulnerabilityAssessment(), recommendation=None,
    )
    reply = compose_reply(decision, None, SentimentReading(distress=0.9, escalate=True))
    assert "bring up your account" not in reply.lower()
    assert "no rush" in reply.lower()


# --------------------------------------------------------------------------- #
# Plain English and the seams
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("term", ["forbearance", "signpost", "moratorium"])
def test_industry_terms_are_translated(term):
    assert term not in plain_english(f"We can offer {term} today.").lower()


def test_longest_term_wins():
    """Hand-ordering the lexicon silently breaks; the sort must handle it."""
    out = plain_english("signpost hardship and forbearance options")
    assert "forbearance" not in out.lower()
    assert "hardship and" not in out.lower()


def test_no_relationship_is_ever_assumed():
    """A bereaved customer may have lost a wife, a parent or a child."""
    out = plain_english("a payment concession while the estate is settled")
    assert "husband" not in out.lower()
    assert "wife" not in out.lower()


@pytest.mark.parametrize("raw,unwanted", [
    ("I can arrange a a reduced payment", "a a"),
    ("I can explain the how and when you get your money back", "the how"),
    ("I can put you in touch with probate support and", " and"),
])
def test_composition_seams_are_cleaned(raw, unwanted):
    assert unwanted not in tidy_offer(raw) or not tidy_offer(raw).endswith(unwanted)


def test_offers_become_first_person_offers():
    offers, _ = extract_offers(["Offer a payment holiday. Signpost free debt advice."])
    assert offers
    assert all(o.lower().startswith("i can") for o in offers)


def test_a_clause_can_yield_both_an_offer_and_a_reassurance():
    """'offer practical tools without judgement' carries both; the reassurance
    branch used to swallow the offer with it."""
    offers, reassurances = extract_offers(
        ["Offer practical tools without judgement. Offer a gambling block on the card."])
    assert offers
    assert reassurances


def test_internal_process_is_dropped():
    offers, _ = extract_offers(["Record the affordability concern. Assess whether the "
                                "customer is more susceptible."])
    assert offers == []


def test_at_most_two_offers():
    offers, _ = extract_offers([
        "Offer a payment holiday. Offer a reduced payment. Offer to freeze interest. "
        "Offer a callback."])
    assert len(offers) <= 2


# --------------------------------------------------------------------------- #
# The clarity guardrail
# --------------------------------------------------------------------------- #
def _clarity(reply: str):
    return ClarityGuardrail().check(GuardrailContext(text="", reply=reply))


def test_clarity_passes_a_good_reply():
    result = _clarity(
        "I'm so sorry for your loss. I can stop the interest and letters from today. "
        "Would that help?")
    assert result.passed


def test_clarity_catches_policy_voice():
    result = _clarity("I will reassure the customer that they need not repeat this.")
    assert not result.passed
    assert "about the customer" in result.detail


def test_clarity_catches_jargon_and_codes():
    result = _clarity("I can apply CONC forbearance under VP-J1 for you.")
    assert not result.passed
    assert "will not know" in result.detail


def test_clarity_catches_an_empty_offer():
    result = _clarity("There are some options available to you.")
    assert not result.passed


def test_clarity_catches_a_sentence_too_long_to_hear():
    long_sentence = ("I can arrange " + "a rather detailed and lengthy arrangement " * 5
                     + "for you.")
    result = _clarity(long_sentence)
    assert not result.passed
    assert "too long to follow" in result.detail


def test_clarity_is_advisory_not_blocking():
    """A clumsy sentence is better than silence on a live call."""
    result = _clarity("The customer should be reassured.")
    assert result.severity == "warn"


def test_clarity_ignores_an_empty_reply():
    assert _clarity("").passed
