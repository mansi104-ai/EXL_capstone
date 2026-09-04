"""Tests for the finance domain layer — the niche narrowing.

What these check is not "does the classifier work" but the claims the product
rests on: that the detriment axis is independent of the vulnerability axis, that
a high-harm journey reaches a human whether or not a driver fired, and that
advice which would cause harm is blocked deterministically rather than politely
discouraged in a prompt.
"""
from __future__ import annotations

import pytest

from guardiancx.finance.taxonomy import (
    Journey,
    Product,
    StressIndicator,
    classify_heuristic,
)
from guardiancx.guardrails.base import GuardrailContext
from guardiancx.guardrails.prohibited_action import ProhibitedActionGuardrail
from guardiancx.utils.types import PolicyChunk, Recommendation


# --------------------------------------------------------------------------- #
# Journey classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text,journey", [
    ("My husband passed away and I need to sort out the estate.", Journey.BEREAVEMENT_ESTATE),
    ("Someone told me to move my money to a safe account.", Journey.FRAUD_SCAM),
    ("I've been gambling and I can't stop.", Journey.GAMBLING_HARM),
    ("Can I have a payment holiday for three months?", Journey.FORBEARANCE_REQUEST),
    ("I lost my job on Friday and I can't afford the payment.", Journey.AFFORDABILITY_SHOCK),
    ("My daughter helps me with the online banking.", Journey.THIRD_PARTY_ACCESS),
    ("I just want to check my balance.", Journey.GENERAL_SERVICING),
])
def test_journeys_are_identified(text, journey):
    assert classify_heuristic(text).journey is journey


def test_the_more_harmful_journey_wins_when_two_apply():
    """A scam disclosed inside a collections call is a scam call."""
    text = ("I'm behind on my payments because someone scammed me out of "
            "my savings last month.")
    assert classify_heuristic(text).journey is Journey.FRAUD_SCAM


@pytest.mark.parametrize("text,product", [
    ("I'm behind on the mortgage.", Product.MORTGAGE),
    ("My credit card balance is too high.", Product.CREDIT_CARD),
    ("I want to cancel a direct debit on my current account.", Product.CURRENT_ACCOUNT),
])
def test_products_are_identified(text, product):
    assert classify_heuristic(text).product is product


def test_arrears_are_counted_in_both_phrasings():
    assert classify_heuristic("I'm three months behind.").arrears_months == 3
    assert classify_heuristic("I've missed two payments.").arrears_months == 2


def test_monetary_amounts_are_extracted():
    context = classify_heuristic("I owe AED 124,050 and can only pay AED 5,000 this month.")
    assert "AED 124,050" in context.monetary_amounts


def test_spoken_dirham_amounts_are_extracted():
    """Callers say the currency after the figure as often as before it."""
    context = classify_heuristic("The loan is about 280,000 dirhams and I've paid 20,000 AED.")
    assert any("dirham" in a for a in context.monetary_amounts)
    assert any("AED" in a for a in context.monetary_amounts)


# --------------------------------------------------------------------------- #
# The two axes are independent
# --------------------------------------------------------------------------- #
def test_financial_detriment_without_a_vulnerability_disclosure():
    """A composed, capable customer can still be in serious difficulty — and the
    forbearance duty is engaged regardless of how they sound."""
    context = classify_heuristic(
        "I'm two months in arrears on the loan and I'd like to arrange a payment plan.")
    assert context.acute
    assert context.high_harm
    assert StressIndicator.ARREARS in context.stress_indicators


def test_life_event_without_financial_detriment():
    """Bereavement is a vulnerability driver but not, by itself, financial stress."""
    context = classify_heuristic(
        "My mother passed away and I need to close her savings account.")
    assert context.journey is Journey.BEREAVEMENT_ESTATE
    assert context.stress_indicators == []


def test_essential_spend_conflict_is_treated_as_acute():
    context = classify_heuristic(
        "It's the electricity bill or paying you — I've no money for food otherwise.")
    assert context.acute


# --------------------------------------------------------------------------- #
# Obligations and prohibitions
# --------------------------------------------------------------------------- #
def test_every_journey_maps_to_at_least_one_obligation():
    for journey in Journey:
        context = classify_heuristic("placeholder")
        context.journey = journey
        assert context.obligations, f"{journey} has no regulatory mapping"


def test_high_harm_journeys_carry_prohibitions():
    for journey in (Journey.GAMBLING_HARM, Journey.FRAUD_SCAM,
                    Journey.ARREARS_COLLECTIONS, Journey.AFFORDABILITY_SHOCK):
        context = classify_heuristic("placeholder")
        context.journey = journey
        assert context.prohibited, f"{journey} has no prohibited actions"


# --------------------------------------------------------------------------- #
# The prohibited-action guardrail
# --------------------------------------------------------------------------- #
def _check(summary, adaptations, journey, retrieved=None):
    return ProhibitedActionGuardrail().check(GuardrailContext(
        text="",
        recommendation=Recommendation(summary=summary, adaptations=adaptations,
                                      citations=["VP-J9"], confidence=0.9),
        retrieved=retrieved or [],
        journey=journey,
    ))


def test_offering_credit_for_gambling_harm_is_blocked():
    result = _check("Offer a consolidation loan to clear the gambling debts.",
                    [], Journey.GAMBLING_HARM)
    assert not result.passed
    assert result.severity == "block"


def test_limit_increase_for_gambling_harm_is_blocked():
    result = _check("Support the customer.",
                    ["Increase the credit limit so they have some headroom."],
                    Journey.GAMBLING_HARM)
    assert not result.passed


def test_safe_account_advice_on_a_scam_is_blocked():
    result = _check("Ask the customer to move their money to a safe account.",
                    [], Journey.FRAUD_SCAM)
    assert not result.passed


def test_asking_for_credentials_is_blocked_on_every_journey():
    result = _check("Confirm the full card number with the customer.",
                    [], Journey.GENERAL_SERVICING)
    assert not result.passed


def test_a_prohibition_stated_as_a_prohibition_is_not_a_breach():
    """Policy names forbidden actions precisely in order to forbid them."""
    result = _check("Support the customer without judgement.",
                    ["Do not offer additional credit or a limit increase."],
                    Journey.GAMBLING_HARM)
    assert result.passed


def test_a_verbatim_policy_quotation_is_not_a_breach():
    """The no-LLM path composes advice by quoting the retrieved clauses, and
    those clauses describe scam patterns in order to warn about them."""
    clause = PolicyChunk(
        policy_reference="VP-C5", title="Scam and fraud vulnerability",
        text=('Where a customer may be the target of a scam (urgency, a "safe account" '
              "request, romance or investment pressure), slow the interaction down and "
              "refer to the fraud team."),
    )
    result = _check("Consider the prescribed adaptation.",
                    [f"{clause.title}: {clause.text}"],
                    Journey.FRAUD_SCAM, retrieved=[clause])
    assert result.passed


def test_sound_advice_passes():
    result = _check("Offer a gambling block and signpost GamCare.",
                    ["Apply a gambling block on the card.",
                     "Offer a cooling-off period on credit applications."],
                    Journey.GAMBLING_HARM)
    assert result.passed


def test_no_recommendation_is_not_a_breach():
    result = ProhibitedActionGuardrail().check(
        GuardrailContext(text="", recommendation=None, journey=Journey.GAMBLING_HARM))
    assert result.passed
