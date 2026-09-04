"""Tests for the two conduct guardrails the outbound channel added.

Both run on the **reply** — the sentence the customer actually hears — rather
than on the recommendation, and that is the point of them. The recommendation is
advice to a handler and a person reads it before anything happens. On a voice
call the reply is spoken, and nothing reads it first.

`credential_request` guards the challenge flow: the approved flow confirms
identity from what the bank already knows, and never asks the customer to reveal
a secret.

`approved_wording` guards the treatment strategy: "approved wording only,
without deviation" is either enforced on the sentence or it is decorative.
"""
from __future__ import annotations

import pytest

from guardiancx.guardrails.base import GuardrailContext
from guardiancx.guardrails.credential_request import CredentialRequestGuardrail
from guardiancx.guardrails.manager import default_reply_guardrails
from guardiancx.guardrails.wording import ApprovedWordingGuardrail
from guardiancx.rag.chunking import chunk_policy_file
from guardiancx.utils.types import PolicyChunk

POLICY = "data/policies/banking_journeys.md"


@pytest.fixture(scope="module")
def retrieved():
    clauses = {c.ref: c for c in chunk_policy_file(POLICY)}
    out = []
    for ref in ("VP-J1", "VP-J14"):
        c = clauses[ref]
        out.append(PolicyChunk(
            policy_reference=c.ref, title=c.title, text=c.text, score=0.9,
            journeys=c.journey_values, offers=c.offers,
            offers_by_language=c.offers_by_language))
    return out


def _credential(reply: str):
    return CredentialRequestGuardrail().check(GuardrailContext(text="", reply=reply))


def _wording(reply: str, retrieved, language="en", strategy="early_arrears_d1_7"):
    return ApprovedWordingGuardrail().check(GuardrailContext(
        text="", reply=reply, retrieved=retrieved,
        strategy=strategy, language=language))


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reply", [
    "What is your PIN?",
    "Can you read me the one-time passcode we just sent?",
    "Could you confirm the three digits on the back of the card?",
    "Just give me the full card number please.",
    "I need your internet banking username and password.",
    "Tell me the verification code on your phone.",
    "What's the CVV?",
])
def test_a_reply_asking_for_a_credential_is_blocked(reply):
    result = _credential(reply)
    assert not result.passed
    assert result.severity == "block"


@pytest.mark.parametrize("reply", [
    "Could you confirm the last four digits of your Emirates ID?",
    "Can I take your date of birth?",
    "I will never ask you for your PIN or your password.",
    "Don't share that code with anyone, not even someone claiming to be us.",
    "No one from the bank will ever ask for your one-time password.",
    "I can reduce your instalment to something you can actually manage.",
])
def test_the_approved_challenge_and_the_fraud_warning_both_pass(reply):
    """The sentences that warn about credentials are the ones a fraud call most
    needs. A guardrail that blocked them would remove the protection it exists
    to enforce."""
    assert _credential(reply).passed


def test_a_negation_in_a_previous_sentence_does_not_launder_the_request():
    """"We would never ask for this. Now, what is your PIN?" is the shape a
    prompt-injected or drifting model actually produces."""
    assert not _credential("We would never ask for this. Now, what is your PIN?").passed


def test_the_finding_names_the_credential_without_echoing_an_answer():
    result = _credential("Please read me the one-time passcode.")
    assert "one-time passcode" in result.detail
    assert result.data["credentials"][0]["credential"] == "one-time passcode"


def test_credentials_are_checked_before_anything_else_in_the_reply_set():
    """Ordered by what cannot be undone. A customer may answer before the
    sentence has finished."""
    names = [g.name for g in default_reply_guardrails()]
    assert names[0] == "credential_request"
    assert names.index("approved_wording") < names.index("customer_clarity")


# --------------------------------------------------------------------------- #
# Approved wording
# --------------------------------------------------------------------------- #
def test_an_approved_offer_passes(retrieved):
    assert _wording(
        "I can reduce your instalment to something you can actually manage.",
        retrieved).passed


def test_two_approved_offers_joined_into_one_sentence_still_pass(retrieved):
    """The composer joins two offers and drops the repeated "I can", because two
    sentences each opening "I can" read as a list rather than a choice. That is
    composition of approved wording, not deviation from it."""
    assert _wording(
        "I can reduce your instalment to something you can actually manage, and "
        "freeze the interest and charges while we sort out a plan.",
        retrieved).passed


def test_an_invented_offer_is_blocked(retrieved):
    result = _wording("I understand. I can write off half the balance today.", retrieved)
    assert not result.passed
    assert result.severity == "block"
    assert "write off half the balance" in result.data["unapproved"][0]


def test_an_invented_offer_beside_approved_ones_is_still_caught(retrieved):
    """The realistic failure: a model keeps the approved offers and adds one."""
    result = _wording(
        "I can reduce your instalment to something you can actually manage. "
        "I can also make sure this never affects your visa.",
        retrieved)
    assert not result.passed


def test_a_reply_that_makes_no_commitment_passes(retrieved):
    assert _wording("Thank you for confirming that. What happened this month?",
                    retrieved).passed


def test_a_commitment_with_nothing_retrieved_is_blocked(retrieved):
    """A promise with no clause behind it is the case most worth catching."""
    result = _wording("I can arrange a six-month payment holiday for you.", [])
    assert not result.passed
    assert "no approved wording was retrieved" in result.detail


def test_an_inbound_call_with_no_strategy_is_not_scripted(retrieved):
    """This is not a general style rule. An inbound call was never meant to run
    from a script, and applying the rule there would block honest replies."""
    assert _wording("I can look at whatever would help most.", retrieved,
                    strategy=None).passed


def test_approved_arabic_passes_and_invented_arabic_does_not(retrieved):
    assert _wording(
        "يمكنني تخفيض القسط الشهري إلى مبلغ تستطيع سداده فعلاً. هل يناسبك ذلك؟",
        retrieved, language="ar").passed
    assert not _wording("يمكنني شطب نصف المبلغ اليوم.", retrieved, language="ar").passed


def test_approved_urdu_passes_and_invented_urdu_does_not(retrieved):
    assert _wording(
        "میں آپ کی ماہانہ قسط کم کر سکتی ہوں تاکہ آپ آسانی سے ادا کر سکیں۔",
        retrieved, language="ur").passed
    assert not _wording(
        "میں آج آپ کا آدھا قرض معاف کر سکتی ہوں۔",
        retrieved, language="ur").passed


def test_an_english_commitment_inside_a_non_english_call_is_still_checked(retrieved):
    """A fallback reply, or a customer who switched. Either way the sentence
    commits the bank and has to be approved."""
    assert not _wording("I can waive the whole balance.", retrieved,
                        language="ar").passed
