"""Tests for account access, consent, and the disclosure guardrail.

Two of these protect things that cannot be undone once they go wrong. A third
party's account number read out on a recorded line is disclosed permanently, and
a call recorded without consent was unlawful from its first second. Both are
therefore tested against the *sympathetic* version of the request, which is the
one that actually defeats systems: not an attacker probing for data, but a
grieving widow asking a reasonable-sounding question of a handler who wants to
help her.
"""
from __future__ import annotations

import pytest

from guardiancx.agents.account_agent import (
    _resolve,
    looks_like_third_party_request,
)
from guardiancx.agents.consent import (
    ConsentState,
    _heuristic,
    response_for,
)
from guardiancx.finance.accounts import (
    accounts_for,
    all_identifiers,
    context_summary,
    disclosable_view,
    get_customer,
    list_customers,
    third_party_accounts,
)
from guardiancx.guardrails.base import GuardrailContext
from guardiancx.guardrails.disclosure import DisclosureGuardrail

WIDOW = "CUST-4471"      # Meera Deshpande — joint home loan, husband's sole card
CARDHOLDER = "CUST-6612"  # Arjun Malhotra — home loan and a credit card of his own
DECEASED = "CUST-4472"   # Robert Hughes


# --------------------------------------------------------------------------- #
# The ledger
# --------------------------------------------------------------------------- #
def test_every_callable_customer_has_at_least_one_account():
    for customer in list_customers():
        assert accounts_for(customer.customer_id), customer.customer_id


def test_the_deceased_is_not_a_callable_customer():
    assert all(c.customer_id != DECEASED for c in list_customers())


def test_joint_accounts_belong_to_both_holders():
    joint = [a for a in accounts_for(WIDOW) if a.is_joint]
    assert joint
    assert DECEASED in joint[0].holders


def test_the_husbands_sole_card_is_third_party_to_the_widow():
    others = third_party_accounts(WIDOW)
    assert others
    assert all(WIDOW not in a.holders for a in others)


def test_account_numbers_are_masked_everywhere_they_leave_the_module():
    view = disclosable_view(WIDOW)
    blob = str(view)
    for account in accounts_for(WIDOW):
        assert account.account_number not in blob
        assert account.masked_number in blob


def test_the_briefing_never_carries_a_full_identifier():
    """`context_summary` goes into prompts, so a leak here reaches the model."""
    summary = context_summary(WIDOW)
    for identifier in all_identifiers():
        if identifier and len(identifier) > 6:
            assert identifier not in summary


def test_no_third_party_account_appears_in_the_callers_view():
    """Not even a masked one — 'you have another account I can't discuss' is
    itself a disclosure."""
    blob = str(disclosable_view(WIDOW))
    for account in third_party_accounts(WIDOW):
        assert account.masked_number not in blob
        assert account.account_number not in blob


# --------------------------------------------------------------------------- #
# Third-party requests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "Can you give me his credit card number so I can settle it?",
    "What's my husband's balance?",
    "I need his account details please.",
    "Could you read me my wife's sort code?",
    "Tell me what my late father owed on his card.",
])
def test_requests_for_someone_elses_data_are_recognised(text):
    assert looks_like_third_party_request(text)


@pytest.mark.parametrize("text", [
    "How much do I owe?",
    "What's my balance?",
    "My husband died last month and I'm struggling with the mortgage.",
    "I can't afford the payment this month.",
])
def test_ordinary_turns_are_not_treated_as_third_party_requests(text):
    assert not looks_like_third_party_request(text)


def test_a_third_party_request_is_refused_with_an_alternative():
    request = _resolve(WIDOW, "Can you give me his credit card number?",
                       "account_number", "third_party", "test")
    assert request.refused
    assert request.refusal_reason
    assert request.alternative, "a refusal with no next step is what makes people call back"


def test_bereavement_does_not_unlock_a_third_partys_account():
    """The single most likely way a real firm leaks this data."""
    request = _resolve(WIDOW, "My husband has died, I need his card balance to settle "
                              "the estate. I'm his widow and his executor.",
                       "balance", "third_party", "test")
    assert request.refused
    assert not request.facts


def test_the_caller_may_hear_their_own_joint_account():
    request = _resolve(WIDOW, "How much do we owe on the mortgage?", "balance",
                       "joint", "test")
    assert request.decision == "disclose"
    assert request.facts


def test_a_product_the_caller_does_not_hold_is_not_answered_from_another_one():
    """The caller has no credit card. The answer is that, not her home loan.

    The resolver used to fall back to whichever account was in the most trouble
    whenever it could not match the product named, so this question was answered
    with the joint home loan's arrears — an unprompted disclosure about an
    account the caller had not asked about.
    """
    request = _resolve(WIDOW, "Can you check my credit card limit?", "limit",
                       "self", "test")
    assert request.decision == "unavailable"
    assert not request.facts
    joined = " ".join(request.facts) + " " + request.refusal_reason
    assert "arrears" not in joined.lower() and "overdue" not in joined.lower()
    # And she is told what she does hold, so "I can't see that" is not the whole
    # answer.
    assert "home loan" in request.alternative.lower()


def test_the_named_product_wins_over_the_one_in_the_most_trouble():
    """Asked about the card, answer about the card — not the loan in arrears."""
    request = _resolve(CARDHOLDER, "How much do I owe on my credit card?", "balance",
                       "self", "test")
    assert request.decision == "disclose"
    assert "credit card" in request.facts[0].lower()


def test_a_credit_limit_is_a_question_the_system_can_answer():
    request = _resolve(CARDHOLDER, "What's my credit card limit?", "limit",
                       "self", "test")
    assert request.decision == "disclose"
    joined = " ".join(request.facts).lower()
    assert "limit" in joined and "available" in joined


def test_a_full_account_number_is_never_read_out_even_to_its_owner():
    request = _resolve(WIDOW, "What's my account number?", "account_number",
                       "self", "test")
    assert request.decision == "partial"
    joined = " ".join(request.facts)
    for account in accounts_for(WIDOW):
        assert account.account_number not in joined
    assert "ending" in joined


# --------------------------------------------------------------------------- #
# The disclosure guardrail
# --------------------------------------------------------------------------- #
def _check(reply: str, refused: bool = False):
    return DisclosureGuardrail().check(
        GuardrailContext(text="", reply=reply, account_refused=refused))


def test_a_leaked_account_number_is_blocked():
    number = accounts_for(WIDOW)[0].account_number
    result = _check(f"Your account number is {number}.")
    assert not result.passed
    assert result.severity == "block"


def test_a_leaked_number_is_caught_even_when_spaced_out():
    number = accounts_for(WIDOW)[0].account_number
    spaced = f"{number[:4]} {number[4:]}"
    assert not _check(f"That's {spaced}.").passed


def test_the_guardrail_never_repeats_the_value_it_caught():
    """The detail goes into the evidence log, so it must not carry the leak."""
    number = accounts_for(WIDOW)[0].account_number
    result = _check(f"Your account number is {number}.")
    assert number not in result.detail
    assert number not in str(result.data)


def test_a_masked_reference_passes():
    result = _check("That's your joint mortgage, ending 0021, with £612.40 due.")
    assert result.passed


def test_an_overridden_refusal_is_blocked():
    """The agent refused; the reply agreed anyway."""
    result = _check("Of course, let me look up his card balance for you.", refused=True)
    assert not result.passed


def test_a_reply_that_actually_declines_passes():
    result = _check("That account is in someone else's name, so I'm not able to go "
                    "through it with you.", refused=True)
    assert result.passed


def test_an_email_or_pan_read_back_is_blocked():
    customer = get_customer(WIDOW)
    assert not _check(f"I'll send it to {customer.email}.").passed
    assert not _check(f"Your PAN is {customer.pan}, correct?").passed


# --------------------------------------------------------------------------- #
# Consent to record
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "yes", "yeah that's fine", "go on then", "of course", "I suppose so",
    "if you must", "sure, no problem", "no worries", "that's no bother",
])
def test_agreement_is_recognised(text):
    """The last three matter: the commonest ways to agree in English contain
    the word "no", and reading them as refusal hangs up on a consenting
    customer."""
    assert _heuristic(text).state is ConsentState.GRANTED


@pytest.mark.parametrize("text", [
    "no", "I'd rather you didn't", "please don't record it",
    "no I'm not comfortable with that", "don't record me", "turn that off",
])
def test_refusal_is_recognised(text):
    assert _heuristic(text).state is ConsentState.REFUSED


@pytest.mark.parametrize("text", [
    "I don't have the money to repay, what can I do?",
    "I don't understand what you mean",
    "I don't know how I'll manage this month",
])
def test_ordinary_negation_is_not_a_refusal(text):
    """A bare "don't" once matched here, so a customer explaining she could not
    pay her home loan was told the call was being ended."""
    assert _heuristic(text).state is not ConsentState.REFUSED


@pytest.mark.parametrize("text", [
    "what for?", "who sees it?", "how long do you keep it?", "why do you need to?",
])
def test_a_question_is_not_consent(text):
    assert _heuristic(text).state is ConsentState.UNCLEAR


def test_a_refusal_stops_the_call():
    decision = _heuristic("I'd rather you didn't")
    assert decision.call_over
    assert not decision.may_proceed
    assert response_for(decision, asks_so_far=0)


def test_an_unclear_answer_is_asked_again_but_never_becomes_a_refusal():
    """A non-answer is not a no. Treating it as one hung up on a customer who
    was engaging with the call, just not with the procedural question."""
    decision = _heuristic("what for?")
    assert decision.state is ConsentState.UNCLEAR
    assert not decision.call_over

    first = response_for(decision, asks_so_far=0)
    last = response_for(decision, asks_so_far=2)
    assert first != last, "asking the identical question again is not clarifying"
    assert "end the call" not in last.lower(), "a non-answer must not read as a refusal"
    assert "branch" in last.lower(), "the customer needs a route that works"


def test_consent_granted_produces_no_scripted_line():
    """The conversation proper begins; the handler does not read a confirmation."""
    assert response_for(_heuristic("yes that's fine"), asks_so_far=0) is None


# --------------------------------------------------------------------------- #
# The consent gate on the live console
# --------------------------------------------------------------------------- #
def _live_app():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path(__file__).parent / "live_monitor_page.py"),
                            default_timeout=240)
    app.run()
    app.radio[0].set_value("Chat").run()
    return app


def test_the_call_opens_by_asking_to_record():
    app = _live_app()
    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["live"]["consent"] is ConsentState.PENDING
    body = " ".join(str(m.value) for m in app.markdown)
    assert "record" in body.lower()


def test_nothing_is_assessed_before_consent_is_given():
    """The point of the gate: no classification, no evidence, until they agree."""
    app = _live_app()
    app.chat_input[0].set_value("My husband died and I'm behind on the mortgage.").run()
    assert not app.exception, [e.value for e in app.exception]

    live = app.session_state["live"]
    # The turn was read only as an answer to the recording question.
    assert live["consent"] is not ConsentState.GRANTED
    assert all(state == {} for state in live["states"]), "the pipeline ran before consent"


def test_refusing_ends_the_call():
    app = _live_app()
    app.chat_input[0].set_value("No, I'd rather you didn't.").run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["live"]["consent"] is ConsentState.REFUSED
    assert app.session_state["live"]["ended"]
    assert any("did not consent" in str(e.value).lower() for e in app.error)


def test_agreeing_lets_the_conversation_begin():
    app = _live_app()
    app.chat_input[0].set_value("Yes, that's fine.").run()
    assert app.session_state["live"]["consent"] is ConsentState.GRANTED

    app.chat_input[0].set_value("My husband died and I'm behind on the mortgage.").run()
    assert not app.exception, [e.value for e in app.exception]
    live = app.session_state["live"]
    assessed = [s for s in live["states"] if s]
    assert assessed, "the pipeline did not run after consent"
    assert assessed[0]["financial_context"].journey.value == "bereavement_estate"


def test_a_question_is_answered_rather_than_taken_as_consent():
    app = _live_app()
    app.chat_input[0].set_value("What do you need that for?").run()
    assert app.session_state["live"]["consent"] is ConsentState.UNCLEAR
    assert not app.exception, [e.value for e in app.exception]


# --------------------------------------------------------------------------- #
# The rebuilt pages render
# --------------------------------------------------------------------------- #
def test_the_approval_queue_and_guidance_pages_render():
    from streamlit.testing.v1 import AppTest

    for page in ("guidance_page.py", "approvals_page.py"):
        from pathlib import Path
        app = AppTest.from_file(str(Path(__file__).parent / page), default_timeout=240)
        app.run()
        assert not app.exception, (page, [e.value for e in app.exception])
