"""Tests for the multilingual half of the collections channel.

The claim being defended is narrow and worth stating precisely: **the customer
only ever hears wording a reviewer approved, in the language the call is being
held in.** Not "the agent speaks Arabic" — an agent that speaks Arabic by putting
approved English through a translation model speaks wording nobody signed off, in
a language the reviewer who signed off the English may not read.

That claim has three failure modes and there is a test for each:

* approved wording exists for a language but is not used;
* approved wording does not exist for a language and something is invented;
* approved wording is used, but wrapped in a frame from a different language, so
  the reply is in no language at all.

The last one is the easiest to ship by accident and the hardest to spot in a
demo, because the offer — the part a reader's eye goes to — looks perfect.
"""
from __future__ import annotations

import pytest

from guardiancx.agents.reply import (
    LANGUAGE_SCAFFOLD,
    approved_offers,
    can_compose_in,
    compose_reply,
)
from guardiancx.finance.taxonomy import FinancialContext, Journey
from guardiancx.rag.chunking import chunk_policy_file
from guardiancx.services.speech import locale_for, voice_for
from guardiancx.utils import language as L
from guardiancx.utils.types import (
    CaseDecision,
    Driver,
    DriverSignal,
    PolicyChunk,
    Recommendation,
    SentimentReading,
    VulnerabilityAssessment,
)

POLICY = "data/policies/banking_journeys.md"


@pytest.fixture(scope="module")
def clauses():
    return {c.ref: c for c in chunk_policy_file(POLICY)}


def _retrieved(clauses, *refs):
    """Clauses as the reply composer sees them — through the vector store's model."""
    out = []
    for ref in refs:
        c = clauses[ref]
        out.append(PolicyChunk(
            policy_reference=c.ref, title=c.title, text=c.text, score=0.9,
            journeys=c.journey_values, offers=c.offers,
            offers_by_language=c.offers_by_language))
    return out


def _decision():
    return CaseDecision(
        conversation_id="T", turn_index=0,
        assessment=VulnerabilityAssessment(
            signals=[DriverSignal(driver=Driver.RESILIENCE, score=0.8)]),
        recommendation=Recommendation(
            summary="Offer forbearance.", adaptations=["Offer forbearance."],
            citations=["VP-J1"], confidence=0.8),
    )


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text, expected", [
    ("I have missed two payments this month", "en"),
    ("لم أستطع دفع القسط هذا الشهر", "ar"),
    ("میں اس مہینے قسط ادا نہیں کر سکا", "ur"),
    ("मैं इस महीने भुगतान नहीं कर सका", "hi"),
    ("എനിക്ക് ഈ മാസം പണം അടയ്ക്കാൻ കഴിഞ്ഞില്ല", "ml"),
    ("wala akong pera para sa bayad ngayon po", "fil"),
])
def test_the_language_of_an_utterance_is_identified(text, expected):
    assert L.detect(text).language == expected


def test_arabic_and_urdu_are_separated_by_script_not_by_guess():
    """They share an alphabet. The separator is the handful of letters Urdu added
    — without it, every Urdu speaker is answered in Arabic by a confident system."""
    arabic = L.detect("لا أستطيع الدفع")
    urdu = L.detect("میں ادائیگی نہیں کر سکتا")
    assert arabic.language == "ar" and urdu.language == "ur"
    assert "Urdu-only" in urdu.basis


def test_a_short_utterance_is_not_evidence_of_a_language():
    """"Yes" is not a language sample, and treating it as one switches the call
    on the least informative turn in it."""
    assert L.detect("yes").confidence < 0.75


def test_a_weak_detection_does_not_switch_the_call():
    settled, why = L.resolve_call_language("ur", "ok")
    assert settled == "ur" and "continuing" in why


def test_a_strong_detection_does_switch_the_call():
    settled, why = L.resolve_call_language("en", "لم أستطع دفع القسط هذا الشهر")
    assert settled == "ar" and "switched" in why


def test_an_unmapped_language_falls_back_rather_than_raising():
    """A call in progress is not the place to discover an unmapped locale."""
    assert L.normalise("zz-ZZ") == "en"


def test_the_recogniser_candidate_list_respects_the_engine_limit():
    """Azure identifies at most four candidate locales at the start of a call.
    A longer list is not more coverage, it is a silently truncated one."""
    locales = L.autodetect_locales("ml", also=["ar", "ur", "hi", "fil"])
    assert len(locales) <= L.MAX_AUTODETECT_CANDIDATES
    assert locales[0] == "ml-IN"          # the customer's own language leads
    assert "en-GB" in locales             # English is always retained


# --------------------------------------------------------------------------- #
# Voice and locale
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("code, locale", [
    ("en", "en-GB"), ("ar", "ar-AE"), ("ur", "ur-PK"),
    ("hi", "hi-IN"), ("ml", "ml-IN"), ("fil", "fil-PH"),
])
def test_each_language_has_a_recogniser_locale_and_a_voice(code, locale):
    assert locale_for(code) == locale
    assert voice_for(code).startswith(locale)


def test_a_region_tag_resolves_to_the_same_voice():
    assert voice_for("ar-AE") == voice_for("ar")


# --------------------------------------------------------------------------- #
# Approved wording
# --------------------------------------------------------------------------- #
def test_a_clause_carries_its_approved_wording_per_language(clauses):
    offers, fell_back = clauses["VP-J1"].offers_in("ar")
    assert offers and not fell_back
    assert all(any("؀" <= ch <= "ۿ" for ch in o) for o in offers)


def test_a_missing_language_falls_back_to_english_and_says_so(clauses):
    """The flag is the point. Speaking English at a customer who asked for
    Malayalam is a service failure worth recording; inventing Malayalam is a
    compliance breach. Falling back and flagging is neither."""
    offers, fell_back = clauses["VP-J1"].offers_in("ml")
    assert fell_back is True
    assert offers == clauses["VP-J1"].offers


def test_the_fallback_flag_survives_the_vector_store_round_trip(clauses):
    """The composer sees a PolicyChunk, not a Chunk. If the flag is lost in
    between, the one fact a reviewer needs is lost exactly where it would have
    been recorded."""
    retrieved = _retrieved(clauses, "VP-J1")
    _, substituted = approved_offers(retrieved, "ml")
    assert substituted is True
    _, not_substituted = approved_offers(retrieved, "ar")
    assert not_substituted is False


# --------------------------------------------------------------------------- #
# Composition
# --------------------------------------------------------------------------- #
def test_a_call_composes_in_a_language_with_both_wording_and_scaffolding(clauses):
    assert can_compose_in("ar", _retrieved(clauses, "VP-J1", "VP-J14"))
    assert can_compose_in("ur", _retrieved(clauses, "VP-J1", "VP-J14"))


def test_a_language_with_no_approved_wording_for_the_clause_does_not_compose(clauses):
    """VP-J9 is authored in English only. That is a gap in the corpus, and the
    right response to a gap is English, not improvisation."""
    assert not can_compose_in("ar", _retrieved(clauses, "VP-J9"))


def test_a_language_with_no_approved_scaffolding_does_not_compose(clauses):
    """Malayalam has a voice and a recogniser but no signed-off frame. Offers
    alone are not a reply."""
    assert "ml" not in LANGUAGE_SCAFFOLD
    assert not can_compose_in("ml", _retrieved(clauses, "VP-J1"))


def test_an_arabic_reply_is_entirely_approved_arabic(clauses):
    """The failure this guards against is an Arabic offer inside an English
    frame — which reads as a success in a demo and as nothing at all to the
    customer."""
    reply = compose_reply(
        _decision(), FinancialContext(journey=Journey.ARREARS_COLLECTIONS),
        SentimentReading(distress=0.8), _retrieved(clauses, "VP-J1"), language="ar")

    approved = set(clauses["VP-J1"].offers_by_language["ar"])
    scaffold = set(LANGUAGE_SCAFFOLD["ar"].values())
    for sentence in [s.strip() for s in reply.split(".") if s.strip()]:
        assert any(sentence in phrase or phrase.strip(".").strip() in sentence
                   for phrase in approved | scaffold), sentence
    assert "I can" not in reply


def test_an_urdu_reply_uses_the_urdu_sentence_terminator(clauses):
    reply = compose_reply(
        _decision(), FinancialContext(journey=Journey.ARREARS_COLLECTIONS),
        None, _retrieved(clauses, "VP-J1"), language="ur")
    assert "۔" in reply          # the danda
    assert "Would that help?" not in reply


def test_a_language_without_approved_wording_gets_english_not_invention(clauses):
    reply = compose_reply(
        _decision(), FinancialContext(journey=Journey.ARREARS_COLLECTIONS),
        None, _retrieved(clauses, "VP-J1"), language="ml")
    assert "I can reduce your instalment" in reply


def test_the_localised_reply_states_no_figures(clauses):
    """A balance dropped into an approved Arabic sentence makes a sentence nobody
    approved. Offering to go through the account is the honest move."""
    class _Account:
        decision = "disclose"
        facts = ["AED 25,950 to bring it up to date"]
        refused = False
        refusal_reason = ""
        alternative = ""

    reply = compose_reply(
        _decision(), FinancialContext(journey=Journey.ARREARS_COLLECTIONS),
        None, _retrieved(clauses, "VP-J1"), account=_Account(), language="ar")
    assert "25,950" not in reply
    assert LANGUAGE_SCAFFOLD["ar"]["account"] in reply
