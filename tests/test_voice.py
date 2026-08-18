"""Tests for the live-call layer: endpointing, prosody and streaming PII.

These are the parts of the system that run *during* a call, where a wrong answer
is not a bad row in a report but an agent talking over a bereaved customer or a
sort code reaching a log. They are tested against the failure they exist to
prevent, not just their happy path.
"""
from __future__ import annotations

import array
import io
import math
import wave

import pytest

from guardiancx.voice import endpointing, prosody
from guardiancx.voice.live_pii import LivePIIRedactor, redact


# --------------------------------------------------------------------------- #
# End-of-utterance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "my husband passed away in March and",          # trailing conjunction
    "I wanted to ask about the",                    # trailing article
    "I think it was, um",                           # trailing filler
    "my sort code is",                              # about to dictate a value
    "it's 09 01",                                   # digits still coming
    "I need to talk about my mortgage because",     # subordinating conjunction
])
def test_incomplete_utterances_score_low(text):
    score, _ = endpointing.score_completeness(text)
    assert score < 0.35, f"{text!r} scored {score}"


@pytest.mark.parametrize("text", [
    "I've lost my job and I can't make this month's payment.",
    "Can you help me with the arrears?",
    "yes",
    "I don't know",
    "My husband died last month and I am struggling to keep up with everything",
])
def test_complete_utterances_score_high(text):
    score, _ = endpointing.score_completeness(text)
    assert score >= 0.5, f"{text!r} scored {score}"


def test_unfinished_speech_waits_longer_than_finished_speech():
    """The whole point of semantic endpointing: the wait adapts to the words."""
    finished, _ = endpointing.score_completeness("I can't pay this month.")
    unfinished, _ = endpointing.score_completeness("I can't pay this month because")
    assert (endpointing.required_silence_ms(unfinished)
            > endpointing.required_silence_ms(finished))


def test_distress_extends_every_threshold():
    """A customer who is struggling gets more room, never less."""
    calm = endpointing.required_silence_ms(0.5, distress=0.0)
    upset = endpointing.required_silence_ms(0.5, distress=0.9)
    assert upset > calm


def test_pause_mid_sentence_does_not_commit_the_turn():
    decision = endpointing.detect("my husband passed away and", silence_ms=900)
    assert not decision.complete


def test_long_silence_eventually_commits_even_mid_sentence():
    """A hung turn has to end somehow — but only well past the normal window."""
    decision = endpointing.detect("my husband passed away and", silence_ms=9000)
    assert decision.complete
    assert decision.source == "timeout"


def test_finished_sentence_commits_on_a_short_pause():
    decision = endpointing.detect("I can't pay this month.", silence_ms=1200)
    assert decision.complete


def test_no_timing_information_falls_back_to_semantics():
    """The text channel has no silence to measure."""
    assert endpointing.detect("Can you help me?", silence_ms=0).complete
    assert not endpointing.detect("I wanted to ask about the", silence_ms=0).complete


# --------------------------------------------------------------------------- #
# Live PII
# --------------------------------------------------------------------------- #
def test_spoken_sort_code_is_redacted():
    result = redact("my sort code is oh nine, oh one, double two")
    assert "nine" not in result.text.lower()
    assert result.spoken_runs == 1


def test_spoken_digits_are_redacted():
    result = redact("the account number is one two three four five six seven eight")
    assert "three four" not in result.text
    assert result.total >= 1


def test_ordinary_numbers_in_conversation_survive():
    """Over-redaction destroys the transcript a handler has to read."""
    result = redact("I've got two cards and three direct debits")
    assert "two cards" in result.text
    assert result.total == 0


def test_written_uk_identifiers_are_masked():
    # AB…C is a valid NI prefix/suffix pair; the letters D, F, I, Q, U and V are
    # never issued, and the pattern excludes them deliberately.
    text = ("my NI number is AB 12 34 56 C, I live at SW1A 1AA "
            "and I was born on 12/03/1958")
    result = redact(text)
    assert "AB 12 34 56 C" not in result.text
    assert "SW1A 1AA" not in result.text
    assert "12/03/1958" not in result.text
    assert {"ni_number", "postcode", "date_of_birth"} <= set(result.redactions)


def test_announcement_is_flagged_before_the_value_is_spoken():
    """The guard fires on the announcement, not on the disclosure."""
    result = redact("hold on, my account number is")
    assert result.expecting == "account"


def test_credential_request_is_flagged():
    result = redact("do you want my PIN")
    assert result.security_request is True


def test_value_split_across_two_fragments_is_still_caught():
    """A dictated figure rarely arrives in one partial result."""
    redactor = LivePIIRedactor()
    first = redactor.feed("right, my sort code is")
    assert first.expecting == "sort_code"
    second = redactor.feed("oh nine oh one")
    assert "nine" not in second.text.lower()


def test_redactor_accumulates_totals_across_a_call():
    redactor = LivePIIRedactor()
    redactor.feed("my email is jane@example.com")
    redactor.feed("and my postcode is SW1A 1AA")
    assert set(redactor.total_redactions) == {"email", "postcode"}


# --------------------------------------------------------------------------- #
# Prosody
# --------------------------------------------------------------------------- #
def _tone(frequency: float, seconds: float, amplitude: float, rate: int = 16000) -> bytes:
    """A synthetic voiced signal — enough for the analyser to find a pitch."""
    samples = array.array("h")
    for i in range(int(rate * seconds)):
        value = amplitude * math.sin(2 * math.pi * frequency * i / rate)
        samples.append(int(max(-32767, min(32767, value))))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples.tobytes())
    return buf.getvalue()


def test_analyse_recovers_the_pitch_of_a_known_tone():
    signals = prosody.analyse(_tone(180.0, 1.2, 9000))
    assert signals.available
    assert 160 <= signals.pitch_hz <= 200


def test_malformed_audio_degrades_rather_than_raises():
    signals = prosody.analyse(b"not a wav file at all")
    assert not signals.available
    assert signals.summary() == "no voice signal (text channel)"


def test_silence_is_reported_as_unavailable():
    signals = prosody.analyse(_tone(180.0, 0.1, 9000))
    assert not signals.available


def test_derived_indicators_stay_bounded():
    signals = prosody.derive(
        duration_s=5.0, loudness=9.9, loudness_variation=9.9,
        pitch_hz=900, pitch_variation=9.9, speech_ratio=2.0,
        transcript="word " * 400,
    )
    for value in (signals.agitation, signals.tremor, signals.hesitancy, signals.distress):
        assert 0.0 <= value <= 1.0


def test_shaking_halting_voice_reads_as_more_distressed_than_a_loud_one():
    """Loudness alone must not dominate — an angry caller is not a vulnerable one."""
    loud = prosody.derive(duration_s=4, loudness=0.9, loudness_variation=0.1,
                          pitch_hz=200, pitch_variation=0.05, speech_ratio=0.95)
    shaking = prosody.derive(duration_s=4, loudness=0.35, loudness_variation=0.8,
                             pitch_hz=250, pitch_variation=0.45, speech_ratio=0.45)
    assert shaking.distress > loud.distress


def test_browser_measurements_produce_the_same_shape():
    signals = prosody.from_browser({
        "duration_s": 3.4, "loudness": 0.4, "loudness_variation": 0.3,
        "pitch_hz": 210, "pitch_variation": 0.2, "speech_ratio": 0.7,
    }, "I can't make this payment")
    assert signals.available
    assert signals.words_per_minute > 0
    assert "browser" in signals.note


def test_browser_measurements_missing_returns_unavailable():
    assert not prosody.from_browser({}, "hello").available
