"""Prosody analysis — reading distress from the call audio, not the transcript.

A transcript flattens a call. "I don't know what I'm going to do" reads the same
whether it was said evenly or through tears, and a vulnerability system that only
sees text will score them identically. On a real call the handler hears the
difference immediately, and FG21/1 expects the firm to act on it.

This module recovers some of that signal from the raw audio using standard-library
DSP only — no numpy, no librosa, no native build. Five measurements, each chosen
because it is robust on 8-16 kHz telephony-grade speech and because it maps onto
something a handler would actually notice:

| Measurement        | How                                   | What it indicates          |
|--------------------|---------------------------------------|----------------------------|
| Loudness           | RMS energy over 30 ms frames          | agitation / withdrawal     |
| Loudness variation | coefficient of variation of frame RMS | emotional instability      |
| Pitch              | autocorrelation F0 per voiced frame   | tension (raised pitch)     |
| Pitch variation    | std/mean of F0                        | tremor, unsteady voice     |
| Speech rate proxy  | voiced-frame ratio and pause ratio    | pressured speech / halting |

**These are indicators, never conclusions.** The output is deliberately typed as
signals with a confidence, is always fused with the text before it influences
anything, and can never on its own trigger a vulnerability finding — a loud voice
is not a vulnerable customer. Everything is computed locally; the audio is never
stored.

Two capture paths feed the same derivation. A push-to-talk recording is analysed
here from its WAV bytes; a live call measures the same quantities in the browser
with the Web Audio API, frame by frame, and posts the raw measurements up. Both
converge on :func:`derive`, so a signal means exactly the same thing whichever
way the audio arrived — and the thresholds live in one place rather than being
reimplemented in JavaScript and drifting.
"""
from __future__ import annotations

import array
import io
import math
import statistics
import sys
import wave

from pydantic import BaseModel

from ..utils.logging import get_logger

log = get_logger("voice.prosody")

_FRAME_MS = 30
# Human speech F0 sits roughly here; searching outside it wastes cycles and
# invites octave errors on telephony-band audio.
_F0_MIN_HZ = 70
_F0_MAX_HZ = 400
# Frames quieter than this fraction of the utterance's peak RMS are treated as
# pauses rather than speech. Relative, so it survives varying mic gain.
_SILENCE_RATIO = 0.18


class VoiceSignals(BaseModel):
    """Acoustic measurements from one captured utterance."""

    duration_s: float = 0.0
    loudness: float = 0.0            # mean frame RMS, normalised 0-1
    loudness_variation: float = 0.0  # coefficient of variation
    pitch_hz: float = 0.0            # mean F0 over voiced frames
    pitch_variation: float = 0.0     # std/mean of F0
    speech_ratio: float = 0.0        # voiced frames / total frames
    pause_ratio: float = 0.0         # 1 - speech_ratio
    words_per_minute: float = 0.0    # requires the transcript; 0 when unknown

    # Derived, bounded 0-1 indicators
    agitation: float = 0.0           # loud, fast, high-pitched
    tremor: float = 0.0              # unsteady pitch and loudness
    hesitancy: float = 0.0           # long pauses, halting delivery
    distress: float = 0.0            # fused indicator

    available: bool = False
    note: str = ""

    def summary(self) -> str:
        """A one-line description for the detection prompt and the trace."""
        if not self.available:
            return "no voice signal (text channel)"
        parts = [
            f"{self.duration_s:.1f}s",
            f"loudness {self.loudness:.2f} (var {self.loudness_variation:.2f})",
            f"pitch {self.pitch_hz:.0f}Hz (var {self.pitch_variation:.2f})",
            f"pauses {self.pause_ratio:.0%}",
        ]
        if self.words_per_minute:
            parts.append(f"{self.words_per_minute:.0f} wpm")
        parts.append(f"agitation {self.agitation:.2f} · tremor {self.tremor:.2f} · "
                     f"hesitancy {self.hesitancy:.2f}")
        return " · ".join(parts)

    @property
    def flags(self) -> list[str]:
        """Human-readable indicators for the live signal rail."""
        out = []
        if self.agitation >= 0.6:
            out.append("raised, pressured voice")
        if self.tremor >= 0.6:
            out.append("unsteady voice")
        if self.hesitancy >= 0.6:
            out.append("halting, long pauses")
        if self.pitch_hz and self.pitch_variation >= 0.35:
            out.append("wide pitch swing")
        return out


# --------------------------------------------------------------------------- #
# WAV decoding
# --------------------------------------------------------------------------- #
def _decode(data: bytes) -> tuple[array.array, int]:
    """16-bit WAV bytes → (mono samples, sample rate)."""
    with wave.open(io.BytesIO(data), "rb") as src:
        channels, width, rate = src.getnchannels(), src.getsampwidth(), src.getframerate()
        frames = src.readframes(src.getnframes())
    if width != 2 or not frames:
        raise ValueError("Expected non-empty 16-bit PCM WAV.")

    samples = array.array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":  # WAV is little-endian on the wire
        samples.byteswap()

    if channels > 1:
        samples = array.array("h", (
            sum(samples[i:i + channels]) // channels
            for i in range(0, len(samples) - channels + 1, channels)
        ))
    return samples, rate


def _rms(window) -> float:
    if not window:
        return 0.0
    return math.sqrt(sum(float(s) * s for s in window) / len(window))


# A candidate lag is only accepted as pitch above this normalised correlation.
# Voiced speech is strongly periodic; noise and fricatives are not, so this is
# what stops line hiss being reported as a 300 Hz voice.
_VOICING_FLOOR = 0.5


def _estimate_f0(window: array.array, rate: int) -> float:
    """Autocorrelation pitch estimate for one frame, 0.0 if unvoiced.

    The correlation is normalised by the energy of *both* overlapping segments,
    not by the overlap length. Dividing by length looks like a normalisation but
    is not one: the overlap shrinks as the lag grows, so the quotient inflates
    with lag and the search is biased toward long lags — that is, toward reporting
    a pitch below the true one. Energy normalisation is the standard NCCF form,
    is bounded to [-1, 1], and makes the acceptance threshold a real correlation
    rather than an energy-dependent guess.
    """
    n = len(window)
    min_lag = max(2, int(rate / _F0_MAX_HZ))
    max_lag = min(n - 1, int(rate / _F0_MIN_HZ))
    if max_lag <= min_lag:
        return 0.0

    mean = sum(window) / n
    centred = [float(s) - mean for s in window]
    if sum(v * v for v in centred) <= 0:
        return 0.0

    best_lag, best_corr = 0, 0.0
    for lag in range(min_lag, max_lag + 1):
        overlap = n - lag
        corr = energy_a = energy_b = 0.0
        for i in range(overlap):
            a, b = centred[i], centred[i + lag]
            corr += a * b
            energy_a += a * a
            energy_b += b * b
        denominator = math.sqrt(energy_a * energy_b)
        if denominator <= 0:
            continue
        norm = corr / denominator
        if norm > best_corr:
            best_corr, best_lag = norm, lag

    if not best_lag or best_corr < _VOICING_FLOOR:
        return 0.0
    return rate / best_lag


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _scale(value: float, low: float, high: float) -> float:
    """Map a raw measurement onto 0-1 across the band that matters."""
    if high <= low:
        return 0.0
    return _clamp((value - low) / (high - low))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def analyse(wav_bytes: bytes, transcript: str = "") -> VoiceSignals:
    """Measure prosody from one captured utterance.

    Returns an unavailable-but-valid result rather than raising, so a malformed
    or silent recording degrades the call to text rather than breaking it.
    """
    try:
        samples, rate = _decode(wav_bytes)
    except Exception as exc:  # noqa: BLE001
        log.info("Prosody analysis skipped: %s", exc)
        return VoiceSignals(note=f"audio could not be analysed ({exc})")

    duration = len(samples) / rate if rate else 0.0
    if duration < 0.35:
        return VoiceSignals(duration_s=round(duration, 2),
                            note="recording too short to analyse")

    frame_len = max(1, int(rate * _FRAME_MS / 1000))
    frames = [samples[i:i + frame_len]
              for i in range(0, len(samples) - frame_len + 1, frame_len)]
    if not frames:
        return VoiceSignals(duration_s=round(duration, 2), note="no analysable frames")

    energies = [_rms(f) for f in frames]
    peak = max(energies) or 1.0
    threshold = peak * _SILENCE_RATIO
    voiced = [(f, e) for f, e in zip(frames, energies) if e >= threshold]

    speech_ratio = len(voiced) / len(frames)
    if not voiced:
        return VoiceSignals(duration_s=round(duration, 2), note="silence only")

    voiced_energies = [e for _, e in voiced]
    mean_energy = statistics.fmean(voiced_energies)
    # 32768 is full scale for 16-bit PCM.
    loudness = _clamp(mean_energy / 32768.0 * 6.0)
    loudness_variation = (
        statistics.pstdev(voiced_energies) / mean_energy if mean_energy else 0.0
    )

    # Pitch on a bounded sample of voiced frames — autocorrelation is O(n·lags)
    # and a live call cannot afford to analyse every frame of a long utterance.
    sample_frames = voiced[:: max(1, len(voiced) // 60)][:60]
    pitches = [p for p, _ in ((_estimate_f0(f, rate), e) for f, e in sample_frames) if p]
    pitch_hz = statistics.fmean(pitches) if pitches else 0.0
    pitch_variation = (
        statistics.pstdev(pitches) / pitch_hz if len(pitches) > 1 and pitch_hz else 0.0
    )

    return derive(
        duration_s=duration,
        loudness=loudness,
        loudness_variation=loudness_variation,
        pitch_hz=pitch_hz,
        pitch_variation=pitch_variation,
        speech_ratio=speech_ratio,
        transcript=transcript,
        note="measured from call audio",
    )


def derive(duration_s: float, loudness: float, loudness_variation: float,
           pitch_hz: float, pitch_variation: float, speech_ratio: float,
           transcript: str = "", note: str = "measured from call audio") -> VoiceSignals:
    """Turn raw acoustic measurements into the bounded 0-1 indicators.

    The single place the thresholds live. Both capture paths — WAV analysis here
    and Web Audio measurement in the browser — call this, so "agitation 0.7"
    means the same thing on a push-to-talk recording and a live call.
    """
    words = len(transcript.split())
    wpm = (words / duration_s * 60.0) if (words and duration_s) else 0.0

    # Bands are set for telephony speech: ~150 Hz is a relaxed male voice, ~230 Hz
    # a relaxed female one, and sustained elevation above ~260 Hz on either
    # reads as tension. Speech above ~185 wpm is pressured; below ~110 is halting.
    pitch_tension = _scale(pitch_hz, 190.0, 300.0) if pitch_hz else 0.0
    rate_pressure = _scale(wpm, 150.0, 210.0) if wpm else 0.0
    agitation = _clamp(0.45 * _scale(loudness, 0.30, 0.85)
                       + 0.35 * pitch_tension
                       + 0.20 * rate_pressure)
    tremor = _clamp(0.55 * _scale(pitch_variation, 0.18, 0.45)
                    + 0.45 * _scale(loudness_variation, 0.35, 0.85))
    pause_ratio = 1.0 - _clamp(speech_ratio)
    slow_speech = _scale(130.0 - wpm, 0.0, 60.0) if wpm else 0.0
    hesitancy = _clamp(0.6 * _scale(pause_ratio, 0.28, 0.62) + 0.4 * slow_speech)

    # Distress is deliberately conservative: tremor and hesitancy weigh more than
    # loudness, because an angry customer is not necessarily a vulnerable one
    # while a shaking, halting voice very often is.
    distress = _clamp(0.45 * tremor + 0.30 * hesitancy + 0.25 * agitation)

    return VoiceSignals(
        duration_s=round(duration_s, 2),
        loudness=round(_clamp(loudness), 3),
        loudness_variation=round(max(0.0, loudness_variation), 3),
        pitch_hz=round(max(0.0, pitch_hz), 1),
        pitch_variation=round(max(0.0, pitch_variation), 3),
        speech_ratio=round(_clamp(speech_ratio), 3),
        pause_ratio=round(pause_ratio, 3),
        words_per_minute=round(wpm, 1),
        agitation=round(agitation, 3),
        tremor=round(tremor, 3),
        hesitancy=round(hesitancy, 3),
        distress=round(distress, 3),
        available=True,
        note=note,
    )


def from_browser(measurements: dict, transcript: str = "") -> VoiceSignals:
    """Build signals from measurements taken in the browser during a live call.

    The Web Audio API gives the page the same time-domain frames this module
    reads out of a WAV, so the browser measures loudness, its variation, an
    autocorrelation pitch estimate and the voiced-frame ratio, and posts those
    six numbers up. Deriving here rather than there keeps one definition of what
    the indicators mean.
    """
    if not measurements or not measurements.get("duration_s"):
        return empty("live call — no acoustic measurements returned")
    try:
        return derive(
            duration_s=float(measurements.get("duration_s", 0.0)),
            loudness=float(measurements.get("loudness", 0.0)),
            loudness_variation=float(measurements.get("loudness_variation", 0.0)),
            pitch_hz=float(measurements.get("pitch_hz", 0.0)),
            pitch_variation=float(measurements.get("pitch_variation", 0.0)),
            speech_ratio=float(measurements.get("speech_ratio", 0.0)),
            transcript=transcript,
            note="measured live in the browser",
        )
    except (TypeError, ValueError) as exc:
        log.info("Browser prosody measurements unusable: %s", exc)
        return empty("live call — measurements unusable")


def empty(note: str = "text channel") -> VoiceSignals:
    """The no-audio case, so callers never have to handle None."""
    return VoiceSignals(note=note)
