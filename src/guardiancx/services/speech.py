"""Azure Speech transcription service for GuardianCX.

Two capture paths, because they have different deployment constraints:

* **Browser capture (works everywhere, incl. Streamlit Cloud).** The UI records
  audio in the user's browser via ``st.audio_input`` and hands the WAV bytes to
  :meth:`SpeechService.transcribe_wav`, which posts them to Azure Speech's REST
  endpoint. This is plain HTTPS — no native SDK, no audio device on the host.
* **Server microphone (local only).** :meth:`SpeechService.recognize_once` uses
  the Azure Speech SDK against the *host's* default microphone. On a cloud host
  there is no audio device, so this path is unavailable there by definition.

`status()` reports exactly which paths are live so the UI can guide the user.
"""
from __future__ import annotations

import array
import importlib.util
import io
import sys
import wave
from typing import Optional

import requests

from config.settings import Settings, get_settings

from ..utils.logging import get_logger

log = get_logger("services.speech")

# Azure Speech short-audio REST endpoint expects 16-bit mono PCM at 8 or 16 kHz.
_TARGET_RATE = 16000
_REST_TIMEOUT = 30


def _sdk_installed() -> bool:
    return importlib.util.find_spec("azure.cognitiveservices.speech") is not None


# --------------------------------------------------------------------------- #
# WAV normalisation
#
# Browsers record at whatever rate the device offers (commonly 44.1/48 kHz, and
# sometimes stereo). Azure's REST endpoint accepts only 8/16 kHz mono, so we
# down-mix and resample here using the standard library alone — `audioop` was
# removed in Python 3.13, and pulling in numpy/pydub for a few seconds of speech
# is not worth the dependency.
# --------------------------------------------------------------------------- #
def _resample(samples: array.array, src_rate: int, dst_rate: int) -> array.array:
    """Box-average resampler. Averaging (rather than picking nearest) acts as a
    crude low-pass, which keeps decimation from aliasing speech into mush."""
    if src_rate == dst_rate:
        return samples
    n_out = int(len(samples) * dst_rate / src_rate)
    out = array.array("h", bytes(2 * n_out))
    ratio = src_rate / dst_rate
    for i in range(n_out):
        start = int(i * ratio)
        end = min(max(start + 1, int((i + 1) * ratio)), len(samples))
        window = samples[start:end]
        out[i] = int(sum(window) / len(window)) if window else 0
    return out


def _normalise_wav(data: bytes) -> bytes:
    """Convert arbitrary 16-bit WAV bytes to 16 kHz mono PCM WAV."""
    with wave.open(io.BytesIO(data), "rb") as src:
        channels, width, rate = src.getnchannels(), src.getsampwidth(), src.getframerate()
        frames = src.readframes(src.getnframes())

    if width != 2:
        raise RuntimeError(f"Unsupported audio: {width * 8}-bit samples (expected 16-bit).")
    if not frames:
        raise RuntimeError("The recording was empty.")

    samples = array.array("h")
    samples.frombytes(frames)
    if sys.byteorder == "big":  # WAV is little-endian on the wire
        samples.byteswap()

    if channels > 1:
        samples = array.array("h", (
            sum(samples[i:i + channels]) // channels
            for i in range(0, len(samples) - channels + 1, channels)
        ))
    samples = _resample(samples, rate, _TARGET_RATE)

    payload = samples
    if sys.byteorder == "big":
        payload = array.array("h", samples)
        payload.byteswap()

    out = io.BytesIO()
    with wave.open(out, "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(_TARGET_RATE)
        dst.writeframes(payload.tobytes())
    return out.getvalue()


class SpeechService:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    # --- capability ------------------------------------------------------
    def status(self) -> dict:
        key = bool(self.settings.azure_speech_key)
        region = bool(self.settings.azure_speech_region)
        credentials = key and region
        sdk = _sdk_installed()

        if credentials:
            reason = "Azure Speech ready — record with the microphone below."
        else:
            reason = ("AZURE_SPEECH_KEY / AZURE_SPEECH_REGION are not configured. "
                      "Set them in .env locally, or in the app's Streamlit secrets "
                      "when deployed.")
        return {
            "available": credentials,      # browser capture + REST transcription
            "mic_available": credentials and sdk,  # host microphone (local only)
            "sdk_installed": sdk,
            "credentials": credentials,
            "region": self.settings.azure_speech_region or "",
            "reason": reason,
        }

    @property
    def available(self) -> bool:
        return self.status()["available"]

    # --- browser capture (deployment-safe) -------------------------------
    def transcribe_wav(self, data: bytes, language: str = "en-GB") -> Optional[str]:
        """Transcribe WAV bytes recorded in the browser. Returns None on silence."""
        if not self.available:
            raise RuntimeError(self.status()["reason"])

        audio = _normalise_wav(data)
        url = (f"https://{self.settings.azure_speech_region}.stt.speech.microsoft.com"
               f"/speech/recognition/conversation/cognitiveservices/v1?language={language}")
        resp = requests.post(
            url,
            headers={
                "Ocp-Apim-Subscription-Key": self.settings.azure_speech_key,
                "Content-Type": f"audio/wav; codecs=audio/pcm; samplerate={_TARGET_RATE}",
                "Accept": "application/json",
            },
            data=audio,
            timeout=_REST_TIMEOUT,
        )
        if resp.status_code == 401:
            raise RuntimeError("Azure Speech rejected the key (401) — check the key and region match.")
        if resp.status_code != 200:
            raise RuntimeError(f"Azure Speech returned {resp.status_code}: {resp.text[:200]}")

        payload = resp.json()
        status = payload.get("RecognitionStatus")
        if status == "Success":
            return payload.get("DisplayText") or None
        if status in ("NoMatch", "InitialSilenceTimeout"):
            return None  # silence / unintelligible
        raise RuntimeError(f"Azure Speech could not transcribe the audio ({status}).")

    # --- host microphone (local runs only) -------------------------------
    def recognize_once(self, language: str = "en-GB", timeout_seconds: int = 15) -> Optional[str]:
        """Capture one spoken phrase from the *host's* default microphone.

        Only meaningful when the app runs on the same machine as the speaker —
        a cloud host has no audio device. Deployed apps use `transcribe_wav`.
        """
        import azure.cognitiveservices.speech as speechsdk  # lazy import

        cfg = speechsdk.SpeechConfig(
            subscription=self.settings.azure_speech_key,
            region=self.settings.azure_speech_region,
        )
        cfg.speech_recognition_language = language
        audio = speechsdk.audio.AudioConfig(use_default_microphone=True)
        recognizer = speechsdk.SpeechRecognizer(speech_config=cfg, audio_config=audio)
        result = recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text or None
        if result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            raise RuntimeError(f"Azure Speech canceled: {details.reason} — {details.error_details}")
        return None  # NoMatch / silence


_SPEECH: SpeechService | None = None


def get_speech() -> SpeechService:
    global _SPEECH
    if _SPEECH is None:
        _SPEECH = SpeechService()
    return _SPEECH
