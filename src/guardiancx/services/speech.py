"""Azure Speech services for GuardianCX — the ears and the voice of the call.

Three capabilities, which together close the loop the live console needs:
speech in (`transcribe_wav`), speech out (`synthesize`), and a short-lived
browser credential (`issue_token`) so continuous recognition can run in the
page without the subscription key ever reaching it.

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
import html
import importlib.util
import io
import sys
import time
import wave
from typing import Optional

import requests

from config.settings import Settings, get_settings

from ..utils import language as lang
from ..utils.logging import get_logger

log = get_logger("services.speech")

# Azure Speech short-audio REST endpoint expects 16-bit mono PCM at 8 or 16 kHz.
_TARGET_RATE = 16000
_REST_TIMEOUT = 30

# Default synthesis voice and recogniser locale, used when nothing is known
# about the call yet. Everything after the first utterance goes through
# `utils.language`, which owns the locale and voice for each of the languages
# this bank's book actually speaks — see `voice_for` and `locale_for` below.
#
# The default is English because an outbound call whose customer record carries
# no language has to open in something, and English is the market's lingua
# franca. It is a starting point, not an assumption: the moment the customer
# speaks, `utils.language.resolve_call_language` decides what the rest of the
# call is held in.
DEFAULT_VOICE = lang.voice_for("en")
DEFAULT_LANGUAGE = lang.stt_locale("en")


def voice_for(language: str) -> str:
    """The neural voice that answers a call held in `language`."""
    return lang.voice_for(language)


def locale_for(language: str) -> str:
    """The recogniser locale for a call held in `language`."""
    return lang.stt_locale(language)

# Speaking-style hints per emotional register. Not every voice supports every
# style, so synthesis retries without the style block if Azure rejects it.
STYLE_FOR_EMOTION = {
    "distressed": "empathetic",
    "sad": "empathetic",
    "anxious": "empathetic",
    "confused": "friendly",
    "frustrated": "calm",
    "angry": "calm",
    "relieved": "friendly",
    "hopeful": "friendly",
    "calm": "",
}

# Tokens issued for the browser are valid for 10 minutes; refresh well inside
# that so a long call never stalls on an expired credential.
TOKEN_TTL_SECONDS = 540


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
        self._token: Optional[str] = None
        self._token_expires: float = 0.0

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
            "tts_available": credentials,  # spoken replies
            "streaming_available": credentials,  # continuous browser recognition
            "sdk_installed": sdk,
            "credentials": credentials,
            "region": self.settings.azure_speech_region or "",
            "voice": DEFAULT_VOICE,
            # What the browser recogniser may be told to listen for. Azure caps
            # at-start language identification at four candidates, so this is a
            # bounded list rather than "everything we support" — see
            # `utils.language.autodetect_locales`.
            "languages": sorted(lang.LANGUAGES),
            "reason": reason,
        }

    @property
    def available(self) -> bool:
        return self.status()["available"]

    # --- browser capture (deployment-safe) -------------------------------
    def transcribe_wav(self, data: bytes, language: str = "en") -> Optional[str]:
        """Transcribe WAV bytes recorded in the browser. Returns None on silence.

        `language` is a language tag ("ur"), not a locale — the mapping to
        "ur-PK" belongs in one place, and this is not it. A full locale is
        accepted too and normalises to the same thing.
        """
        if not self.available:
            raise RuntimeError(self.status()["reason"])

        locale = locale_for(language)
        audio = _normalise_wav(data)
        url = (f"https://{self.settings.azure_speech_region}.stt.speech.microsoft.com"
               f"/speech/recognition/conversation/cognitiveservices/v1?language={locale}")
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

    # --- browser credential (continuous recognition in the page) ---------
    def issue_token(self) -> tuple[str, str]:
        """Mint a short-lived Azure Speech token for the browser.

        The live console runs continuous recognition in the page via the Speech
        JS SDK, which needs a credential. Shipping the subscription key to the
        browser would expose it in the page source to every viewer; an issued
        token expires in ten minutes and is scoped to speech alone. Returns
        (token, region).
        """
        if not self.available:
            raise RuntimeError(self.status()["reason"])

        now = time.time()
        if self._token and now < self._token_expires:
            return self._token, self.settings.azure_speech_region

        url = (f"https://{self.settings.azure_speech_region}"
               ".api.cognitive.microsoft.com/sts/v1.0/issueToken")
        resp = requests.post(
            url,
            headers={"Ocp-Apim-Subscription-Key": self.settings.azure_speech_key,
                     "Content-Length": "0"},
            timeout=_REST_TIMEOUT,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Azure Speech token request failed ({resp.status_code}): {resp.text[:200]}"
            )
        self._token = resp.text
        self._token_expires = now + TOKEN_TTL_SECONDS
        return self._token, self.settings.azure_speech_region

    # --- speech out ------------------------------------------------------
    def synthesize(self, text: str, voice: str = "", emotion: str = "",
                   language: str = "en") -> Optional[bytes]:
        """Render the handler's reply as speech. Returns MP3 bytes.

        `emotion` is the customer's state, not the handler's: a caller who is
        distressed gets a reply delivered in an empathetic register, one who is
        angry gets a deliberately calm one. Where the voice does not support the
        requested style Azure rejects the SSML, so the call retries plain — a
        reply that is spoken flatly is far better than one not spoken at all.

        `language` picks the voice unless one is passed explicitly. Speaking
        approved Urdu wording through an English voice produces something no
        Urdu speaker can follow, so the two are chosen together rather than
        separately: the language of the words decides the mouth that says them.
        Expressive styles are largely an en-* feature, so a non-English voice
        simply skips the style attempt rather than paying for a rejected request.
        """
        if not text or not text.strip():
            return None
        if not self.available:
            raise RuntimeError(self.status()["reason"])

        locale = locale_for(language)
        voice = voice or voice_for(language)
        style = STYLE_FOR_EMOTION.get(emotion, "") if locale.startswith("en") else ""
        for attempt_style in ([style, ""] if style else [""]):
            ssml = self._ssml(text, voice, locale, attempt_style)
            audio = self._post_ssml(ssml)
            if audio is not None:
                return audio
        return None

    @staticmethod
    def _ssml(text: str, voice: str, language: str, style: str) -> str:
        body = html.escape(text.strip())
        # Right-to-left text needs no special handling in SSML — the marks that
        # matter are already in the string, and stripping or re-ordering them
        # here would corrupt approved wording. It is passed through untouched.
        # A slightly slower delivery is easier to follow for anyone in distress
        # or with a capability-related need — which is most of this caseload.
        inner = f'<prosody rate="-6%">{body}</prosody>'
        if style:
            inner = (f'<mstts:express-as style="{style}" styledegree="1.2">'
                     f"{inner}</mstts:express-as>")
        return (
            f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            f'xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{language}">'
            f'<voice name="{voice}">{inner}</voice></speak>'
        )

    def _post_ssml(self, ssml: str) -> Optional[bytes]:
        url = (f"https://{self.settings.azure_speech_region}"
               ".tts.speech.microsoft.com/cognitiveservices/v1")
        resp = requests.post(
            url,
            headers={
                "Ocp-Apim-Subscription-Key": self.settings.azure_speech_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
                "User-Agent": "GuardianCX",
            },
            data=ssml.encode("utf-8"),
            timeout=_REST_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.content
        if resp.status_code == 400:
            log.info("Azure TTS rejected the SSML (likely an unsupported style); retrying plain.")
            return None
        raise RuntimeError(f"Azure TTS returned {resp.status_code}: {resp.text[:200]}")

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
