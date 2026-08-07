"""Azure Speech transcription service for GuardianCX.

Provides single-phrase microphone capture for the Live Conversation Monitor.
The Azure SDK is imported lazily, and `status()` reports exactly why speech is
or is not available so the UI can guide the user. Microphone capture requires
running the app locally (there is no audio device on a cloud host).
"""
from __future__ import annotations

import importlib.util
from typing import Optional

from config.settings import Settings, get_settings

from ..utils.logging import get_logger

log = get_logger("services.speech")


def _sdk_installed() -> bool:
    return importlib.util.find_spec("azure.cognitiveservices.speech") is not None


class SpeechService:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

    def status(self) -> dict:
        sdk = _sdk_installed()
        key = bool(self.settings.azure_speech_key)
        region = bool(self.settings.azure_speech_region)
        available = sdk and key and region
        if available:
            reason = "Azure Speech ready."
        elif not sdk:
            reason = "azure-cognitiveservices-speech not installed."
        elif not (key and region):
            reason = "AZURE_SPEECH_KEY / AZURE_SPEECH_REGION not set in .env."
        else:  # pragma: no cover
            reason = "Azure Speech unavailable."
        return {
            "available": available,
            "sdk_installed": sdk,
            "credentials": key and region,
            "region": self.settings.azure_speech_region or "",
            "reason": reason,
        }

    @property
    def available(self) -> bool:
        return self.status()["available"]

    def recognize_once(self, language: str = "en-GB", timeout_seconds: int = 15) -> Optional[str]:
        """Capture one spoken phrase from the default microphone and return text."""
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
