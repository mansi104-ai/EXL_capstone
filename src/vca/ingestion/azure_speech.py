"""Azure Speech real-time transcription.

Three ways to get customer speech from Azure Cognitive Services Speech:

* ``AzureSpeechSource`` — a :class:`TranscriptionSource` that yields utterances
  from continuous recognition (used by the batch pipeline / factory).
* ``AzureSpeechSource.recognize_once`` — capture a single spoken phrase. Ideal
  for a click-to-speak button in the UI.
* ``LiveAzureRecognizer`` — start/stop continuous recognition in the background
  and ``drain()`` finalized utterances. Suited to a polling UI (Streamlit),
  where holding a generator open across reruns is awkward.

The SDK is imported lazily so the rest of the system runs without it installed.
Speaker attribution defaults to ``customer`` (real deployments would use channel
/ diarization metadata).
"""
from __future__ import annotations

import queue
import threading
from collections.abc import Iterator

from ..schemas import Speaker, Utterance
from .base import TranscriptionSource


def _import_sdk():
    try:
        import azure.cognitiveservices.speech as speechsdk  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "azure-cognitiveservices-speech is not installed; "
            "install requirements-full.txt or use another input mode."
        ) from exc
    return speechsdk


class AzureSpeechSource(TranscriptionSource):
    def __init__(
        self,
        conversation_id: str,
        speech_key: str,
        speech_region: str,
        speaker: Speaker = Speaker.CUSTOMER,
        language: str = "en-GB",
        audio_config=None,
    ):
        self.conversation_id = conversation_id
        self.speech_key = speech_key
        self.speech_region = speech_region
        self.speaker = speaker
        self.language = language
        self._audio_config = audio_config

    def _make_recognizer(self, speechsdk):
        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key, region=self.speech_region
        )
        speech_config.speech_recognition_language = self.language
        audio_config = self._audio_config or speechsdk.audio.AudioConfig(
            use_default_microphone=True
        )
        return speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

    def recognize_once(self) -> str | None:
        """Capture one spoken phrase and return its text (or None)."""
        speechsdk = _import_sdk()
        recognizer = self._make_recognizer(speechsdk)
        result = recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text or None
        if result.reason == speechsdk.ResultReason.Canceled:
            details = result.cancellation_details
            raise RuntimeError(f"Azure Speech canceled: {details.reason} — {details.error_details}")
        return None  # NoMatch / silence

    def stream(self) -> Iterator[Utterance]:
        speechsdk = _import_sdk()
        recognizer = self._make_recognizer(speechsdk)
        results: "queue.Queue[str | None]" = queue.Queue()

        def _on_recognized(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
                results.put(evt.result.text)

        def _on_stopped(evt):
            results.put(None)  # sentinel: end of stream

        recognizer.recognized.connect(_on_recognized)
        recognizer.session_stopped.connect(_on_stopped)
        recognizer.canceled.connect(_on_stopped)

        recognizer.start_continuous_recognition()
        turn_index = 0
        try:
            while True:
                text = results.get()
                if text is None:
                    break
                yield Utterance(
                    conversation_id=self.conversation_id,
                    turn_index=turn_index,
                    speaker=self.speaker,
                    text=text,
                    is_final=True,
                )
                turn_index += 1
        finally:
            recognizer.stop_continuous_recognition()


class LiveAzureRecognizer:
    """Background continuous recognizer for a polling UI.

    Call ``start()`` once, then ``drain()`` repeatedly to pull newly finalized
    text since the last poll; ``stop()`` to end. Thread-safe.
    """

    def __init__(
        self,
        speech_key: str,
        speech_region: str,
        language: str = "en-GB",
    ):
        self.speech_key = speech_key
        self.speech_region = speech_region
        self.language = language
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._recognizer = None
        self._lock = threading.Lock()
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            speechsdk = _import_sdk()
            speech_config = speechsdk.SpeechConfig(
                subscription=self.speech_key, region=self.speech_region
            )
            speech_config.speech_recognition_language = self.language
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config, audio_config=audio_config
            )

            def _on_recognized(evt):
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
                    self._queue.put(evt.result.text)

            recognizer.recognized.connect(_on_recognized)
            recognizer.start_continuous_recognition()
            self._recognizer = recognizer
            self._running = True

    def drain(self) -> list[str]:
        """Return all finalized phrases recognized since the last call."""
        out = []
        while True:
            try:
                out.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return out

    def stop(self) -> None:
        with self._lock:
            if self._recognizer is not None:
                try:
                    self._recognizer.stop_continuous_recognition()
                finally:
                    self._recognizer = None
            self._running = False
