"""Azure Speech real-time transcription source.

Uses the Azure Cognitive Services Speech SDK for continuous recognition from the
default microphone (or a supplied audio config). The SDK is imported lazily so
the rest of the system works without it installed.

Every recognised final result is attributed to the ``customer`` speaker by
default; in a real deployment channel/diarization metadata would set the
speaker. This class is only used when AZURE_SPEECH_KEY/REGION are configured and
VCA_FORCE_SIMULATED is not set.
"""
from __future__ import annotations

import queue
from collections.abc import Iterator

from ..schemas import Speaker, Utterance
from .base import TranscriptionSource


class AzureSpeechSource(TranscriptionSource):
    def __init__(
        self,
        conversation_id: str,
        speech_key: str,
        speech_region: str,
        speaker: Speaker = Speaker.CUSTOMER,
        audio_config=None,
    ):
        self.conversation_id = conversation_id
        self.speech_key = speech_key
        self.speech_region = speech_region
        self.speaker = speaker
        self._audio_config = audio_config

    def stream(self) -> Iterator[Utterance]:
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "azure-cognitiveservices-speech is not installed; "
                "install it or use the simulated source."
            ) from exc

        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key, region=self.speech_region
        )
        audio_config = self._audio_config or speechsdk.audio.AudioConfig(
            use_default_microphone=True
        )
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

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
