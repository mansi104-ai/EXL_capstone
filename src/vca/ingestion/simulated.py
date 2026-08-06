"""Simulated transcription source.

Replays a pre-recorded conversation (JSON transcript or chat log) as a stream of
utterances. This is the fallback used when Azure Speech is not configured, and
the default source for the demo / tests.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from pathlib import Path

from ..schemas import Speaker, Utterance
from .base import TranscriptionSource


class SimulatedSource(TranscriptionSource):
    def __init__(
        self,
        conversation_id: str,
        turns: list[dict],
        delay_seconds: float = 0.0,
    ):
        self.conversation_id = conversation_id
        self.turns = turns
        self.delay_seconds = delay_seconds

    @classmethod
    def from_file(cls, path: str | Path, delay_seconds: float = 0.0) -> "SimulatedSource":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            conversation_id=data["conversation_id"],
            turns=data["turns"],
            delay_seconds=delay_seconds,
        )

    def stream(self) -> Iterator[Utterance]:
        for i, turn in enumerate(self.turns):
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            yield Utterance(
                conversation_id=self.conversation_id,
                turn_index=i,
                speaker=Speaker(turn["speaker"]),
                text=turn["text"],
                is_final=True,
            )
