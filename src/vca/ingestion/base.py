"""Base transcription source interface.

A transcription source yields :class:`Utterance` objects as a conversation
progresses. Concrete sources: simulated (text/chat replay) and Azure Speech.
"""
from __future__ import annotations

import abc
from collections.abc import Iterator

from ..schemas import Utterance


class TranscriptionSource(abc.ABC):
    """Yields finalized utterances for a single conversation."""

    @abc.abstractmethod
    def stream(self) -> Iterator[Utterance]:
        """Yield utterances in order as they become available."""
        raise NotImplementedError
