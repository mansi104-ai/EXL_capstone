"""Classifier interface + shared driver ordering."""
from __future__ import annotations

import abc

from ..schemas import Detection, Driver, DriverScore

# Canonical driver order used by the model output vector.
DRIVER_ORDER: list[Driver] = [
    Driver.HEALTH,
    Driver.LIFE_EVENTS,
    Driver.RESILIENCE,
    Driver.CAPABILITY,
]


class BaseClassifier(abc.ABC):
    """Maps utterance text to per-driver scores. Advisory only."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    @abc.abstractmethod
    def predict_scores(self, text: str) -> list[float]:
        """Return one score in [0, 1] per driver, in DRIVER_ORDER."""
        raise NotImplementedError

    def detect(self, conversation_id: str, turn_index: int, text: str) -> Detection:
        raw = self.predict_scores(text)
        scores = [
            DriverScore(driver=d, score=float(s), triggered=float(s) >= self.threshold)
            for d, s in zip(DRIVER_ORDER, raw)
        ]
        return Detection(
            conversation_id=conversation_id,
            turn_index=turn_index,
            text=text,
            scores=scores,
        )
