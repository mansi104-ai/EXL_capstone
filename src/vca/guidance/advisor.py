"""Builds the discreet, advisory-only guidance prompt for the handler.

Takes a Detection (which drivers fired) and the retrieved Adaptations, and
composes a short prompt the handler sees. Nothing here takes action — it only
advises. The guidance is always marked advisory_only=True.
"""
from __future__ import annotations

from ..rag.retriever import BaseRetriever
from ..schemas import Adaptation, Detection, Driver, Guidance

_DRIVER_LABEL = {
    Driver.HEALTH: "Health",
    Driver.LIFE_EVENTS: "Life event",
    Driver.RESILIENCE: "Resilience / financial",
    Driver.CAPABILITY: "Capability",
}


class HandlerAdvisor:
    def __init__(self, retriever: BaseRetriever, per_driver_k: int = 1):
        self.retriever = retriever
        self.per_driver_k = per_driver_k

    def _adaptations_for(self, detection: Detection) -> list[Adaptation]:
        adaptations: list[Adaptation] = []
        seen: set[str] = set()
        for driver in detection.triggered_drivers:
            for adap in self.retriever.retrieve(
                detection.text, driver=driver, top_k=self.per_driver_k
            ):
                if adap.policy_reference not in seen:
                    seen.add(adap.policy_reference)
                    adaptations.append(adap)
        return adaptations

    @staticmethod
    def _compose_message(detection: Detection, adaptations: list[Adaptation]) -> str:
        drivers = ", ".join(_DRIVER_LABEL[d] for d in detection.triggered_drivers)
        lines = [
            f"Possible vulnerability signal — {drivers}.",
            "Suggested adaptation (advisory only — you decide):",
        ]
        for a in adaptations:
            lines.append(f"  • {a.title} [{a.policy_reference}]: {a.guidance}")
        lines.append("This is guidance, not an instruction. Record what you do.")
        return "\n".join(lines)

    def advise(self, detection: Detection) -> Guidance | None:
        """Return guidance if any driver triggered, else None (stay silent)."""
        if not detection.any_triggered:
            return None
        adaptations = self._adaptations_for(detection)
        message = self._compose_message(detection, adaptations)
        return Guidance(
            conversation_id=detection.conversation_id,
            turn_index=detection.turn_index,
            detection=detection,
            adaptations=adaptations,
            message=message,
            advisory_only=True,
        )
