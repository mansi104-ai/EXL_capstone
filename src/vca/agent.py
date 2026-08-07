"""Agentic orchestration + transparent decision trace.

The VCA is an *advisory* agent: for each customer turn it runs a small, fully
inspectable agent loop and emits a step-by-step trace so a handler, an auditor,
or a supervisor can see exactly why guidance did (or did not) appear. This is the
transparency spine of the MVP.

Agent loop per customer utterance:

    1. PERCEIVE   ingest the utterance (typed / uploaded / Azure Speech)
    2. ASSESS     classify vulnerability drivers  -> confidence per driver
    3. RETRIEVE   RAG tool call over the firm policy -> candidate adaptations
    4. ADVISE     compose a discreet, advisory-only prompt for the handler
    5. DECIDE     human-in-the-loop: the handler accepts / modifies / dismisses
    6. RECORD     write detection + guidance + outcome to immutable evidence

Steps 1–4 are produced here (``TurnTrace``); steps 5–6 are driven by the UI /
caller via :meth:`VCAPipeline.record_outcome`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .pipeline import PipelineEvent, VCAPipeline
from .schemas import Driver, Speaker, Utterance


class Step(str, Enum):
    PERCEIVE = "perceive"
    ASSESS = "assess"
    RETRIEVE = "retrieve"
    ADVISE = "advise"
    DECIDE = "decide"
    RECORD = "record"


@dataclass
class TraceStep:
    step: Step
    summary: str
    detail: dict = field(default_factory=dict)


@dataclass
class TurnTrace:
    """The agent's reasoning for one utterance, ready to render."""

    utterance: Utterance
    event: PipelineEvent
    steps: list[TraceStep] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return self.event.flagged


class CareAgent:
    """Thin orchestrator that wraps the pipeline and records a trace."""

    def __init__(self, pipeline: VCAPipeline):
        self.pipeline = pipeline

    def run_turn(self, utterance: Utterance) -> TurnTrace:
        event = self.pipeline.assess(utterance)
        steps: list[TraceStep] = []

        # 1. PERCEIVE
        steps.append(
            TraceStep(
                Step.PERCEIVE,
                f"Received {utterance.speaker.value} turn ({len(utterance.text.split())} words).",
                {"speaker": utterance.speaker.value, "text": utterance.text},
            )
        )

        if utterance.speaker != Speaker.CUSTOMER:
            steps.append(
                TraceStep(
                    Step.ASSESS,
                    "Handler turn — not assessed for vulnerability signals.",
                )
            )
            return TurnTrace(utterance=utterance, event=event, steps=steps)

        # 2. ASSESS
        det = event.detection
        scores = {s.driver.value: round(s.score, 3) for s in det.scores}
        triggered = [d.value for d in det.triggered_drivers]
        steps.append(
            TraceStep(
                Step.ASSESS,
                (
                    f"Classifier flagged: {', '.join(triggered)}."
                    if triggered
                    else "No vulnerability driver above threshold."
                ),
                {"scores": scores, "triggered": triggered},
            )
        )

        if not event.flagged:
            return TurnTrace(utterance=utterance, event=event, steps=steps)

        # 3. RETRIEVE
        guidance = event.guidance
        steps.append(
            TraceStep(
                Step.RETRIEVE,
                f"RAG returned {len(guidance.adaptations)} policy adaptation(s).",
                {
                    "adaptations": [
                        {
                            "driver": a.driver.value,
                            "policy_reference": a.policy_reference,
                            "title": a.title,
                            "retrieval_score": a.retrieval_score,
                        }
                        for a in guidance.adaptations
                    ]
                },
            )
        )

        # 4. ADVISE
        steps.append(
            TraceStep(
                Step.ADVISE,
                "Composed advisory-only guidance for the handler.",
                {"message": guidance.message, "advisory_only": guidance.advisory_only},
            )
        )

        # 5. DECIDE (pending — filled in by the UI when the handler responds)
        steps.append(
            TraceStep(
                Step.DECIDE,
                "Awaiting handler decision (accept / modify / dismiss).",
            )
        )
        return TurnTrace(utterance=utterance, event=event, steps=steps)


# Documented, UI-facing mapping of the four drivers for labels/tooltips.
DRIVER_HELP = {
    Driver.HEALTH: "Serious/terminal illness, mental health, cognitive difficulty.",
    Driver.LIFE_EVENTS: "Bereavement, job loss, divorce, caring responsibilities.",
    Driver.RESILIENCE: "Financial distress, over-indebtedness, no savings buffer.",
    Driver.CAPABILITY: "Low financial/digital literacy, language barrier, third-party help.",
}
