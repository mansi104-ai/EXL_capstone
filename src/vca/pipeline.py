"""End-to-end VCA pipeline.

Wires the stages together:

    ingestion -> classifier -> RAG guidance -> (handler decision) -> evidence

Two granularities are supported:

* ``assess(utterance)`` — advisory only: classify + retrieve guidance, WITHOUT
  writing evidence. This is the perceive→assess→advise part of the agent loop
  and is used for interactive / live turns where the handler decides next.
* ``record_outcome(...)`` — commit the detection, guidance and the handler's
  outcome to the immutable evidence log (the "record" step).
* ``process(source, on_guidance)`` — batch convenience that assesses every
  utterance and records with an outcome callback.

Only customer utterances are assessed for vulnerability signals.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .classifier.base import BaseClassifier
from .classifier.factory import load_classifier
from .config import Config, load_config
from .evidence.store import EvidenceStore
from .guidance.advisor import HandlerAdvisor
from .ingestion.base import TranscriptionSource
from .ingestion.factory import make_source
from .rag.retriever import BaseRetriever, load_retriever
from .schemas import Detection, EvidenceRecord, Guidance, Outcome, Speaker, Utterance

# Called when guidance is produced; returns the handler's outcome.
OutcomeCallback = Callable[[Guidance], Outcome]


@dataclass
class PipelineEvent:
    """One step of the conversation as seen by the pipeline."""

    utterance: Utterance
    detection: Detection | None = None
    guidance: Guidance | None = None
    record: EvidenceRecord | None = None

    @property
    def assessed(self) -> bool:
        """True if this (customer) utterance was run through the classifier."""
        return self.detection is not None

    @property
    def flagged(self) -> bool:
        return self.guidance is not None


class VCAPipeline:
    def __init__(
        self,
        classifier: BaseClassifier,
        advisor: HandlerAdvisor,
        evidence: EvidenceStore,
    ):
        self.classifier = classifier
        self.advisor = advisor
        self.evidence = evidence

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> "VCAPipeline":
        cfg = cfg or load_config()
        classifier = load_classifier(cfg)
        retriever: BaseRetriever = load_retriever(
            policy_path=cfg.path(cfg.rag["policy_path"]),
            embed_model=cfg.rag["embed_model"],
            top_k=int(cfg.rag.get("top_k", 3)),
        )
        advisor = HandlerAdvisor(retriever)
        evidence = EvidenceStore(cfg.path(cfg.evidence["path"]))
        return cls(classifier, advisor, evidence)

    # --- single-turn (interactive / live) --------------------------------
    def assess(self, utterance: Utterance) -> PipelineEvent:
        """Advisory assessment of one utterance. Writes no evidence.

        Handler utterances pass through unassessed. Customer utterances are
        classified; if any driver triggers, guidance is attached.
        """
        if utterance.speaker != Speaker.CUSTOMER:
            return PipelineEvent(utterance=utterance)

        detection = self.classifier.detect(
            utterance.conversation_id, utterance.turn_index, utterance.text
        )
        guidance = self.advisor.advise(detection) if detection.any_triggered else None
        return PipelineEvent(utterance=utterance, detection=detection, guidance=guidance)

    def record_outcome(
        self, detection: Detection, guidance: Guidance, outcome: Outcome
    ) -> EvidenceRecord:
        """Commit a detection + guidance + handler outcome to the evidence log."""
        return self.evidence.append(detection, guidance, outcome)

    # --- batch -----------------------------------------------------------
    def process(
        self,
        source: TranscriptionSource,
        on_guidance: OutcomeCallback | None = None,
    ) -> Iterator[PipelineEvent]:
        """Run the pipeline over a source, yielding an event per utterance.

        When an utterance is flagged, an evidence record is written using the
        outcome from ``on_guidance`` (or a default NO_RESPONSE outcome).
        """
        for utt in source.stream():
            event = self.assess(utt)
            if event.guidance is not None:
                outcome = on_guidance(event.guidance) if on_guidance else Outcome()
                event.record = self.record_outcome(event.detection, event.guidance, outcome)
            yield event

    def process_transcript(
        self,
        transcript_path: str | Path,
        conversation_id: str | None = None,
        on_guidance: OutcomeCallback | None = None,
        cfg: Config | None = None,
    ) -> Iterator[PipelineEvent]:
        source = make_source(
            conversation_id=conversation_id or "CONV",
            transcript_path=transcript_path,
            cfg=cfg,
        )
        yield from self.process(source, on_guidance=on_guidance)
