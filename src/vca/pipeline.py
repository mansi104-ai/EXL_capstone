"""End-to-end VCA pipeline.

Wires the stages together:

    ingestion -> classifier -> RAG guidance -> (handler) -> evidence store

Only customer utterances are classified. When a driver triggers, guidance is
produced and an evidence record is written. The handler's outcome can be
supplied via a callback (UI / CLI) or defaults to NO_RESPONSE.
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
from .schemas import EvidenceRecord, Guidance, Outcome, Speaker, Utterance

# Called when guidance is produced; returns the handler's outcome.
OutcomeCallback = Callable[[Guidance], Outcome]


@dataclass
class PipelineEvent:
    """One step of the conversation as seen by the pipeline."""

    utterance: Utterance
    guidance: Guidance | None = None
    record: EvidenceRecord | None = None


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

    def process(
        self,
        source: TranscriptionSource,
        on_guidance: OutcomeCallback | None = None,
    ) -> Iterator[PipelineEvent]:
        """Run the pipeline over a source, yielding an event per utterance."""
        for utt in source.stream():
            # Only the customer's words are assessed for vulnerability signals.
            if utt.speaker != Speaker.CUSTOMER:
                yield PipelineEvent(utterance=utt)
                continue

            detection = self.classifier.detect(
                utt.conversation_id, utt.turn_index, utt.text
            )
            if not detection.any_triggered:
                yield PipelineEvent(utterance=utt)
                continue

            guidance = self.advisor.advise(detection)
            assert guidance is not None  # any_triggered guarantees guidance
            outcome = on_guidance(guidance) if on_guidance else Outcome()
            record = self.evidence.append(detection, guidance, outcome)
            yield PipelineEvent(utterance=utt, guidance=guidance, record=record)

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
