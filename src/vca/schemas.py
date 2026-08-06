"""Shared data models for the VCA pipeline.

These flow between ingestion -> classifier -> RAG -> guidance -> evidence.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Driver(str, Enum):
    """The four regulator-defined vulnerability drivers (FCA-aligned)."""

    HEALTH = "health"
    LIFE_EVENTS = "life_events"
    RESILIENCE = "resilience"
    CAPABILITY = "capability"


class Speaker(str, Enum):
    CUSTOMER = "customer"
    HANDLER = "handler"


class Utterance(BaseModel):
    """A single transcribed turn in a conversation."""

    conversation_id: str
    turn_index: int
    speaker: Speaker
    text: str
    timestamp: datetime = Field(default_factory=_now)
    is_final: bool = True  # False for interim streaming hypotheses


class DriverScore(BaseModel):
    driver: Driver
    score: float
    triggered: bool


class Detection(BaseModel):
    """Classifier output for one utterance. Advisory only."""

    conversation_id: str
    turn_index: int
    text: str
    scores: list[DriverScore]
    timestamp: datetime = Field(default_factory=_now)

    @property
    def triggered_drivers(self) -> list[Driver]:
        return [s.driver for s in self.scores if s.triggered]

    @property
    def any_triggered(self) -> bool:
        return any(s.triggered for s in self.scores)


class Adaptation(BaseModel):
    """A prescribed adaptation retrieved from the firm's policy via RAG."""

    driver: Driver
    title: str
    guidance: str
    policy_reference: str
    retrieval_score: float


class Guidance(BaseModel):
    """The discreet, advisory-only prompt surfaced to the handler."""

    conversation_id: str
    turn_index: int
    detection: Detection
    adaptations: list[Adaptation]
    message: str
    timestamp: datetime = Field(default_factory=_now)
    advisory_only: bool = True  # never an automated action


class HandlerAction(str, Enum):
    ACCEPTED = "accepted"
    DISMISSED = "dismissed"
    MODIFIED = "modified"
    NO_RESPONSE = "no_response"


class Outcome(BaseModel):
    """What the handler did with the guidance, recorded for the evidence trail."""

    action: HandlerAction = HandlerAction.NO_RESPONSE
    note: Optional[str] = None
    recorded_at: datetime = Field(default_factory=_now)


class EvidenceRecord(BaseModel):
    """One immutable, hash-chained entry in the evidence log."""

    record_id: str
    conversation_id: str
    turn_index: int
    detection: Detection
    guidance: Guidance
    outcome: Outcome
    created_at: datetime = Field(default_factory=_now)
    prev_hash: str
    record_hash: str
