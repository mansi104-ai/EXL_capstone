"""Shared data models that flow through the GuardianCX agent graph."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


class Driver(str, Enum):
    """FCA-aligned vulnerability drivers."""

    HEALTH = "health"
    LIFE_EVENTS = "life_events"
    RESILIENCE = "resilience"
    CAPABILITY = "capability"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ApprovalStatus(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class Utterance(BaseModel):
    conversation_id: str
    turn_index: int
    speaker: str  # "customer" | "agent"
    text: str
    timestamp: datetime = Field(default_factory=now_utc)


class DriverSignal(BaseModel):
    driver: Driver
    score: float
    evidence: str = ""


class VulnerabilityAssessment(BaseModel):
    signals: list[DriverSignal] = Field(default_factory=list)
    rationale: str = ""
    source: str = "heuristic"  # "claude" | "heuristic"

    @property
    def triggered(self) -> list[Driver]:
        return [s.driver for s in self.signals if s.score >= 0.5]

    @property
    def max_score(self) -> float:
        return max((s.score for s in self.signals), default=0.0)


class PolicyChunk(BaseModel):
    policy_reference: str
    title: str
    driver: Optional[Driver] = None
    text: str
    score: float = 0.0


class Recommendation(BaseModel):
    summary: str
    adaptations: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)  # policy references
    confidence: float = 0.0
    risk_level: RiskLevel = RiskLevel.LOW
    source: str = "heuristic"


class GuardrailResult(BaseModel):
    name: str
    passed: bool
    detail: str = ""
    severity: str = "info"  # info | warn | block
    data: dict[str, Any] = Field(default_factory=dict)


class GuardrailReport(BaseModel):
    results: list[GuardrailResult] = Field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return any(r.severity == "block" and not r.passed for r in self.results)

    @property
    def warnings(self) -> list[GuardrailResult]:
        return [r for r in self.results if not r.passed and r.severity != "block"]


class CaseDecision(BaseModel):
    """The Supervisor agent's final routing decision for one turn."""

    conversation_id: str
    turn_index: int
    assessment: VulnerabilityAssessment
    recommendation: Optional[Recommendation] = None
    guardrails: GuardrailReport = Field(default_factory=GuardrailReport)
    risk_level: RiskLevel = RiskLevel.LOW
    approval_status: ApprovalStatus = ApprovalStatus.NOT_REQUIRED
    masked_text: str = ""
    created_at: datetime = Field(default_factory=now_utc)
