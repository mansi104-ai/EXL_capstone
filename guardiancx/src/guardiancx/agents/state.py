"""Shared state passed between GuardianCX agents in the LangGraph pipeline."""
from __future__ import annotations

from typing import Any, Optional, TypedDict

from ..utils.types import (
    GuardrailReport,
    PolicyChunk,
    Recommendation,
    VulnerabilityAssessment,
)


class AgentState(TypedDict, total=False):
    # inputs
    conversation_id: str
    customer_id: str
    turn_index: int
    speaker: str
    text: str

    # produced by agents
    masked_text: str
    input_report: GuardrailReport
    assessment: VulnerabilityAssessment
    retrieved: list[PolicyChunk]
    recommendation: Optional[Recommendation]
    output_report: GuardrailReport
    trace: list[dict[str, Any]]
    decision: Any          # CaseDecision
    record_id: str
