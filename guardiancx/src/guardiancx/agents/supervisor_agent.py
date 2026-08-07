"""Supervisor Agent.

The routing brain. Combines the input and output guardrail reports, sets the
final risk level and approval status, and assembles the CaseDecision. High-risk
or guardrail-blocked recommendations are routed to the Human Approval Queue
(status = pending); everything else is cleared for advisory use.
"""
from __future__ import annotations

from ..utils.types import (
    ApprovalStatus,
    CaseDecision,
    GuardrailReport,
    RiskLevel,
    VulnerabilityAssessment,
)
from .state import AgentState


def run(state: AgentState) -> AgentState:
    trace = state.setdefault("trace", [])
    assessment: VulnerabilityAssessment = state.get("assessment") or VulnerabilityAssessment()
    recommendation = state.get("recommendation")
    input_report: GuardrailReport = state.get("input_report") or GuardrailReport()
    output_report: GuardrailReport = state.get("output_report") or GuardrailReport()

    combined = GuardrailReport(results=input_report.results + output_report.results)
    risk = recommendation.risk_level if recommendation else RiskLevel.LOW

    # Route to human approval when: high risk, an output guardrail blocked
    # (e.g. ungrounded citation), or confidence fell below threshold.
    low_confidence = any(
        not r.passed and r.name == "confidence_threshold" for r in output_report.results
    )
    requires_approval = bool(recommendation) and (
        risk == RiskLevel.HIGH or output_report.blocked or low_confidence
    )
    approval = ApprovalStatus.PENDING if requires_approval else (
        ApprovalStatus.NOT_REQUIRED if recommendation else ApprovalStatus.NOT_REQUIRED
    )

    decision = CaseDecision(
        conversation_id=state.get("conversation_id", ""),
        turn_index=state.get("turn_index", 0),
        assessment=assessment,
        recommendation=recommendation,
        guardrails=combined,
        risk_level=risk,
        approval_status=approval,
        masked_text=state.get("masked_text", ""),
    )
    state["decision"] = decision
    trace.append({
        "agent": "supervisor",
        "summary": f"Decision · risk={risk.value} · approval={approval.value}"
                   + (" · guardrail-blocked" if combined.blocked else ""),
    })
    return state
