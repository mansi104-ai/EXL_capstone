"""Supervisor Agent.

The routing brain. It combines everything the specialists produced — the
vulnerability assessment, the financial context, the customer's measured state,
and both guardrail reports — into one CaseDecision, and decides whether a person
has to look at it before it is used.

Routing to the approval queue is deliberately over-inclusive. The cost of a
handler glancing at a recommendation that turned out to be routine is a few
seconds; the cost of acting on unreviewed advice in a bereavement or scam call is
a customer harmed and a breach to report. Five independent conditions each route
on their own:

* the recommendation is high risk;
* an output guardrail blocked it (an ungrounded citation, a prohibited action);
* confidence fell below the configured threshold;
* the sentiment agent asked for escalation;
* the financial context shows detriment already happening.

Everything else is cleared for advisory use — the handler still decides.
"""
from __future__ import annotations

from ..finance.taxonomy import FinancialContext
from ..utils.types import (
    ApprovalStatus,
    CaseDecision,
    GuardrailReport,
    RiskLevel,
    SentimentReading,
    VulnerabilityAssessment,
)
from .state import AgentState


def run(state: AgentState) -> AgentState:
    trace = state.setdefault("trace", [])
    assessment: VulnerabilityAssessment = state.get("assessment") or VulnerabilityAssessment()
    recommendation = state.get("recommendation")
    input_report: GuardrailReport = state.get("input_report") or GuardrailReport()
    output_report: GuardrailReport = state.get("output_report") or GuardrailReport()
    context: FinancialContext = state.get("financial_context") or FinancialContext()
    sentiment: SentimentReading = state.get("sentiment") or SentimentReading()

    combined = GuardrailReport(results=input_report.results + output_report.results)
    risk = recommendation.risk_level if recommendation else RiskLevel.LOW

    low_confidence = any(
        not r.passed and r.name == "confidence_threshold" for r in output_report.results
    )
    reasons: list[str] = []
    if recommendation:
        if risk == RiskLevel.HIGH:
            reasons.append("high risk")
        if output_report.blocked:
            reasons.append("guardrail block")
        if low_confidence:
            reasons.append("low confidence")
        if sentiment.escalate:
            reasons.append("customer distress")
        if context.acute:
            reasons.append("acute financial detriment")

    approval = ApprovalStatus.PENDING if reasons else ApprovalStatus.NOT_REQUIRED

    decision = CaseDecision(
        conversation_id=state.get("conversation_id", ""),
        turn_index=state.get("turn_index", 0),
        assessment=assessment,
        recommendation=recommendation,
        guardrails=combined,
        risk_level=risk,
        approval_status=approval,
        masked_text=state.get("masked_text", ""),
        channel=state.get("channel", "chat"),
        sentiment=sentiment if sentiment.source != "skipped" else None,
        financial_context=context.model_dump(mode="json") if context.source != "skipped" else None,
    )
    state["decision"] = decision
    trace.append({
        "agent": "supervisor",
        "summary": f"Decision · risk={risk.value} · approval={approval.value}"
                   + (f" · routed for: {', '.join(reasons)}" if reasons else ""),
    })
    return state
