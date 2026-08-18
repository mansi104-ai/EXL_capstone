"""Compliance Agent.

Runs the *output* guardrails over the drafted recommendation: confidence
threshold, hallucination-vs-policy grounding, the finance-specific
prohibited-action check for this journey, and the mandatory human-approval
gate for high-risk guidance. Produces the output GuardrailReport the Supervisor
uses to route the case.
"""
from __future__ import annotations

from ..finance.taxonomy import FinancialContext
from ..guardrails.manager import get_guardrail_manager
from .state import AgentState


def run(state: AgentState) -> AgentState:
    trace = state.setdefault("trace", [])
    context: FinancialContext = state.get("financial_context") or FinancialContext()
    report = get_guardrail_manager().run_output(
        text=state.get("masked_text", ""),
        recommendation=state.get("recommendation"),
        retrieved=state.get("retrieved", []),
        journey=context.journey,
    )
    state["output_report"] = report
    flags = [r.name for r in report.results if not r.passed]
    trace.append({
        "agent": "compliance",
        "summary": f"Output guardrails ran · {'blocked' if report.blocked else 'clear'}"
                   + (f" · flags: {', '.join(flags)}" if flags else ""),
    })
    return state
