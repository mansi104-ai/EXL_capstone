"""Evidence Logger Agent.

Persists the CaseDecision to the immutable, hash-chained evidence store and
writes an audit event. This is the last node in the graph; nothing about a
decision is considered final until it has an evidence record.
"""
from __future__ import annotations

from ..database.repository import record_evidence
from ..services.observability import get_observability
from .state import AgentState


def run(state: AgentState) -> AgentState:
    trace = state.setdefault("trace", [])
    decision = state.get("decision")
    if decision is None:
        return state
    record_id = record_evidence(decision, customer_id=state.get("customer_id", ""))
    state["record_id"] = record_id

    obs = get_observability()
    obs.trace(
        "case_decided",
        conversation_id=decision.conversation_id,
        turn_index=decision.turn_index,
        risk=decision.risk_level.value,
        approval=decision.approval_status.value,
        drivers=[d.value for d in decision.assessment.triggered],
    )
    trace.append({"agent": "evidence", "summary": f"Evidence recorded ({record_id[:8]}…)."})
    return state
