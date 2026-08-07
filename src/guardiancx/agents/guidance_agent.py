"""Guidance Agent.

Composes an advisory recommendation grounded strictly in the retrieved policy
clauses. Primary path: Claude with a JSON schema (summary + adaptations +
citations + confidence + risk). Fallback: assemble the recommendation directly
from the retrieved clauses. Citations are always the retrieved policy
references, so the hallucination guardrail can verify grounding.
"""
from __future__ import annotations

from ..services.claude_client import get_claude
from ..utils.types import Recommendation, RiskLevel
from .state import AgentState

_SYSTEM = (
    "You are an advisory guidance assistant for bank agents handling potentially "
    "vulnerable customers. Using ONLY the provided policy clauses, write a short "
    "adaptation the human agent could offer. Never invent policy. Cite the clause "
    "references you used. Output is advisory only — the agent decides."
)

_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "adaptations": {"type": "array", "items": {"type": "string"}},
        "citations": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["summary", "adaptations", "citations", "confidence", "risk_level"],
    "additionalProperties": False,
}

# Drivers whose presence makes a turn high-risk (bereavement, serious illness,
# acute financial distress → mandatory human approval).
_HIGH_RISK_DRIVERS = {"health", "life_events", "resilience"}


def _risk_from_assessment(assessment) -> RiskLevel:
    trig = {d.value for d in assessment.triggered}
    if trig & _HIGH_RISK_DRIVERS and assessment.max_score >= 0.8:
        return RiskLevel.HIGH
    if trig:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _fallback(state: AgentState) -> Recommendation:
    retrieved = state.get("retrieved", [])
    assessment = state.get("assessment")
    adaptations = [f"{c.title}: {c.text}" for c in retrieved]
    citations = [c.policy_reference for c in retrieved]
    conf = round(min(0.6 + 0.1 * len(retrieved), 0.9), 2) if retrieved else 0.0
    return Recommendation(
        summary=("Consider the prescribed adaptation(s) for the detected "
                 "vulnerability signal." if retrieved else "No policy adaptation retrieved."),
        adaptations=adaptations,
        citations=citations,
        confidence=conf,
        risk_level=_risk_from_assessment(assessment),
        source="heuristic",
    )


def run(state: AgentState) -> AgentState:
    trace = state.setdefault("trace", [])
    retrieved = state.get("retrieved", [])
    assessment = state.get("assessment")
    if not retrieved:
        state["recommendation"] = None
        trace.append({"agent": "guidance", "summary": "No retrieval — no recommendation."})
        return state

    policy_text = "\n\n".join(f"[{c.policy_reference}] {c.title}: {c.text}" for c in retrieved)
    user = (
        f"Customer said: {state.get('masked_text', '')}\n\n"
        f"Relevant policy clauses:\n{policy_text}\n\n"
        "Write the advisory adaptation grounded only in these clauses."
    )
    llm = get_claude()
    result = llm.structured(_SYSTEM, user, _SCHEMA, max_tokens=800)
    if result:
        # Keep only citations that were actually retrieved (defense in depth).
        valid_refs = {c.policy_reference for c in retrieved}
        citations = [c for c in result.get("citations", []) if c in valid_refs] or list(valid_refs)
        rec = Recommendation(
            summary=str(result.get("summary", "")),
            adaptations=[str(a) for a in result.get("adaptations", [])],
            citations=citations,
            confidence=float(result.get("confidence", 0.0)),
            risk_level=RiskLevel(result.get("risk_level", "medium")),
            source=llm.provider,
        )
        # Risk floor from deterministic driver logic (don't let the LLM under-rate).
        floor = _risk_from_assessment(assessment)
        if floor == RiskLevel.HIGH:
            rec.risk_level = RiskLevel.HIGH
    else:
        rec = _fallback(state)

    state["recommendation"] = rec
    trace.append({
        "agent": "guidance",
        "summary": f"[{rec.source}] guidance drafted · risk={rec.risk_level.value} · "
                   f"conf={rec.confidence:.2f} · cites {', '.join(rec.citations)}.",
    })
    return state
