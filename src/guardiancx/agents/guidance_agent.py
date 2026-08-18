"""Guidance Agent.

Composes an advisory recommendation grounded strictly in the retrieved policy
clauses. Primary path: the LLM with a JSON schema (summary + adaptations +
citations + confidence + risk). Fallback: assemble the recommendation directly
from the retrieved clauses. Citations are always drawn from the retrieved policy
references, so the hallucination guardrail can verify grounding.

Three things constrain the output beyond the clauses themselves:

* **The journey's obligations** are named in the prompt, so the advice is framed
  against the rule that actually applies rather than against general good manners.
* **The journey's prohibited actions** are named too — offering more credit to a
  customer disclosing gambling harm, pressing for a payment that would leave the
  electricity unpaid. The compliance agent re-checks these deterministically
  afterwards, because a prompt instruction is a request and a guardrail is a
  guarantee.
* **A risk floor** is applied from deterministic logic. The model may raise the
  risk level; it may never lower it below what the drivers, the journey and the
  measured distress imply.
"""
from __future__ import annotations

from ..finance.taxonomy import FinancialContext, Journey
from ..services.claude_client import EFFORT_GUIDANCE, get_claude
from ..utils.types import Recommendation, RiskLevel, SentimentReading
from .prompts import GUIDANCE_SYSTEM
from .state import AgentState

_SYSTEM = GUIDANCE_SYSTEM

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

# Journeys that are high-risk on their own account, because the harm from getting
# them wrong does not depend on the customer disclosing a vulnerability.
_HIGH_RISK_JOURNEYS = {
    Journey.BEREAVEMENT_ESTATE,
    Journey.FRAUD_SCAM,
    Journey.GAMBLING_HARM,
}


def _risk_floor(state: AgentState) -> RiskLevel:
    """The lowest risk level this turn is allowed to carry.

    Three independent routes to high: the vulnerability drivers, the financial
    situation, and the customer's measured distress. Any one is enough — they
    are different ways of being harmed, not three votes on the same question.
    """
    assessment = state.get("assessment")
    context: FinancialContext = state.get("financial_context") or FinancialContext()
    sentiment: SentimentReading = state.get("sentiment") or SentimentReading()

    trig = {d.value for d in assessment.triggered} if assessment else set()

    if trig & _HIGH_RISK_DRIVERS and assessment and assessment.max_score >= 0.8:
        return RiskLevel.HIGH
    if context.journey in _HIGH_RISK_JOURNEYS:
        return RiskLevel.HIGH
    if context.acute:
        return RiskLevel.HIGH
    if sentiment.escalate or sentiment.distress >= 0.7:
        return RiskLevel.HIGH
    if trig or context.high_harm or context.stress_indicators:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _fallback(state: AgentState) -> Recommendation:
    retrieved = state.get("retrieved", [])
    adaptations = [f"{c.title}: {c.text}" for c in retrieved]
    citations = [c.policy_reference for c in retrieved]
    conf = round(min(0.6 + 0.1 * len(retrieved), 0.9), 2) if retrieved else 0.0
    return Recommendation(
        summary=("Consider the prescribed adaptation(s) for the detected "
                 "vulnerability signal." if retrieved else "No policy adaptation retrieved."),
        adaptations=adaptations,
        citations=citations,
        confidence=conf,
        risk_level=_risk_floor(state),
        source="heuristic",
    )


def _build_user_message(state: AgentState) -> str:
    retrieved = state.get("retrieved", [])
    context: FinancialContext = state.get("financial_context") or FinancialContext()
    sentiment: SentimentReading = state.get("sentiment") or SentimentReading()

    policy_text = "\n\n".join(
        f"[{c.policy_reference}] {c.title}: {c.text}" for c in retrieved
    )

    parts = [
        f"Customer said: {state.get('masked_text', '')}",
        "",
        f"Financial context: {context.summary()}"
        + (f" · {context.arrears_months} month(s) in arrears" if context.arrears_months else "")
        + f" · governed by {context.sourcebook}",
    ]
    if context.obligations:
        parts.append("Obligations engaged: " + "; ".join(context.obligations))
    if context.prohibited:
        parts.append("PROHIBITED on this journey — do not propose: "
                     + "; ".join(context.prohibited))
    if sentiment.source != "skipped":
        parts.append(
            f"Customer state: {sentiment.label.lower()} · distress {sentiment.distress:.2f}"
            + (" · voice-corroborated" if sentiment.voice_backed else "")
        )
    parts += [
        "",
        f"Relevant policy clauses:\n{policy_text}",
        "",
        "Write the advisory adaptation grounded only in these clauses.",
    ]
    return "\n".join(parts)


def run(state: AgentState) -> AgentState:
    trace = state.setdefault("trace", [])
    retrieved = state.get("retrieved", [])
    if not retrieved:
        state["recommendation"] = None
        trace.append({"agent": "guidance", "summary": "No retrieval — no recommendation."})
        return state

    llm = get_claude()
    result = llm.structured(_SYSTEM, _build_user_message(state), _SCHEMA,
                            max_tokens=800, effort=EFFORT_GUIDANCE)
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
    else:
        rec = _fallback(state)

    # The model may raise the risk level but never lower it past the floor.
    floor = _risk_floor(state)
    order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
    if order[floor] > order[rec.risk_level]:
        rec.risk_level = floor

    state["recommendation"] = rec
    trace.append({
        "agent": "guidance",
        "summary": f"[{rec.source}] guidance drafted · risk={rec.risk_level.value} · "
                   f"conf={rec.confidence:.2f} · cites {', '.join(rec.citations)}.",
    })
    return state
