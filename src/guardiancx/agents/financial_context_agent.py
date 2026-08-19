"""Financial Context Agent.

The first specialist in the graph, and the one that makes GuardianCX a banking
system rather than a sentiment tool. It answers three questions about the
customer's turn before anything else reasons about it:

* **What product is this about?** — the sourcebook that governs the firm's
  obligations follows from the product.
* **What journey are we in?** — collections, bereavement, scam, forbearance.
  Harm concentrates in a handful of journeys, and the journey is a far sharper
  retrieval key than the raw words.
* **What financial stress is evidenced?** — arrears, essential-spend conflict,
  income shock. This is the detriment axis, independent of the four FCA
  vulnerability drivers, and it is what turns "sounds upset" into "is being
  harmed".

Primary path: the LLM with a strict schema drawn from `finance.taxonomy`.
Fallback: the deterministic phrase classifier, so the journey is never unknown.
Only customer turns are classified.
"""
from __future__ import annotations

from ..finance.taxonomy import (
    FinancialContext,
    Journey,
    Product,
    StressIndicator,
    classify_heuristic,
)
from ..services.claude_client import EFFORT_ANALYSIS, get_claude
from .prompts import FINANCIAL_CONTEXT_SYSTEM
from .state import AgentState, recent_history

_SCHEMA = {
    "type": "object",
    "properties": {
        "product": {"type": "string", "enum": [p.value for p in Product]},
        "journey": {"type": "string", "enum": [j.value for j in Journey]},
        "stress_indicators": {
            "type": "array",
            "items": {"type": "string", "enum": [s.value for s in StressIndicator]},
        },
        "arrears_months": {"type": "integer"},
        "monetary_amounts": {"type": "array", "items": {"type": "string"}},
        "rationale": {"type": "string"},
    },
    "required": ["product", "journey", "stress_indicators", "arrears_months",
                 "monetary_amounts", "rationale"],
    "additionalProperties": False,
}


def _coerce(result: dict, source: str) -> FinancialContext:
    """Build the context from model output, dropping anything outside the
    taxonomy rather than trusting the string."""
    def _enum(cls, value, default):
        try:
            return cls(value)
        except ValueError:
            return default

    indicators: list[StressIndicator] = []
    for raw in result.get("stress_indicators", []) or []:
        try:
            indicators.append(StressIndicator(raw))
        except ValueError:
            continue

    return FinancialContext(
        product=_enum(Product, result.get("product"), Product.UNKNOWN),
        journey=_enum(Journey, result.get("journey"), Journey.GENERAL_SERVICING),
        stress_indicators=indicators,
        arrears_months=max(0, int(result.get("arrears_months") or 0)),
        monetary_amounts=[str(a) for a in (result.get("monetary_amounts") or [])][:6],
        rationale=str(result.get("rationale", "")),
        source=source,
    )


def run(state: AgentState) -> AgentState:
    trace = state.setdefault("trace", [])
    if state.get("speaker") != "customer":
        state["financial_context"] = FinancialContext(source="skipped")
        trace.append({"agent": "financial_context", "summary": "Agent turn — not classified."})
        return state

    text = state.get("masked_text") or state.get("text", "")
    # A short reply only means something against what came before it: "yes" is a
    # journey classification only once you know what was asked.
    history = recent_history(state)
    user = f"{history}\n\nThe customer has just said: {text}" if history else text

    llm = get_claude()
    result = llm.structured(FINANCIAL_CONTEXT_SYSTEM, user, _SCHEMA,
                            max_tokens=600, effort=EFFORT_ANALYSIS)
    context = _coerce(result, llm.provider) if result else classify_heuristic(text)

    # The heuristic reads specific evidence the model can gloss over (a "£"
    # figure, "three months behind"). Where the model returned nothing for those
    # fields, take the deterministic reading rather than losing the detail.
    if context.source != "heuristic":
        fallback = classify_heuristic(text)
        if not context.monetary_amounts:
            context.monetary_amounts = fallback.monetary_amounts
        if not context.arrears_months:
            context.arrears_months = fallback.arrears_months

    state["financial_context"] = context
    trace.append({
        "agent": "financial_context",
        "summary": f"[{context.source}] {context.summary()}"
                   + (f" · {context.arrears_months} month(s) in arrears"
                      if context.arrears_months else ""),
        "obligations": context.obligations,
    })
    return state
