"""Policy Retrieval Agent.

Retrieves the clauses that ground everything downstream. The retrieved set is
what the Guidance agent may use, and what the hallucination guardrail checks
citations against — so what happens here bounds what the system is able to say.

Retrieval keys off two things, not one:

* the **triggered vulnerability drivers**, which is the FCA axis, and
* the **financial journey**, which is the banking axis.

Both matter, and either alone is wrong. Driver-only retrieval hands a bereaved
customer the same clauses whether they are settling an estate or reporting that
someone emptied the account. Journey-only retrieval ignores why this customer
needs the journey handled differently from the last one.

There is also a case the driver axis misses entirely: a customer three payments
behind, matter-of-fact about it, showing no vulnerability driver at all. No
driver triggers, so a driver-only system retrieves nothing and offers nothing —
while CONC 7.3's forbearance duty is fully engaged. So a high-harm journey
retrieves on its own account, whether or not a driver fired.

Measured distress is the third trigger, and the most important. A customer who
says they feel hopeless may match no driver keyword and no journey pattern; if
retrieval waited for one, the system would answer a crisis disclosure with
account servicing. Any of the three routes is enough to retrieve.
"""
from __future__ import annotations

from ..finance.taxonomy import FinancialContext
from ..rag.vector_store import ensure_ingested, get_vector_store
from ..utils.types import PolicyChunk, SentimentReading
from .state import AgentState

# Most clauses one turn's guidance is allowed to rest on. Beyond this the advice
# stops being actionable on a live call.
MAX_CLAUSES = 4

# Distress at or above this retrieves support policy regardless of drivers.
DISTRESS_RETRIEVAL_THRESHOLD = 0.5


def run(state: AgentState) -> AgentState:
    trace = state.setdefault("trace", [])
    assessment = state.get("assessment")
    context: FinancialContext = state.get("financial_context") or FinancialContext()

    sentiment: SentimentReading = state.get("sentiment") or SentimentReading()

    triggered = list(assessment.triggered) if assessment else []
    journey_retrieval = context.high_harm or context.acute
    distress_retrieval = (sentiment.escalate
                          or sentiment.distress >= DISTRESS_RETRIEVAL_THRESHOLD)

    if not triggered and not journey_retrieval and not distress_retrieval:
        state["retrieved"] = []
        trace.append({"agent": "policy",
                      "summary": "No triggered driver, high-harm journey or distress "
                                 "— no retrieval."})
        return state

    ensure_ingested()
    store = get_vector_store()
    query = state.get("masked_text") or state.get("text", "")

    seen: set[str] = set()
    candidates: list[PolicyChunk] = []

    # Per-driver retrieval, journey-boosted.
    for driver in triggered:
        for chunk in store.query(query, driver=driver, top_k=2, journey=context.journey):
            if chunk.policy_reference not in seen:
                seen.add(chunk.policy_reference)
                candidates.append(chunk)

    # Journey or distress retrieval across all drivers — this is what covers the
    # customer in arrears who discloses no vulnerability at all, and the customer
    # in crisis whose words match no keyword.
    if journey_retrieval or distress_retrieval:
        for chunk in store.query(query, driver=None, top_k=3, journey=context.journey):
            if chunk.policy_reference not in seen:
                seen.add(chunk.policy_reference)
                candidates.append(chunk)

    candidates.sort(key=lambda c: c.score, reverse=True)
    retrieved = candidates[:MAX_CLAUSES]
    state["retrieved"] = retrieved

    matched = sum(1 for c in retrieved if c.journey_match)
    trace.append({
        "agent": "policy",
        "summary": f"[RAG:{store.backend}] retrieved {len(retrieved)} clause(s) for "
                   f"{context.journey.value}: {', '.join(c.policy_reference for c in retrieved)}"
                   + (f" ({matched} journey-matched)" if matched else ""),
    })
    return state
