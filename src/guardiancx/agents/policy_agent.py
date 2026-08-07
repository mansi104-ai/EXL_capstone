"""Policy Retrieval Agent.

For each triggered driver, retrieves the most relevant policy clause(s) from the
ChromaDB knowledge base (RAG). The retrieved set is what grounds the Guidance
agent and what the hallucination guardrail checks citations against.
"""
from __future__ import annotations

from ..rag.vector_store import ensure_ingested, get_vector_store
from ..utils.types import PolicyChunk
from .state import AgentState


def run(state: AgentState) -> AgentState:
    trace = state.setdefault("trace", [])
    assessment = state.get("assessment")
    if assessment is None or not assessment.triggered:
        state["retrieved"] = []
        trace.append({"agent": "policy", "summary": "No triggered driver — no retrieval."})
        return state

    ensure_ingested()
    store = get_vector_store()
    query = state.get("masked_text") or state.get("text", "")

    # Retrieve the top-2 clauses per triggered driver, keep the highest-scoring
    # unique clauses overall (cap the set so guidance stays focused).
    seen: set[str] = set()
    candidates: list[PolicyChunk] = []
    for driver in assessment.triggered:
        for chunk in store.query(query, driver=driver, top_k=2):
            if chunk.policy_reference not in seen:
                seen.add(chunk.policy_reference)
                candidates.append(chunk)
    candidates.sort(key=lambda c: c.score, reverse=True)
    retrieved = candidates[:4]
    state["retrieved"] = retrieved
    trace.append({
        "agent": "policy",
        "summary": f"[RAG:{store.backend}] retrieved {len(retrieved)} clause(s): "
                   f"{', '.join(c.policy_reference for c in retrieved)}.",
    })
    return state
