"""Conversation Agent — entry node.

Normalizes the incoming utterance and runs the *input* guardrails (PII masking,
prompt-injection, toxicity). Produces the masked text everything downstream uses,
so raw PII never reaches the LLM or the logs.
"""
from __future__ import annotations

from ..guardrails.manager import get_guardrail_manager
from .state import AgentState


def run(state: AgentState) -> AgentState:
    text = state.get("text", "")
    speaker = state.get("speaker", "customer")
    masked, report = get_guardrail_manager().run_input(text, speaker)
    state["masked_text"] = masked
    state["input_report"] = report
    trace = state.setdefault("trace", [])
    trace.append({
        "agent": "conversation",
        "summary": f"Ingested {speaker} turn; input guardrails ran "
                   f"({sum(1 for r in report.results if not r.passed)} flag(s)).",
    })
    return state
