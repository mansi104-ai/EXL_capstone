"""Shared state passed between GuardianCX agents in the LangGraph pipeline."""
from __future__ import annotations

from typing import Any, Optional, TypedDict

from ..finance.taxonomy import FinancialContext
from ..utils.types import (
    GuardrailReport,
    PolicyChunk,
    Recommendation,
    SentimentReading,
    VulnerabilityAssessment,
)


def recent_history(state: "AgentState", turns: int = 6) -> str:
    """The last few turns, rendered for a prompt.

    Short on purpose. The whole call is rarely relevant and always expensive;
    what matters is the immediate thread — the question the customer is
    answering, and the thing they disclosed two turns ago.
    """
    history = state.get("history") or []
    if not history:
        return ""
    lines = [f"{t.get('speaker', '?').title()}: {t.get('text', '')}"
             for t in history[-turns:]]
    return "Earlier in this call:\n" + "\n".join(lines)


class AgentState(TypedDict, total=False):
    # inputs
    conversation_id: str
    customer_id: str
    turn_index: int
    speaker: str
    text: str
    channel: str                  # "chat" | "voice"
    voice_signals: dict[str, Any]  # VoiceSignals dump when the turn was spoken
    # The call so far, oldest first: [{"speaker": ..., "text": ...}]. Without it
    # every turn is read in isolation, and "Yes." answering "Would that help?"
    # scores as an unremarkable two-letter utterance.
    history: list[dict[str, str]]

    # produced by agents
    masked_text: str
    input_report: GuardrailReport
    account_request: Any        # AccountRequest — what the ledger may tell them
    financial_context: FinancialContext
    sentiment: SentimentReading
    assessment: VulnerabilityAssessment
    retrieved: list[PolicyChunk]
    recommendation: Optional[Recommendation]
    output_report: GuardrailReport
    trace: list[dict[str, Any]]
    decision: Any          # CaseDecision
    record_id: str
