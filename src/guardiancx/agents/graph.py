"""Assemble the GuardianCX multi-agent pipeline.

The flow:

    conversation → account access → financial context → sentiment
                → vulnerability → policy → guidance → compliance
                → supervisor → evidence

Ten specialists, ordered so that each has what it needs. Account access runs
second, because whether the caller may be told something has to be settled before
anything is drafted that might say it; financial context runs early because the
journey it identifies is what makes retrieval precise; sentiment runs before
detection so the acoustic read is available as corroborating evidence; evidence
runs last because nothing is final until it is recorded.

If LangGraph is installed, the nodes are wired into a real StateGraph. Otherwise
a functionally identical sequential runner executes the same node functions, so
the app works either way. Both return the final AgentState.
"""
from __future__ import annotations

from typing import Callable

from ..utils.logging import get_logger
from . import (
    account_agent,
    compliance_agent,
    conversation_agent,
    evidence_agent,
    financial_context_agent,
    guidance_agent,
    policy_agent,
    sentiment_agent,
    supervisor_agent,
    vulnerability_agent,
)
from .state import AgentState

log = get_logger("agents.graph")

# Ordered pipeline of (name, node function).
PIPELINE: list[tuple[str, Callable[[AgentState], AgentState]]] = [
    ("conversation", conversation_agent.run),
    ("account_access", account_agent.run),
    ("financial_context", financial_context_agent.run),
    ("sentiment", sentiment_agent.run),
    ("vulnerability", vulnerability_agent.run),
    ("policy", policy_agent.run),
    ("guidance", guidance_agent.run),
    ("compliance", compliance_agent.run),
    ("supervisor", supervisor_agent.run),
    ("evidence", evidence_agent.run),
]


class Pipeline:
    def __init__(self):
        self.backend = "sequential"
        self._graph = self._build_langgraph()

    def _build_langgraph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except Exception as exc:  # noqa: BLE001
            log.info("LangGraph not installed; using sequential runner (%s).", exc)
            return None

        g = StateGraph(AgentState)
        for name, fn in PIPELINE:
            g.add_node(name, fn)
        g.add_edge(START, PIPELINE[0][0])
        for (prev, _), (nxt, _) in zip(PIPELINE, PIPELINE[1:]):
            g.add_edge(prev, nxt)
        g.add_edge(PIPELINE[-1][0], END)
        self.backend = "langgraph"
        return g.compile()

    def invoke(self, state: AgentState) -> AgentState:
        state.setdefault("trace", [])
        if self._graph is not None:
            return self._graph.invoke(state)
        for _, fn in PIPELINE:
            state = fn(state)
        return state


_PIPELINE: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = Pipeline()
    return _PIPELINE


def process_turn(conversation_id: str, customer_id: str, turn_index: int,
                 speaker: str, text: str, channel: str = "chat",
                 voice_signals: dict | None = None) -> AgentState:
    """Run one utterance through the graph.

    `channel` and `voice_signals` are what let the same pipeline serve a typed
    chat and a live call: on voice the acoustic measurements travel with the turn
    and the sentiment agent fuses them with the words.
    """
    state: AgentState = {
        "conversation_id": conversation_id,
        "customer_id": customer_id,
        "turn_index": turn_index,
        "speaker": speaker,
        "text": text,
        "channel": channel,
        "trace": [],
    }
    if voice_signals:
        state["voice_signals"] = voice_signals
    return get_pipeline().invoke(state)


def process_conversation(conversation: dict) -> list[AgentState]:
    """Run every turn of a synthetic conversation through the pipeline."""
    states: list[AgentState] = []
    for i, turn in enumerate(conversation["turns"]):
        states.append(process_turn(
            conversation_id=conversation["conversation_id"],
            customer_id=conversation.get("customer_id", ""),
            turn_index=i,
            speaker=turn["speaker"],
            text=turn["text"],
            channel=turn.get("channel", conversation.get("channel", "chat")),
            voice_signals=turn.get("voice_signals"),
        ))
    return states
