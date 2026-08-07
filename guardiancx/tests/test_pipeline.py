from guardiancx.agents.graph import process_turn
from guardiancx.database.db import init_engine
from guardiancx.database.repository import clear_all, verify_chain


def setup_module(_):
    init_engine()
    clear_all()


def test_pipeline_detects_and_grounds():
    state = process_turn("T", "C", 0, "customer",
                         "My husband passed away and I can't afford the loan.")
    decision = state["decision"]
    triggered = {d.value for d in decision.assessment.triggered}
    assert "life_events" in triggered
    # recommendation must be grounded in retrieved policy
    assert decision.recommendation is not None
    assert decision.recommendation.citations


def test_neutral_turn_no_recommendation():
    state = process_turn("T", "C", 1, "customer", "Can I order a new debit card?")
    assert state["decision"].recommendation is None


def test_agent_turn_not_assessed():
    state = process_turn("T", "C", 2, "agent", "How can I help you today?")
    assert not state["decision"].assessment.triggered


def test_evidence_chain_valid_after_runs():
    process_turn("T", "C", 3, "customer", "I've been diagnosed with cancer.")
    ok, bad = verify_chain()
    assert ok and bad is None
