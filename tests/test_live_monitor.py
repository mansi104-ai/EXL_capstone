"""Render tests for the live console.

The live monitor is where the most moving parts meet — a custom component, a
streamed response, session state that spans reruns, and the full nine-agent
pipeline behind each turn. It is also the page a reviewer will spend their time
on, so these tests drive it the way a person would and assert that nothing
raises and that the signals actually reach the screen.

`AppTest` runs the page headlessly. The voice component's JavaScript does not
execute under it, so the call channel is exercised as far as the component
boundary and the chat channel carries the full turn.
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).parent / "live_monitor_page.py")


def _app() -> AppTest:
    app = AppTest.from_file(PAGE, default_timeout=180)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def test_page_renders_on_every_channel():
    app = _app()
    channels = app.radio[0]
    for option in ("Call", "Chat", "Library"):
        channels.set_value(option).run()
        assert not app.exception, (option, [e.value for e in app.exception])


def test_chat_turn_runs_the_pipeline_and_fills_the_rail():
    app = _app()
    app.radio[0].set_value("Chat").run()

    app.chat_input[0].set_value(
        "My husband passed away last month and I'm three months behind on the mortgage."
    ).run()
    assert not app.exception, [e.value for e in app.exception]

    live = app.session_state["live"]
    # The customer turn, then the streamed handler reply, both recorded.
    assert len(live["turns"]) == 2
    assert live["turns"][0]["speaker"] == "customer"
    assert live["turns"][1]["speaker"] == "agent"
    assert live["turns"][1]["text"]

    state = live["states"][0]
    assert state["financial_context"].journey.value == "bereavement_estate"
    assert state["decision"].risk_level.value == "high"
    assert state["decision"].approval_status.value == "pending"

    # The endpointing model ran on the typed turn too.
    assert live["turns"][0]["eou"]["completeness"] > 0


def test_disclosed_pii_never_reaches_the_transcript():
    app = _app()
    app.radio[0].set_value("Chat").run()
    app.chat_input[0].set_value(
        "I can't pay this month — my sort code is 09-01-22 and my email is jane@example.com"
    ).run()
    assert not app.exception, [e.value for e in app.exception]

    live = app.session_state["live"]
    stored = live["turns"][0]["text"]
    assert "09-01-22" not in stored
    assert "jane@example.com" not in stored
    assert "REDACTED" in stored
    assert set(live["redactor"].total_redactions) >= {"sort_code", "email"}


def test_saving_puts_the_session_in_the_library():
    from guardiancx.database.repository import (
        delete_saved_conversation,
        get_saved_conversation,
    )

    app = _app()
    app.radio[0].set_value("Chat").run()
    app.chat_input[0].set_value("I lost my job and I can't make this payment.").run()
    assert not app.exception, [e.value for e in app.exception]

    conversation_id = app.session_state["live"]["conv_id"]
    save = next(b for b in app.button if b.label == "Save to library")
    save.click().run()
    assert not app.exception, [e.value for e in app.exception]

    try:
        record = get_saved_conversation(conversation_id)
        assert record is not None
        assert record["origin"] == "live"
        assert record["turn_count"] == len(app.session_state["live"]["turns"])
        assert record["max_risk"] in {"low", "medium", "high"}
    finally:
        delete_saved_conversation(conversation_id)


def test_new_conversation_clears_the_session():
    app = _app()
    app.radio[0].set_value("Chat").run()
    app.chat_input[0].set_value("I'm behind on my credit card.").run()
    first_id = app.session_state["live"]["conv_id"]
    assert app.session_state["live"]["turns"]

    next(b for b in app.button if b.label == "New conversation").click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["live"]["conv_id"] != first_id
    assert app.session_state["live"]["turns"] == []


def test_library_replays_a_seeded_conversation():
    app = _app()
    app.radio[0].set_value("Library").run()
    replay = next(b for b in app.button if b.label == "Replay through GuardianCX")
    replay.click().run()
    assert not app.exception, [e.value for e in app.exception]

    states = app.session_state["replay_states"]
    assert states
    # Every turn came back with a decision, and the customer turns were classified.
    assert all("decision" in state for state in states)
    assert any(state["financial_context"].journey.value != "general_servicing"
               for state in states if state.get("speaker") == "customer")
