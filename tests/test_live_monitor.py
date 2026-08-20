"""Render tests for the live console.

The live monitor is where the most moving parts meet — a custom component, a
streamed response, session state that spans reruns, and the full nine-agent
pipeline behind each turn. It is also the page a reviewer will spend their time
on, so these tests drive it the way a person would and assert that nothing
raises and that the signals actually reach the screen.

`AppTest` runs the page headlessly. The voice component's JavaScript does not
execute under it, so the call channel is exercised as far as the component
boundary and the chat channel carries the full turn.

Every conversation test walks the opening of a real call first — consent, name,
date of birth — because the console now runs that spine before it will discuss
anything. `_serving` is that walk; the stages themselves are tested in
`test_accounts.py`.
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE = str(Path(__file__).parent / "live_monitor_page.py")


def _app() -> AppTest:
    app = AppTest.from_file(PAGE, default_timeout=240)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def _serving(app: AppTest | None = None) -> AppTest:
    """A chat call taken through consent, identification and verification.

    Meera Deshpande is used throughout because her record carries the joint home
    loan and the deceased spouse's sole card, which is what the account and
    disclosure behaviour is worth testing against.
    """
    app = app or _app()
    app.radio[0].set_value("Chat").run()
    for line in ("Yes, that's fine.", "It's Meera Deshpande", "12th March 1958"):
        app.chat_input[0].set_value(line).run()
        assert not app.exception, (line, [e.value for e in app.exception])
    call = app.session_state["live"]["call"]
    assert call.customer_id == "CUST-4471", "identification did not complete"
    assert call.is_verified, "verification did not complete"
    return app


def _customer_turns(live: dict) -> list[dict]:
    """Customer turns from the conversation proper, past the opening stages."""
    return [t for t in live["turns"]
            if t["speaker"] == "customer" and t.get("stage") == "serving"]


def test_page_renders_on_every_channel():
    app = _app()
    channels = app.radio[0]
    for option in ("Call", "Chat", "Library"):
        channels.set_value(option).run()
        assert not app.exception, (option, [e.value for e in app.exception])


def test_chat_turn_runs_the_pipeline_and_fills_the_rail():
    app = _serving()
    app.chat_input[0].set_value(
        "My husband passed away last month and I'm three months behind on the mortgage."
    ).run()
    assert not app.exception, [e.value for e in app.exception]

    live = app.session_state["live"]
    # The customer turn, then the streamed handler reply, both recorded.
    turns = live["turns"]
    assert turns[-2]["speaker"] == "customer"
    assert turns[-1]["speaker"] == "agent"
    assert turns[-1]["text"]

    # The state for the last customer turn — not the first non-empty one, which
    # now belongs to the caller giving their name.
    state = live["states"][len(turns) - 2]
    assert state["financial_context"].journey.value == "bereavement_estate"
    assert state["decision"].risk_level.value == "high"
    assert state["decision"].approval_status.value == "pending"

    # The endpointing model ran on the typed turn too.
    assert turns[-2]["eou"]["completeness"] > 0


def test_disclosed_pii_never_reaches_the_transcript():
    app = _serving()
    app.chat_input[0].set_value(
        "I can't pay this month — my PAN is AFZPD1274K and my email is meera@example.in"
    ).run()
    assert not app.exception, [e.value for e in app.exception]

    live = app.session_state["live"]
    stored = _customer_turns(live)[-1]["text"]
    assert "AFZPD1274K" not in stored
    assert "meera@example.in" not in stored
    assert "REDACTED" in stored
    assert set(live["redactor"].total_redactions) >= {"pan", "email"}


def test_the_date_of_birth_verifies_but_never_reaches_the_transcript():
    """Verification reads the raw utterance; only the verdict survives."""
    app = _serving()
    live = app.session_state["live"]
    assert live["call"].verified == ["date_of_birth"]
    spoken = " ".join(t["text"] for t in live["turns"] if t["speaker"] == "customer")
    assert "1958" not in spoken
    assert "DATE_OF_BIRTH_REDACTED" in spoken


def test_the_caller_hears_their_own_figures():
    app = _serving()
    app.chat_input[0].set_value("How much do I owe on the home loan?").run()
    assert not app.exception, [e.value for e in app.exception]
    reply = app.session_state["live"]["turns"][-1]["text"]
    assert "₹" in reply, reply
    assert "ending" in reply, "the account should be named by its last four digits"


def test_a_third_partys_account_is_still_refused_mid_call():
    app = _serving()
    app.chat_input[0].set_value("Can you tell me my husband's credit card balance?").run()
    assert not app.exception, [e.value for e in app.exception]
    reply = app.session_state["live"]["turns"][-1]["text"].lower()
    assert "someone else" in reply or "not able" in reply


def test_saving_puts_the_session_in_the_library():
    from guardiancx.database.repository import (
        delete_saved_conversation,
        get_saved_conversation,
    )

    app = _serving()
    app.chat_input[0].set_value("I lost my job and I can't make this EMI.").run()
    assert not app.exception, [e.value for e in app.exception]

    conversation_id = app.session_state["live"]["conv_id"]
    save = next(b for b in app.button if b.label == "Save & end call")
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


def test_ending_the_call_closes_it_and_a_new_one_starts_clean():
    app = _serving()
    app.chat_input[0].set_value("I'm behind on my credit card.").run()
    first_id = app.session_state["live"]["conv_id"]
    assert app.session_state["live"]["turns"]

    next(b for b in app.button if b.label == "End call").click().run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["live"]["ended"]

    next(b for b in app.button if b.label == "Start a new call").click().run()
    assert not app.exception, [e.value for e in app.exception]
    live = app.session_state["live"]
    assert live["conv_id"] != first_id
    assert live["turns"] == []
    assert live["call"].customer_id == "", "a new call must not remember the last caller"


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


# --------------------------------------------------------------------------- #
# Sessions that outlive a deploy
# --------------------------------------------------------------------------- #
def test_a_session_written_by_an_older_release_is_repaired():
    """`st.session_state` survives a deploy.

    A tab left open across a release still holds the dict the *old* code wrote,
    so a key added since is simply absent — which crashed the page with
    `KeyError: 'call'` in front of whoever was looking at it.
    """
    app = _app()
    live = app.session_state["live"]
    # Simulate the pre-call-flow shape.
    for key in ("call", "ended", "consent_asks"):
        live.pop(key, None)
    app.run()

    assert not app.exception, [e.value for e in app.exception]
    repaired = app.session_state["live"]
    assert "call" in repaired and "ended" in repaired


def test_a_session_whose_turns_and_states_disagree_is_reset():
    """Pairing a customer's words with another turn's decision would show the
    reviewer something untrue, so the session starts again instead."""
    app = _serving()
    live = app.session_state["live"]
    live["states"].pop()          # knock them out of step
    app.run()

    assert not app.exception, [e.value for e in app.exception]
    assert app.session_state["live"]["turns"] == []


# --------------------------------------------------------------------------- #
# The console must never be left waiting
# --------------------------------------------------------------------------- #
# The browser console shows "Thinking" from the moment it commits a turn until
# Python hands back a new reply id. That id is therefore the only thing standing
# between a working call and a frozen one: if it does not change, the console
# waits forever, with the recogniser still running and the customer still
# talking into a line that has stopped answering.
#
# These drive the session functions directly rather than through AppTest,
# because what is being asserted is the state the component is handed, not what
# is drawn on the page.
from guardiancx.ui import live_monitor as LM  # noqa: E402


# Consent, name, date of birth — the spine every call runs before it will
# discuss anything.
OPENING = ("Yes, that's fine.", "It's Meera Deshpande", "12th March 1958")


def _walk_opening(ss: dict) -> list[str]:
    """Take the call through its opening, collecting the id after each turn."""
    ids = []
    for line in OPENING:
        LM._customer_turn(ss, line, channel="voice")
        ids.append(ss["reply_id"])
    return ids


def _answered_call() -> dict:
    """A voice call taken through consent, name and date of birth."""
    ss = LM._new_session()
    _walk_opening(ss)
    return ss


def test_every_handler_line_gives_the_console_a_fresh_reply_id():
    ids = _walk_opening(LM._new_session())
    assert all(ids), "a handler line left the console with no reply id"
    assert len(set(ids)) == len(ids), "a repeated reply id would freeze the console"


def test_a_failed_synthesis_still_releases_the_console(monkeypatch):
    """Azure being out of quota is a degraded call, not a dead one.

    The id used to be minted only alongside audio, so any synthesis failure left
    the console showing "Thinking" for the rest of the call.
    """
    def explode(*_args, **_kwargs):
        raise RuntimeError("speech quota exhausted")

    monkeypatch.setattr(LM.get_speech(), "synthesize", explode)

    ss = LM._new_session()
    ids = _walk_opening(ss)

    assert ss["reply_audio"] is None, "no audio should have survived the failure"
    assert all(ids) and len(set(ids)) == len(ids)
    assert ss["call"].stage.value == "serving", "the call still has to progress"


def test_an_empty_draft_still_says_something():
    """Every generator behind the reply can produce nothing. Silence on a phone
    call is not an acceptable output, and it also strands the console."""
    ss = _answered_call()
    LM._customer_turn(ss, "Can you check my credit card limit?", channel="voice")
    assert ss["pending"], "a serving turn should defer its reply to the render pass"

    before = ss["reply_id"]
    LM._finalise_reply(ss, "")

    assert ss["turns"][-1]["speaker"] == "agent"
    assert ss["turns"][-1]["text"].strip()
    assert ss["reply_id"] != before, "the console was never told the turn was answered"


def test_the_serving_reply_answers_the_product_that_was_asked_about():
    """She has no credit card, so she is told that — not read her home loan."""
    ss = _answered_call()
    LM._customer_turn(ss, "Can you check my credit card limit?", channel="voice")
    reply = "".join(LM._reply_stream(ss))
    LM._finalise_reply(ss, reply)

    spoken = ss["turns"][-1]["text"].lower()
    assert "credit card" in spoken
    assert "overdue" not in spoken and "arrears" not in spoken
