"""The Live Conversation Monitor.

One console, two channels, one pipeline.

* **Call** keeps the line open. Continuous recognition streams the customer's
  words in as they speak, the end-of-utterance model decides when they have
  actually finished, the reply is streamed as the model writes it, and it is
  spoken back in a neural voice. The whole loop — speech in, nine agents, speech
  out — runs per turn.
* **Chat** is the same pipeline over typed text, for a channel where there is no
  audio to read.
* **Library** replays anything saved, live sessions and the seeded set alike.

To the right of the conversation sits the signal rail: what the agents currently
believe, updated every turn. It is the part a handler would actually watch — the
journey and the obligations it engages, the four vulnerability drivers, the
customer's measured state, what the voice is doing that the words are not, and
what has been redacted out of the transcript before it went anywhere.

Nothing here is autonomous. The reply is a *draft* for a human handler, the
guidance is advisory, and anything high-risk is routed to the approval queue
before it may be used.
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Any, Iterator, Optional

import streamlit as st

from ..agents.graph import process_conversation, process_turn
from ..agents.prompts import HANDLER_REPLY_SYSTEM
from ..agents.consent import (
    CONSENT_REQUEST,
    MAX_CONSENT_ASKS,
    ConsentState,
    classify as classify_consent,
    response_for as consent_response,
)
from ..agents.call_flow import (
    CALL_CLOSED,
    CallStage,
    CallState,
    advance,
    opening_objective,
    wants_to_end,
)
from ..agents.reply import (
    build_reply_prompt,
    compose_reply,
    confirm_offer,
    is_affirmation,
    stage_line,
)
from ..database.repository import (
    delete_saved_conversation,
    list_saved_conversations,
    save_conversation,
)
from ..finance.accounts import BANK_NAME, context_summary, get_customer
from ..finance.taxonomy import JOURNEY_LABELS, PRODUCT_LABELS, STRESS_LABELS, Journey, Product
from ..guardrails.manager import get_guardrail_manager
from ..services.claude_client import EFFORT_REPLY, get_llm
from ..services.speech import get_speech
from ..services.synthetic_data import list_conversations
from ..utils.logging import get_logger
from ..utils.types import Driver, RiskLevel
from ..voice import endpointing, prosody
from ..voice.live_pii import LivePIIRedactor
from . import components as C
from .voice_console import voice_console

log = get_logger("ui.live_monitor")

# Opening lines that exercise a different journey each — the demo is more
# convincing when the first thing a reviewer tries is not the only thing that
# works.
_EXAMPLES = [
    "My husband passed away last month and I'm three months behind on the mortgage.",
    "I lost my job on Friday and I genuinely can't make this month's payment.",
    "Someone called saying they were from the bank and told me to move my money "
    "to a safe account.",
    "I've been gambling again and I've maxed out the card. I don't know what to do.",
    "My daughter usually helps me with this — I find the online banking really confusing.",
]

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _session() -> dict:
    """The live session's own slice of Streamlit state.

    `st.session_state` outlives a deploy. A browser tab left open across a
    release still holds the session dict the *old* code wrote, so any key added
    since — `call`, `ended` — is simply absent, and the first line that reads one
    raises `KeyError` in front of whoever is looking at the screen.

    So the shape is checked on every access rather than only on first use.
    Missing keys are filled from a fresh session, which keeps a call in progress
    alive across a deploy; a transcript that has lost its per-turn states is
    beyond repair and is reset outright.
    """
    live = st.session_state.get("live")
    if not isinstance(live, dict):
        _reset_session()
        return st.session_state["live"]

    template = _new_session()
    missing = [key for key in template if key not in live]
    if missing:
        log.info("Session predates keys %s; filling them in.", missing)
        for key in missing:
            live[key] = template[key]

    # The turns and their states are written in step. If they are not, the
    # renderer would pair a customer's words with another turn's decision, so
    # the session is started again rather than shown wrong.
    if len(live.get("turns", [])) != len(live.get("states", [])):
        log.warning("Session turns and states are out of step; resetting.")
        _reset_session()

    return st.session_state["live"]


def _reset_session(_unused: str = "") -> None:
    """Start a fresh call."""
    st.session_state["live"] = _new_session()


def _new_session() -> dict:
    """A blank call.

    No caller is chosen up front. A real inbound call begins with the handler
    knowing nothing but the ringing phone, and the identification stage is where
    that changes — pre-selecting the customer skipped the part of the call this
    system most needs to get right.

    Kept separate from `_reset_session` so `_session` can use it as the template
    for repairing a session written by an older release.
    """
    token = uuid.uuid4().hex[:6].upper()
    return {
        "conv_id": f"CALL-{token}",
        "customer_id": "",           # set by identification
        "turns": [],                 # {speaker, text, channel, voice, eou, redaction}
        "states": [],                # AgentState per turn ({} for procedural turns)
        "redactor": LivePIIRedactor(),
        "last_seq": 0,               # highest voice-console utterance handled
        "pending": None,             # a customer turn awaiting its streamed reply
        "reply_audio": None,
        "reply_id": "",
        "saved_at": "",
        "last_audio": None,
        "consent": ConsentState.PENDING,
        "consent_asks": 0,
        "consent_rationale": "",
        "call": CallState(),         # where the call has got to
        "ended": "",                 # why the call ended, if it has
    }


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def live_monitor() -> None:
    C.hero(
        "Live Conversation Monitor",
        "Run a real conversation — spoken or typed — through the nine-agent pipeline, "
        "and watch every signal the agents act on as it forms.",
    )

    mode = st.radio(
        "Channel",
        ["Call", "Chat", "Library"],
        horizontal=True,
        label_visibility="collapsed",
        key="live_mode",
    )

    if mode == "Library":
        _library()
        return

    ss = _session()
    left, right = st.columns([1.55, 1], gap="large")

    with right:
        _signal_rail(ss)

    with left:
        # Streamlit addresses an element by where it sits in the tree, and a
        # custom component whose position moves is torn down and rebuilt — the
        # iframe reloads, the recogniser dies, and the console comes back sitting
        # on "Start call" mid-conversation.
        #
        # The header grows a row the moment the caller is verified ("Account on
        # file"), and another when the call is saved. Both sit above the console,
        # so passing verification silently shifted it down one slot and dropped
        # the live call — at the exact moment the call became useful.
        #
        # Claiming the slots up front fixes the console's address for the life of
        # the session: whatever the header decides to show, it grows inside its
        # own container and the console never moves.
        header = st.container()
        call_area = st.container()
        transcript = st.container()

        with header:
            _session_header(ss)
        with call_area:
            if mode == "Call" and not ss["ended"]:
                _call_controls(ss)
        with transcript:
            _conversation(ss, mode)

    # The chat box is pinned to the window, so it lives outside the columns.
    if mode == "Chat" and not ss["ended"]:
        placeholder = {
            ConsentState.PENDING: "Answer the recording question…",
            ConsentState.UNCLEAR: "Answer the recording question…",
        }.get(ss["consent"], _input_placeholder(ss))
        typed = st.chat_input(placeholder)
        if typed and typed.strip():
            _customer_turn(ss, typed.strip(), channel="chat")
            st.rerun()


def _session_header(ss: dict) -> None:
    call: CallState = ss["call"]
    customer = get_customer(call.customer_id) if call.customer_id else None

    c1, c2, c3 = st.columns([2.4, 1, 1])
    with c1:
        who = customer.name if customer else "caller not yet identified"
        st.markdown(f"**{BANK_NAME}** · call {ss['conv_id']} · {who}")
        st.caption(_stage_caption(call, ss))
    if c2.button("Save & end call", width="stretch", disabled=not ss["turns"]):
        _save(ss)
        _end_call(ss, "Ended by the handler; conversation saved.")
        st.rerun()
    if c3.button("End call", width="stretch", disabled=not ss["turns"]):
        _end_call(ss, "Ended by the handler.")
        st.rerun()

    if customer is not None and call.is_verified:
        with st.expander("Account on file"):
            st.code(context_summary(call.customer_id), language="text")
            st.caption("Masked before it reaches the agents, the transcript or the "
                       "evidence record.")
    if ss["saved_at"]:
        st.caption(f"Saved to the conversation library at {ss['saved_at']}.")


def _stage_caption(call: CallState, ss: dict) -> str:
    """One line telling the handler where the call is, and why."""
    if ss["ended"]:
        return f"Call closed — {ss['ended']}"
    if ss["consent"] is not ConsentState.GRANTED:
        return "Waiting for consent to record — nothing is being assessed yet."
    return {
        CallStage.IDENTIFY: "Taking the caller's name.",
        CallStage.VERIFY: "Confirming identity — no account details until this passes.",
        CallStage.SERVING: "Identity confirmed. Account details may be discussed.",
        CallStage.UNVERIFIED: "Not verified — general help only, no account details.",
    }.get(call.stage, "")


def _end_call(ss: dict, reason: str) -> None:
    ss["ended"] = reason
    ss["call"].stage = CallStage.CLOSED
    ss["pending"] = None


def _input_placeholder(ss: dict) -> str:
    """What the chat box invites, given where the call has got to."""
    return {
        CallStage.IDENTIFY: "Give the caller's name…",
        CallStage.VERIFY: "Answer the security question…",
        CallStage.SERVING: "Say what the customer says…",
        CallStage.UNVERIFIED: "Say what the customer says…",
    }.get(ss["call"].stage, "Say what the customer says…")


def _refused_notice(ss: dict) -> None:
    """The call ended because the customer declined to be recorded."""
    for turn in ss["turns"]:
        _render_turn(turn, {})
    st.error(
        "**Call ended — the customer did not consent to being recorded.**  \n"
        "Nothing about this conversation was assessed, and no evidence record was "
        "written. This system analyses what is said, so without consent there is "
        "nothing lawful for it to do."
    )
    if st.button("Start a new call", type="primary"):
        _reset_session(ss["customer_id"])
        st.rerun()


# --------------------------------------------------------------------------- #
# Call channel
# --------------------------------------------------------------------------- #
def _call_controls(ss: dict) -> None:
    speech = get_speech()
    status = speech.status()

    # Same reasoning as the containers in `live_monitor`, one level down: these
    # notices come and go depending on whether a token could be minted this run,
    # and any one of them appearing above the console would move it. The notices
    # render into a slot claimed before the console, so they read above it on the
    # page without ever sitting above it in the tree.
    notices = st.container()
    console = st.container()

    token, region = "", ""
    if status["available"]:
        try:
            token, region = speech.issue_token()
        except Exception as exc:  # noqa: BLE001
            with notices:
                st.warning("Speech recognition is unavailable, so the call will use "
                           "the browser's built-in recogniser instead.")
            log.info("Speech token could not be issued: %s", exc)
    else:
        with notices:
            st.info(
                "Speech services are not configured, so the call will use the "
                "browser's built-in recogniser (Chrome or Edge) and the reply will "
                "be text only."
            )

    with console:
        result = voice_console(
            token=token,
            region=region,
            distress=_current_distress(ss),
            speak_audio=ss["reply_audio"],
            speak_id=ss["reply_id"],
            key=f"voice_{ss['conv_id']}",
        )

    # A committed utterance arrives once; the same value is replayed on every
    # rerun, so the sequence number is what distinguishes new speech from an echo.
    if result and result.get("event") == "utterance" and result.get("seq", 0) > ss["last_seq"]:
        ss["last_seq"] = result["seq"]
        signals = prosody.from_browser(result.get("prosody") or {}, result.get("text", ""))
        _customer_turn(
            ss,
            result.get("text", ""),
            channel="voice",
            voice=signals,
            silence_ms=int(result.get("silence_ms") or 0),
        )
        st.rerun()

    with st.expander("Push to talk instead (records one utterance, analysed on the server)"):
        st.caption(
            "The live call measures prosody in the browser. This path uploads the "
            "recording and analyses the waveform here instead — the same six "
            "measurements, at full audio resolution."
        )
        audio = st.audio_input("Record the customer, then stop", key=f"ptt_{ss['conv_id']}")
        if audio is not None:
            data = audio.getvalue()
            fingerprint = hash(data)
            if data and fingerprint != ss["last_audio"]:
                ss["last_audio"] = fingerprint
                _push_to_talk(ss, data)


def _push_to_talk(ss: dict, data: bytes) -> None:
    speech = get_speech()
    if not speech.available:
        st.warning("Speech services are not configured, so this recording cannot be "
                   "transcribed.")
        return
    try:
        with st.spinner("Transcribing…"):
            spoken = speech.transcribe_wav(data)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Speech transcription failed: {exc}")
        return
    if not spoken:
        st.warning("No speech was recognised — please try again.")
        return
    signals = prosody.analyse(data, spoken)
    _customer_turn(ss, spoken, channel="voice", voice=signals)
    st.rerun()


# --------------------------------------------------------------------------- #
# Turn handling
# --------------------------------------------------------------------------- #
def _customer_turn(ss: dict, raw_text: str, channel: str,
                   voice: Optional[prosody.VoiceSignals] = None,
                   silence_ms: int = 0) -> None:
    """Take one customer utterance through the call.

    Order matters here. Consent gates everything. After that the pipeline runs on
    *every* turn, including the procedural ones, because a customer routinely
    discloses the thing that matters while the handler is still taking their
    name. Only then does the state machine decide what the handler says next.
    """
    voice = voice or prosody.empty()

    if ss["ended"]:
        return

    # The customer can hang up at any point, and they say so in ordinary words.
    # This is checked before anything else, because continuing to process a turn
    # that said "goodbye" is how a phone system ends up answering someone who
    # has already gone.
    if wants_to_end(raw_text):
        ss["turns"].append({
            "speaker": "customer", "text": raw_text, "channel": channel,
            "voice": None, "eou": None, "redaction": None,
            "stage": ss["call"].stage.value,
        })
        ss["states"].append({})
        _record_handler_turn(ss, CALL_CLOSED, channel)
        _end_call(ss, "The customer ended the call.")
        return

    # Nothing is classified, scored or recorded until they agree to be recorded.
    if ss["consent"] is not ConsentState.GRANTED:
        _consent_turn(ss, raw_text, channel)
        return

    call: CallState = ss["call"]

    # Identity checks read the *raw* utterance, because the redactor is about to
    # mask exactly the values being checked. Only the verdict survives.
    outcome = advance(call, raw_text)
    ss["call"] = outcome.state

    redaction = ss["redactor"].feed(raw_text)
    eou = endpointing.detect(redaction.text, silence_ms=silence_ms,
                             distress=voice.distress)

    state = process_turn(
        ss["conv_id"], ss["call"].customer_id, len(ss["turns"]), "customer",
        redaction.text,
        channel=channel,
        voice_signals=voice.model_dump() if voice.available else None,
        history=_history(ss),
    )
    ss["turns"].append({
        "speaker": "customer",
        "text": redaction.text,
        "channel": channel,
        "voice": voice.model_dump() if voice.available else None,
        "eou": eou.model_dump(),
        "redaction": redaction.model_dump(),
        "stage": ss["call"].stage.value,
        "flow_note": outcome.note,
    })
    ss["states"].append(state)
    ss["saved_at"] = ""

    if ss["call"].stage is CallStage.SERVING and outcome.objective == "respond":
        # The conversation proper: the reply is drafted from policy, account and
        # guidance, and streamed on the next render.
        ss["pending"] = {"customer_text": redaction.text, "channel": channel}
    else:
        # A procedural stage. The line is still generated rather than scripted,
        # and it bends around anything the customer just disclosed.
        _procedural_reply(ss, outcome.objective, state, channel)


def _history(ss: dict) -> list[dict]:
    """The call so far, for the agents that need the thread."""
    return [{"speaker": t["speaker"], "text": t["text"]} for t in ss["turns"]]


def _disclosure_summary(state: dict) -> str:
    """What the customer just disclosed that a handler must acknowledge."""
    assessment = state.get("assessment")
    sentiment = state.get("sentiment")
    if assessment is None or not assessment.triggered:
        if sentiment is not None and sentiment.distress >= 0.6:
            return f"they sound {sentiment.label.lower()}"
        return ""
    evidence = [s.evidence for s in assessment.signals
                if s.driver in assessment.triggered and s.evidence]
    drivers = ", ".join(d.value.replace("_", " ") for d in assessment.triggered)
    return f"{drivers} — {evidence[0]}" if evidence else drivers


def _procedural_reply(ss: dict, objective: str, state: dict, channel: str) -> None:
    """Generate and record the handler's line at a non-serving stage."""
    call: CallState = ss["call"]
    customer = get_customer(call.customer_id) if call.customer_id else None
    line = stage_line(
        objective,
        customer_name=customer.name if customer else call.claimed_name,
        last_utterance=ss["turns"][-1]["text"] if ss["turns"] else "",
        disclosure=_disclosure_summary(state),
        history=_history_text(ss),
    )
    _record_handler_turn(ss, line, channel)


def _history_text(ss: dict, turns: int = 4) -> str:
    recent = ss["turns"][-turns:]
    if not recent:
        return ""
    lines = [f"{t['speaker'].title()}: {t['text']}" for t in recent]
    return "Earlier in this call:\n" + "\n".join(lines)


def _record_handler_turn(ss: dict, line: str, channel: str) -> None:
    """Append a handler line, check it, and speak it on the call channel."""
    if not line or not line.strip():
        return
    call: CallState = ss["call"]
    previous = ss["states"][-1] if ss["states"] else {}
    context = previous.get("financial_context") if previous else None
    account = previous.get("account_request") if previous else None
    report = get_guardrail_manager().run_reply(
        line,
        journey=context.journey if context else None,
        account_refused=bool(account is not None and getattr(account, "refused", False)),
    )
    ss["turns"].append({
        "speaker": "agent", "text": line.strip(), "channel": channel,
        "voice": None, "eou": None, "redaction": None,
        "clarity": [r.model_dump() for r in report.results],
        "stage": call.stage.value,
    })
    ss["states"].append({})
    if channel == "voice":
        _speak(ss, line)


def _consent_turn(ss: dict, raw_text: str, channel: str) -> None:
    """Handle the customer's answer to the recording request."""
    decision = classify_consent(raw_text)
    ss["consent"] = decision.state
    ss["consent_rationale"] = decision.rationale

    ss["turns"].append({
        "speaker": "customer", "text": raw_text, "channel": channel,
        "voice": None, "eou": None, "redaction": None,
        "consent": decision.model_dump(mode="json"),
    })
    ss["states"].append({})

    reply = consent_response(decision, ss["consent_asks"])

    if decision.state is ConsentState.GRANTED:
        # Consent given: the call moves on to finding out who is calling, and
        # the handler's next line is that question rather than a confirmation
        # nobody needs to hear.
        ss["call"].stage = CallStage.IDENTIFY
        _record_handler_turn(ss, stage_line(
            opening_objective(ss["call"]),
            last_utterance=raw_text,
            history=_history_text(ss),
        ), channel)
        return

    if decision.state is ConsentState.REFUSED:
        ss["ended"] = "The customer did not consent to being recorded."
        ss["call"].stage = CallStage.CLOSED

    if decision.state is ConsentState.UNCLEAR:
        # Asked again. A non-answer never becomes a refusal on its own — only
        # the customer saying no does that.
        ss["consent_asks"] += 1
        if ss["consent_asks"] >= MAX_CONSENT_ASKS:
            # Out of asks. The call closes, but not as a refusal — the customer
            # never said no, and the record must not claim they did.
            ss["ended"] = "No answer to the recording question."

    if reply:
        ss["turns"].append({
            "speaker": "agent", "text": reply, "channel": channel,
            "voice": None, "eou": None, "redaction": None, "clarity": [],
        })
        ss["states"].append({})
        if channel == "voice":
            _speak(ss, reply)


def _speak(ss: dict, text: str) -> None:
    """Synthesise a line for playback on the call channel.

    The id is minted for every handler line, before synthesis is attempted and
    whether or not it succeeds, because the console uses it as the signal that
    the turn has been answered. It used to be set only alongside audio, which
    tied the browser's whole state machine to Azure being reachable: with speech
    unconfigured, out of quota, or simply failing, the id never changed, the
    console never left "Thinking", and the call froze mid-sentence with no way
    back. A reply nobody can hear is a degraded call. A console that stops
    listening without saying so is a dead one.
    """
    ss["reply_audio"] = None
    ss["reply_id"] = uuid.uuid4().hex[:12]

    speech = get_speech()
    if not speech.available:
        return
    try:
        audio = speech.synthesize(text, emotion=_current_emotion(ss))
    except Exception as exc:  # noqa: BLE001
        log.warning("Reply could not be synthesised; the console will show it as "
                    "text only (%s).", exc)
        return
    if audio:
        ss["reply_audio"] = audio


def _reply_stream(ss: dict) -> Iterator[str]:
    """Yield the handler's draft reply as the model produces it."""
    pending = ss["pending"]
    state = ss["states"][-1]
    decision = state.get("decision")
    llm = get_llm()

    if llm.available:
        # The prompt is built from policy already translated into customer-facing
        # offers, so the model is reading the vocabulary it should be using
        # rather than the industry term it just saw.
        user = build_reply_prompt(
            pending["customer_text"], decision,
            context=state.get("financial_context"),
            sentiment=state.get("sentiment"),
            retrieved=state.get("retrieved"),
            account=state.get("account_request"),
            history=_history_text(ss, turns=6),
        )
        produced = False
        for chunk in llm.stream_text(HANDLER_REPLY_SYSTEM, user,
                                     max_tokens=200, effort=EFFORT_REPLY):
            produced = True
            yield chunk
        if produced:
            return

    # No model, or the model produced nothing: compose the reply from policy
    # ourselves, released a sentence at a time so the channel behaves the same
    # way. This is the path every demo runs on until a key is configured.
    # A bare "yes" carries no signal of its own — it means whatever the handler
    # just offered. Without a model to read the thread, that link has to be made
    # here or the reply asks the customer what they need, one turn after
    # offering it to them.
    if is_affirmation(pending["customer_text"]):
        previous = next((t["text"] for t in reversed(ss["turns"][:-1])
                         if t["speaker"] == "agent"), "")
        yield from _stream_sentences(confirm_offer(previous))
        return

    composed = compose_reply(decision, state.get("financial_context"),
                             state.get("sentiment"), retrieved=state.get("retrieved"),
                             account=state.get("account_request"))
    yield from _stream_sentences(composed)


def _stream_sentences(text: str) -> Iterator[str]:
    """Release a composed reply a sentence at a time.

    Splitting on ". " and re-appending it turned "Would that help?" into
    "Would that help?." — the terminator has to be kept, not assumed.
    """
    for sentence in re.findall(r"[^.!?]+[.!?]*\s*", text):
        if sentence.strip():
            yield sentence
            time.sleep(0.12)


# What the handler says when the draft comes back empty. Every generator behind
# the reply can legitimately produce nothing — no model configured, a stream that
# failed, a composer with no facts and no policy to work from — and on a phone
# call the result of that was silence: no line recorded, so no reply id, so a
# console waiting on a turn that was never going to be answered. Silence is not
# an acceptable output of a support line, so there is always a line.
EMPTY_REPLY = ("Sorry — bear with me one moment. Could you tell me a little more "
               "about what you need?")


def _finalise_reply(ss: dict, reply: str) -> None:
    """Record the streamed handler reply, and speak it on the call channel."""
    pending = ss["pending"]
    ss["pending"] = None
    reply = (reply or "").strip()
    if not reply:
        log.warning("The reply draft came back empty; falling back to a holding line.")
        reply = EMPTY_REPLY
    _record_handler_turn(ss, reply, pending["channel"])


# --------------------------------------------------------------------------- #
# Conversation rendering
# --------------------------------------------------------------------------- #
def _conversation(ss: dict, mode: str) -> None:
    # The call always opens the same way: the handler asks to record, and waits.
    if ss["consent"] is ConsentState.PENDING and not ss["turns"]:
        with st.chat_message("assistant"):
            st.markdown(f"**Handler:** {CONSENT_REQUEST}")
        st.caption("Nothing is assessed until the customer answers. Try “yes, "
                   "that's fine”, “I'd rather you didn't”, or “what for?”.")
        return

    for turn, state in zip(ss["turns"], ss["states"]):
        _render_turn(turn, state)

    if ss["ended"]:
        # A refusal is a correct outcome, not a failure — but it must not be
        # missed, so it is styled louder than a handler simply hanging up.
        if ss["consent"] is ConsentState.REFUSED:
            st.error(
                f"**Call ended — the customer did not consent to being recorded.** "
                f"Nothing was assessed and no evidence record was written. This "
                f"system analyses what is said, so without consent there is "
                f"nothing lawful for it to do.")
        else:
            st.info(f"**Call ended.** {ss['ended']}")
        if st.button("Start a new call", type="primary", key="new_after_end"):
            _reset_session()
            st.rerun()
        return

    if ss["consent"] is ConsentState.UNCLEAR:
        st.caption("The customer has not answered yes or no — the handler asks once "
                   "more, plainly. A question is not consent.")
        return

    if ss["pending"]:
        with st.chat_message("assistant"):
            st.markdown("**Handler (draft):**")
            reply = st.write_stream(_reply_stream(ss))
        _finalise_reply(ss, reply if isinstance(reply, str) else "".join(reply))
        st.rerun()

    if ss["call"].stage is CallStage.SERVING and len(ss["turns"]) <= 6:
        st.caption("Identity confirmed. Try: “I've missed two EMIs on the home loan”, "
                   "“my husband passed away last month”, or “someone took ₹40,000 "
                   "from my account”.")


def _render_turn(turn: dict, state: dict) -> None:
    role = "user" if turn["speaker"] == "customer" else "assistant"
    with st.chat_message(role):
        label = "Customer" if turn["speaker"] == "customer" else "Handler"
        icon = " 🎙️" if turn.get("channel") == "voice" else ""
        st.markdown(f"**{label}{icon}:** {turn['text']}")
        if turn.get("flow_note"):
            st.caption(turn["flow_note"])

        if turn["speaker"] == "customer":
            _consent_signal(turn)
            _turn_signals(turn)
        else:
            _reply_signals(turn)

        decision = state.get("decision")
        if turn["speaker"] != "customer" or decision is None:
            return

        triggered = decision.assessment.triggered
        heading = "Agent decision trace" + (
            " — vulnerability signal detected" if triggered else " — no driver triggered")
        with st.expander(heading, expanded=bool(triggered)):
            C.agent_trace(state.get("trace", []))
            st.markdown("**Guardrails applied**")
            C.guardrail_badges([r.model_dump() for r in decision.guardrails.results])
        if decision.recommendation:
            C.recommendation_card(decision)


def _turn_signals(turn: dict) -> None:
    """The per-turn line under a customer message: redaction, endpointing, voice."""
    bits: list[str] = []

    redaction = turn.get("redaction") or {}
    if redaction.get("redactions"):
        caught = ", ".join(f"{k.replace('_', ' ')}×{v}"
                           for k, v in redaction["redactions"].items())
        bits.append(f"🛡️ redacted {caught}")
    if redaction.get("security_request"):
        bits.append("⛔ credential request — never ask for this")

    eou = turn.get("eou") or {}
    if eou:
        bits.append(f"⏹️ endpoint: {eou['reason']} "
                    f"(completeness {eou['completeness']:.2f}, "
                    f"{eou['required_silence_ms']}ms window)")

    voice = turn.get("voice") or {}
    if voice.get("available"):
        flags = prosody.VoiceSignals(**voice).flags
        bits.append("🔊 " + ("; ".join(flags) if flags else "voice steady"))

    if bits:
        st.caption("  ·  ".join(bits))


def _consent_signal(turn: dict) -> None:
    """Show how the customer's answer to the recording request was read."""
    consent = turn.get("consent")
    if not consent:
        return
    state = consent.get("state", "")
    mark = {"granted": "Consent given", "refused": "Consent refused",
            "unclear": "Answer unclear"}.get(state, state)
    st.caption(f"{mark} — {consent.get('rationale', '')} [{consent.get('source', '')}]")


def _reply_signals(turn: dict) -> None:
    """The clarity read on a drafted reply — shown to the handler who will send it."""
    for result in turn.get("clarity") or []:
        mark = "OK" if result["passed"] else "!"
        st.caption(f"{mark} {result['detail']}")


# --------------------------------------------------------------------------- #
# Signal rail
# --------------------------------------------------------------------------- #
def _customer_states(ss: dict) -> list[dict]:
    return [s for t, s in zip(ss["turns"], ss["states"]) if t["speaker"] == "customer"]


def _latest(ss: dict, key: str):
    for state in reversed(_customer_states(ss)):
        value = state.get(key)
        if value is not None:
            return value
    return None


def _current_distress(ss: dict) -> float:
    sentiment = _latest(ss, "sentiment")
    return float(sentiment.distress) if sentiment else 0.0


def _current_emotion(ss: dict) -> str:
    sentiment = _latest(ss, "sentiment")
    return sentiment.emotion if sentiment else ""


def _signal_rail(ss: dict) -> None:
    states = _customer_states(ss)
    st.markdown("##### Live signals")
    if not states:
        C.quiet("The rail fills in as soon as the customer speaks. Every number here "
                "is one an agent produced this turn — nothing is precomputed.")
        return

    latest = states[-1]
    decision = latest.get("decision")
    context = latest.get("financial_context")
    sentiment = latest.get("sentiment")

    # --- what this call is, in the firm's terms --------------------------
    C.rail_title("Financial context")
    if context:
        st.markdown(
            f"**{JOURNEY_LABELS.get(context.journey, context.journey.value)}**  ·  "
            f"{PRODUCT_LABELS.get(context.product, '—')}  ·  `{context.sourcebook}`"
        )
        C.chips([STRESS_LABELS.get(s, s.value) for s in context.stress_indicators],
                tone="alert" if context.acute else "on")
        if context.arrears_months:
            C.quiet(f"{context.arrears_months} month(s) in arrears")
        if context.obligations:
            with st.expander("Obligations engaged"):
                for obligation in context.obligations:
                    st.markdown(f"- {obligation}")
                if context.prohibited:
                    st.markdown("**Must not be offered on this journey**")
                    for item in context.prohibited:
                        st.markdown(f"- {item}")

    # --- the FCA axis ----------------------------------------------------
    C.rail_title("Vulnerability drivers — highest this call")
    peaks = {d: 0.0 for d in Driver}
    for state in states:
        assessment = state.get("assessment")
        if not assessment:
            continue
        for signal in assessment.signals:
            peaks[signal.driver] = max(peaks[signal.driver], signal.score)
    for driver, score in peaks.items():
        C.meter(driver.value.replace("_", " "), score, C.scale_color(score))

    # --- how the customer is ---------------------------------------------
    C.rail_title("Customer state")
    if sentiment:
        st.markdown(f"**{sentiment.label}**" +
                    ("  ·  voice-corroborated" if sentiment.voice_backed else ""))
        C.meter("distress", sentiment.distress, C.scale_color(sentiment.distress))
        C.meter("arousal", sentiment.arousal, C.scale_color(sentiment.arousal))
        C.meter("valence", (sentiment.valence + 1) / 2,
                C.scale_color(1 - (sentiment.valence + 1) / 2),
                caption=f"raw {sentiment.valence:+.2f} (−1 negative → +1 positive)")
        if sentiment.escalate:
            st.error("Escalate — slow down, stop any collections or sales process, "
                     "and prioritise the person over the transaction.")
        _distress_trend(states)

    # --- what the voice is doing -----------------------------------------
    voice_dump = None
    for state in reversed(states):
        if state.get("voice_signals"):
            voice_dump = state["voice_signals"]
            break
    C.rail_title("Voice signals")
    if voice_dump:
        voice = prosody.VoiceSignals(**voice_dump)
        C.meter("agitation", voice.agitation, C.scale_color(voice.agitation))
        C.meter("tremor", voice.tremor, C.scale_color(voice.tremor))
        C.meter("hesitancy", voice.hesitancy, C.scale_color(voice.hesitancy))
        C.quiet(voice.summary())
    else:
        C.quiet("Text channel — no acoustic signal to read.")

    # --- what never left the room ----------------------------------------
    C.rail_title("Live PII redaction")
    totals = ss["redactor"].total_redactions
    if totals:
        C.chips([f"{k.replace('_', ' ')} ×{v}" for k, v in totals.items()], tone="ok")
        C.quiet("Redacted on the way in — the agents, the logs and this screen only "
                "ever saw the masked text.")
    else:
        C.quiet("Nothing personal has been disclosed yet this call.")
    if ss["redactor"].expecting:
        st.warning(f"Holding — the caller is about to give their "
                   f"{ss['redactor'].expecting.replace('_', ' ')}.")

    # --- where it lands ---------------------------------------------------
    C.rail_title("Routing")
    if decision:
        st.markdown(
            f"{C.risk_pill(decision.risk_level.value)} "
            f"{C.pill('approval: ' + decision.approval_status.value, C.WARN if decision.approval_status.value == 'pending' else C.ACCENT)}",
            unsafe_allow_html=True,
        )
        if decision.approval_status.value == "pending":
            C.quiet("Routed to the Human Approval Queue — a person signs this off "
                    "before it may be used.")


def _distress_trend(states: list[dict]) -> None:
    """Distress across the call. One line, because the shape is the point: a call
    that is getting worse needs a different response from one that is settling."""
    series = [s["sentiment"].distress for s in states if s.get("sentiment")]
    if len(series) < 2:
        return
    st.line_chart({"distress": series}, height=110, color=C.DANGER)


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def _save(ss: dict) -> None:
    states = _customer_states(ss)
    drivers, journeys = set(), set()
    peak_distress, max_risk = 0.0, RiskLevel.LOW
    order = {RiskLevel.LOW: 0, RiskLevel.MEDIUM: 1, RiskLevel.HIGH: 2}
    product = Product.UNKNOWN

    for state in states:
        assessment = state.get("assessment")
        if assessment:
            drivers.update(d.value for d in assessment.triggered)
        context = state.get("financial_context")
        if context:
            journeys.add(context.journey.value)
            if context.product is not Product.UNKNOWN:
                product = context.product
        sentiment = state.get("sentiment")
        if sentiment:
            peak_distress = max(peak_distress, sentiment.distress)
        decision = state.get("decision")
        if decision and order[decision.risk_level] > order[max_risk]:
            max_risk = decision.risk_level

    call: CallState = ss["call"]
    customer = get_customer(call.customer_id) if call.customer_id else None
    channel = "voice" if any(t.get("channel") == "voice" for t in ss["turns"]) else "chat"
    save_conversation(
        conversation_id=ss["conv_id"],
        turns=[{"speaker": t["speaker"], "text": t["text"], "channel": t.get("channel", "chat")}
               for t in ss["turns"]],
        customer_id=call.customer_id,
        customer_name=customer.name if customer else "Unidentified caller",
        product=PRODUCT_LABELS.get(product, ""),
        channel=channel,
        origin="live",
        max_risk=max_risk.value,
        drivers=sorted(drivers),
        journeys=sorted(journeys),
        peak_distress=peak_distress,
        note=(f"Captured live on the {channel} channel. "
              + ("Caller verified." if call.is_verified else "Caller not verified.")),
    )
    ss["saved_at"] = time.strftime("%H:%M:%S")


# --------------------------------------------------------------------------- #
# Library
# --------------------------------------------------------------------------- #
def _library() -> None:
    saved = list_saved_conversations()
    seeded = list_conversations()

    options: dict[str, dict[str, Any]] = {}
    for record in saved:
        badge = "live" if record["origin"] == "live" else "seed"
        options[f"[{badge}] {record['conversation_id']} — {record['customer_name'] or 'Live session'} "
                f"· {record['turn_count']} turns · risk {record['max_risk']}"] = {
            "kind": "saved", "record": record}
    for conv in seeded:
        options[f"[seed] {conv['conversation_id']} — {conv['customer_name']} "
                f"· {len(conv['turns'])} turns"] = {"kind": "seed", "record": conv}

    if not options:
        st.info("Nothing saved yet. Run a conversation on the Call or Chat channel "
                "and press *Save to library*.")
        return

    st.caption(f"{len(saved)} saved · {len(seeded)} seeded. Live sessions appear here as "
               "soon as they are saved, and feed the Customer Timeline and Analytics "
               "pages alongside the seeded set.")
    choice = st.selectbox("Conversation", list(options))
    entry = options[choice]
    record = entry["record"]

    if entry["kind"] == "saved":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Turns", record["turn_count"])
        c2.metric("Peak risk", record["max_risk"])
        c3.metric("Peak distress", f"{record['peak_distress']:.2f}")
        c4.metric("Channel", record["channel"])
        if record["journeys"]:
            C.chips([JOURNEY_LABELS.get(Journey(j), j) for j in record["journeys"]
                     if j in Journey._value2member_map_], tone="on")
        st.caption(f"Saved {record['updated_at'][:19].replace('T', ' ')} · "
                   f"customer {record['customer_id']} · {record['note']}")
    else:
        st.caption(f"Product: {record['product']} · Channel: {record['channel']} · "
                   f"Customer: {record['customer_id']}")

    conversation = {
        "conversation_id": record["conversation_id"],
        "customer_id": record.get("customer_id", ""),
        "channel": record.get("channel", "chat"),
        "turns": record["turns"],
    }

    c1, c2 = st.columns([1, 3])
    if c1.button("Replay through GuardianCX", type="primary"):
        with st.spinner("Nine agents working through every turn…"):
            st.session_state["replay_states"] = process_conversation(conversation)
            st.session_state["replay_id"] = record["conversation_id"]
    if entry["kind"] == "saved" and record["origin"] == "live":
        if c2.button("Delete this saved conversation"):
            delete_saved_conversation(record["conversation_id"])
            st.session_state.pop("replay_states", None)
            st.rerun()

    if st.session_state.get("replay_id") == record["conversation_id"]:
        for turn, state in zip(conversation["turns"],
                               st.session_state.get("replay_states", [])):
            _render_turn(
                {"speaker": turn["speaker"], "text": turn["text"],
                 "channel": turn.get("channel", "chat"), "voice": None,
                 "eou": None, "redaction": None},
                state,
            )
    else:
        with st.expander("Transcript", expanded=True):
            for turn in conversation["turns"]:
                st.markdown(f"**{turn['speaker'].title()}:** {turn['text']}")
