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

import time
import uuid
from typing import Any, Iterator, Optional

import streamlit as st

from ..agents.graph import process_conversation, process_turn
from ..agents.prompts import HANDLER_REPLY_SYSTEM
from ..database.repository import (
    delete_saved_conversation,
    list_saved_conversations,
    save_conversation,
)
from ..finance.taxonomy import JOURNEY_LABELS, PRODUCT_LABELS, STRESS_LABELS, Journey, Product
from ..services.claude_client import EFFORT_REPLY, get_llm
from ..services.speech import get_speech
from ..services.synthetic_data import list_conversations
from ..utils.types import Driver, RiskLevel
from ..voice import endpointing, prosody
from ..voice.live_pii import LivePIIRedactor
from . import components as C
from .voice_console import voice_console

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

_OPENERS = {
    Driver.LIFE_EVENTS: "I'm very sorry to hear that.",
    Driver.HEALTH: "Thank you for letting me know, and I'm sorry you're dealing with this.",
    Driver.RESILIENCE: "I understand, and I want to make this as manageable as possible for you.",
    Driver.CAPABILITY: "Of course — I'll keep this simple and go at your pace.",
}


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def _session() -> dict:
    """The live session's own slice of Streamlit state."""
    if "live" not in st.session_state:
        _reset_session()
    return st.session_state["live"]


def _reset_session() -> None:
    token = uuid.uuid4().hex[:6].upper()
    st.session_state["live"] = {
        "conv_id": f"LIVE-{token}",
        "customer_id": f"CUST-{token}",
        "turns": [],          # {speaker, text, channel, voice, eou, redaction}
        "states": [],         # AgentState per turn
        "redactor": LivePIIRedactor(),
        "last_seq": 0,        # highest voice-console utterance handled
        "pending": None,      # a customer turn awaiting its streamed reply
        "reply_audio": None,
        "reply_id": "",
        "saved_at": "",
        "last_audio": None,
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
        _session_header(ss)
        if mode == "Call":
            _call_controls(ss)
        _conversation(ss, mode)

    # The chat box is pinned to the window, so it lives outside the columns.
    if mode == "Chat":
        typed = st.chat_input("Type what the customer says…")
        if typed and typed.strip():
            _customer_turn(ss, typed.strip(), channel="chat")
            st.rerun()


def _session_header(ss: dict) -> None:
    llm = get_llm()
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.caption(f"Conversation **{ss['conv_id']}** · customer **{ss['customer_id']}** · "
               f"agent model {llm.provider_label}")
    if c2.button("Save to library", width="stretch", disabled=not ss["turns"]):
        _save(ss)
    if c3.button("New conversation", width="stretch"):
        _reset_session()
        st.rerun()
    if ss["saved_at"]:
        st.caption(f"Saved to the conversation library at {ss['saved_at']} — "
                   "it now appears under Library, Customer Timeline and Analytics.")


# --------------------------------------------------------------------------- #
# Call channel
# --------------------------------------------------------------------------- #
def _call_controls(ss: dict) -> None:
    speech = get_speech()
    status = speech.status()

    token, region = "", ""
    if status["available"]:
        try:
            token, region = speech.issue_token()
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Azure Speech token could not be issued — falling back to the "
                       f"browser's own recogniser. ({exc})")
    else:
        st.info(
            "Azure Speech is not configured, so the call will use the browser's built-in "
            "recogniser (Chrome or Edge) and there will be no spoken reply. "
            f"{status['reason']}"
        )

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
        st.warning("Azure Speech is not configured, so this recording cannot be transcribed.")
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
    """Take one customer utterance through redaction, endpointing and the graph."""
    voice = voice or prosody.empty()

    # Redaction runs first, on the way in. Everything downstream — the agents, the
    # evidence record, the screen — sees only the redacted text.
    redaction = ss["redactor"].feed(raw_text)

    # The authoritative endpointing decision. On a call the browser has already
    # made a timing call to commit the turn; re-running the model here is what
    # gets traced and recorded, and it is the only decision on the text channel.
    eou = endpointing.detect(
        redaction.text, silence_ms=silence_ms, distress=voice.distress,
    )

    state = process_turn(
        ss["conv_id"], ss["customer_id"], len(ss["turns"]), "customer", redaction.text,
        channel=channel,
        voice_signals=voice.model_dump() if voice.available else None,
    )
    ss["turns"].append({
        "speaker": "customer",
        "text": redaction.text,
        "channel": channel,
        "voice": voice.model_dump() if voice.available else None,
        "eou": eou.model_dump(),
        "redaction": redaction.model_dump(),
    })
    ss["states"].append(state)
    # The reply is drafted on the next render so it can be streamed into view.
    ss["pending"] = {"customer_text": redaction.text, "channel": channel}
    ss["saved_at"] = ""


def _reply_stream(ss: dict) -> Iterator[str]:
    """Yield the handler's draft reply as the model produces it."""
    pending = ss["pending"]
    state = ss["states"][-1]
    decision = state.get("decision")
    llm = get_llm()

    if llm.available:
        rec = decision.recommendation if decision else None
        guidance = ((rec.summary + " " + " ".join(rec.adaptations)) if rec
                    else "(no specific policy retrieved)")
        context = state.get("financial_context")
        sentiment = state.get("sentiment")
        user = (
            f"Customer said: {pending['customer_text']}\n"
            f"Financial context: {context.summary() if context else 'unclassified'}\n"
            f"Customer state: {sentiment.label if sentiment else 'unknown'}\n"
            f"Policy guidance: {guidance}\n\n"
            "Write the handler's reply:"
        )
        produced = False
        for chunk in llm.stream_text(HANDLER_REPLY_SYSTEM, user,
                                     max_tokens=200, effort=EFFORT_REPLY):
            produced = True
            yield chunk
        if produced:
            return

    # No model, or the model produced nothing: fall back to the policy-grounded
    # template, released a sentence at a time so the channel behaves the same way.
    for sentence in _template_reply(decision).split(". "):
        if sentence:
            yield sentence.rstrip(".") + ". "
            time.sleep(0.12)


def _template_reply(decision) -> str:
    """A policy-grounded handler reply, used when no LLM is configured."""
    if decision is None:
        return "Thank you — how can I help you today?"
    if not decision.assessment.triggered:
        return "Thanks — I can help you with that. Let me pull up your account."
    top = max(decision.assessment.signals, key=lambda s: s.score).driver
    opener = _OPENERS.get(top, "Thank you for telling me.")
    rec = decision.recommendation
    if rec and rec.adaptations:
        return (f"{opener} Here's how I can help, in line with our policy: "
                f"{_short(rec.adaptations[0])}")
    return f"{opener} Let me talk you through the support available."


def _short(text: str, limit: int = 220) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    dot = cut.rfind(". ")
    return (cut[: dot + 1] if dot > 60 else cut).rstrip() + "…"


def _finalise_reply(ss: dict, reply: str) -> None:
    """Record the handler's turn and, on a call, synthesise it for playback."""
    pending = ss["pending"]
    ss["pending"] = None
    reply = (reply or "").strip()
    if not reply:
        return

    state = process_turn(
        ss["conv_id"], ss["customer_id"], len(ss["turns"]), "agent", reply,
        channel=pending["channel"],
    )
    ss["turns"].append({"speaker": "agent", "text": reply, "channel": pending["channel"],
                        "voice": None, "eou": None, "redaction": None})
    ss["states"].append(state)

    if pending["channel"] != "voice":
        return

    speech = get_speech()
    if not speech.available:
        return
    try:
        # The customer's emotional state selects the speaking style — a distressed
        # caller hears an empathetic delivery, an angry one a deliberately calm one.
        emotion = _current_emotion(ss)
        audio = speech.synthesize(reply, emotion=emotion)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"The reply could not be synthesised: {exc}")
        return
    if audio:
        ss["reply_audio"] = audio
        ss["reply_id"] = uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# Conversation rendering
# --------------------------------------------------------------------------- #
def _conversation(ss: dict, mode: str) -> None:
    if not ss["turns"] and not ss["pending"]:
        st.info(
            "Start the conversation. On **Call**, press *Start call* and speak; on "
            "**Chat**, type below. Try one of these:"
        )
        for example in _EXAMPLES[:3]:
            st.markdown(f"- *“{example}”*")
        return

    for turn, state in zip(ss["turns"], ss["states"]):
        _render_turn(turn, state)

    if ss["pending"]:
        with st.chat_message("assistant"):
            st.markdown("**Handler (draft):**")
            reply = st.write_stream(_reply_stream(ss))
        _finalise_reply(ss, reply if isinstance(reply, str) else "".join(reply))
        st.rerun()


def _render_turn(turn: dict, state: dict) -> None:
    role = "user" if turn["speaker"] == "customer" else "assistant"
    with st.chat_message(role):
        label = "Customer" if turn["speaker"] == "customer" else "Handler (draft)"
        icon = " 🎙️" if turn.get("channel") == "voice" else ""
        st.markdown(f"**{label}{icon}:** {turn['text']}")

        if turn["speaker"] == "customer":
            _turn_signals(turn)

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

    channel = "voice" if any(t.get("channel") == "voice" for t in ss["turns"]) else "chat"
    save_conversation(
        conversation_id=ss["conv_id"],
        turns=[{"speaker": t["speaker"], "text": t["text"], "channel": t.get("channel", "chat")}
               for t in ss["turns"]],
        customer_id=ss["customer_id"],
        customer_name="Live session",
        product=PRODUCT_LABELS.get(product, ""),
        channel=channel,
        origin="live",
        max_risk=max_risk.value,
        drivers=sorted(drivers),
        journeys=sorted(journeys),
        peak_distress=peak_distress,
        note=f"Captured live on the {channel} channel.",
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
