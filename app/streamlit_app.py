"""Vulnerable Customer Care Agent — MVP handler console.

A transparent, advisory-only agent for regulated service conversations. Handlers
can converse four ways — type a turn, speak (live Azure Speech), replay a sample,
or upload a transcript — and see the agent's full decision trace, the firm's
prescribed adaptation (via RAG), and record an auditable outcome for every
detection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vca.agent import DRIVER_HELP, CareAgent, Step  # noqa: E402
from vca.config import load_config  # noqa: E402
from vca.ingestion.factory import azure_status  # noqa: E402
from vca.ingestion.loaders import load_conversation  # noqa: E402
from vca.pipeline import VCAPipeline  # noqa: E402
from vca.reporting.metrics import build_report  # noqa: E402
from vca.schemas import Driver, HandlerAction, Outcome, Speaker, Utterance  # noqa: E402

st.set_page_config(page_title="Vulnerable Customer Care Agent", page_icon="🛟", layout="wide")

STEP_ICON = {
    Step.PERCEIVE: "👂", Step.ASSESS: "🧭", Step.RETRIEVE: "📚",
    Step.ADVISE: "💬", Step.DECIDE: "🧑", Step.RECORD: "🔏",
}
DRIVER_EMOJI = {
    Driver.HEALTH: "🩺", Driver.LIFE_EVENTS: "🕊️",
    Driver.RESILIENCE: "💷", Driver.CAPABILITY: "🧩",
}


# --------------------------------------------------------------------------- #
# Resources & session state
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_agent():
    cfg = load_config()
    pipeline = VCAPipeline.from_config(cfg)
    return cfg, pipeline, CareAgent(pipeline)


cfg, pipeline, agent = get_agent()


def _init_state():
    ss = st.session_state
    ss.setdefault("turns", [])          # list of turn dicts (see add_turn)
    ss.setdefault("next_id", 0)
    ss.setdefault("conversation_id", "LIVE-001")
    ss.setdefault("live", None)         # LiveAzureRecognizer


_init_state()


def add_turn(speaker: str, text: str):
    """Run the agent on a new turn and append it to the conversation."""
    text = text.strip()
    if not text:
        return
    ss = st.session_state
    utt = Utterance(
        conversation_id=ss.conversation_id,
        turn_index=ss.next_id,
        speaker=Speaker(speaker),
        text=text,
    )
    trace = agent.run_turn(utt)
    ss.turns.append({"id": ss.next_id, "speaker": speaker, "text": text,
                     "trace": trace, "outcome": None, "note": ""})
    ss.next_id += 1


def load_turns(turns: list[dict]):
    for t in turns:
        add_turn(t["speaker"], t["text"])


def reset_conversation():
    st.session_state.turns = []
    st.session_state.next_id = 0


def record_decision(turn: dict, action: HandlerAction, note: str = ""):
    trace = turn["trace"]
    ev = trace.event
    outcome = Outcome(action=action, note=note or None)
    pipeline.record_outcome(ev.detection, ev.guidance, outcome)
    turn["outcome"] = action.value
    turn["note"] = note


# --------------------------------------------------------------------------- #
# Sidebar — transparency: engines, credentials, config, skills
# --------------------------------------------------------------------------- #
def render_sidebar():
    az = azure_status(cfg)
    with st.sidebar:
        st.header("🛟 VCA console")
        st.caption("Advisory-only. The agent never acts — the handler decides.")

        st.subheader("Engine status")
        clf = type(pipeline.classifier).__name__
        rtr = type(pipeline.advisor.retriever).__name__
        clf_live = clf == "TransformerClassifier"
        rtr_live = rtr == "EmbeddingRetriever"
        st.markdown(
            f"- Classifier: {'🟢' if clf_live else '🟡'} `{clf}`"
            + ("" if clf_live else "  \n  <sub>fine-tune to upgrade</sub>"),
            unsafe_allow_html=True,
        )
        st.markdown(
            f"- RAG retriever: {'🟢' if rtr_live else '🟡'} `{rtr}`",
        )
        st.markdown(
            f"- Azure Speech: {'🟢 ready' if az['available'] else '🟡 fallback'}"
        )
        if not az["available"]:
            st.caption(az["reason"])

        st.subheader("Config")
        st.caption(
            f"Drivers: {', '.join(cfg.drivers)}  \n"
            f"Threshold: {cfg.classifier.get('threshold')} · RAG top-k: {cfg.rag.get('top_k')}"
        )

        with st.expander("Agentic-CX skills showcased"):
            st.markdown(
                "- **Agent loop & orchestration** — perceive→assess→retrieve→"
                "advise→decide→record (see each turn's trace)\n"
                "- **Tool use** — classifier + RAG retriever as agent tools\n"
                "- **RAG** — grounding guidance in the firm's policy\n"
                "- **Guardrails** — advisory-only, human-in-the-loop\n"
                "- **Observability** — full decision trace + confidences\n"
                "- **Evaluation** — per-driver precision/recall (`scripts/evaluate.py`)\n"
                "- **Auditability** — hash-chained immutable evidence\n"
                "- **Multi-modal input** — text, file, live speech"
            )

        st.subheader("Evidence")
        ok, bad = pipeline.evidence.verify_chain()
        st.markdown(f"Chain: {'🟢 valid' if ok else '🔴 broken at ' + str(bad)}")
        st.caption(f"{len(pipeline.evidence.read_all())} record(s) logged")


# --------------------------------------------------------------------------- #
# Rendering a turn + its agent trace
# --------------------------------------------------------------------------- #
def render_driver_scores(detection):
    for s in detection.scores:
        d = s.driver
        c1, c2, c3 = st.columns([3, 6, 2])
        c1.markdown(f"{DRIVER_EMOJI[d]} **{d.value}**", help=DRIVER_HELP[d])
        c2.progress(min(max(s.score, 0.0), 1.0))
        c3.markdown(f"{'🚩' if s.triggered else '·'} {s.score:.2f}")


def render_trace(trace):
    for step in trace.steps:
        with st.container():
            st.markdown(f"{STEP_ICON.get(step.step, '•')} **{step.step.value.upper()}** — {step.summary}")
            if step.step == Step.ASSESS and step.detail.get("scores"):
                render_driver_scores(trace.event.detection)
            elif step.step == Step.RETRIEVE:
                for a in step.detail.get("adaptations", []):
                    st.caption(f"`{a['policy_reference']}` {a['title']} · score {a['retrieval_score']}")


def render_guidance_card(turn):
    guidance = turn["trace"].event.guidance
    with st.container(border=True):
        drivers = ", ".join(d.value for d in guidance.detection.triggered_drivers)
        st.warning(f"**Advisory signal — {drivers}** · guidance only, you decide")
        for a in guidance.adaptations:
            st.markdown(f"**{a.title}** `[{a.policy_reference}]`")
            st.write(a.guidance)

        if turn["outcome"] is None:
            st.markdown("**Record your action** (written to the evidence log):")
            b1, b2, b3 = st.columns(3)
            if b1.button("✅ Accept", key=f"acc{turn['id']}"):
                record_decision(turn, HandlerAction.ACCEPTED)
                st.rerun()
            if b2.button("🚫 Dismiss", key=f"dis{turn['id']}"):
                record_decision(turn, HandlerAction.DISMISSED)
                st.rerun()
            note = b3.text_input("Modify", key=f"note{turn['id']}",
                                 placeholder="adapted action…", label_visibility="collapsed")
            if b3.button("✏️ Save modified", key=f"mod{turn['id']}"):
                record_decision(turn, HandlerAction.MODIFIED, note)
                st.rerun()
        else:
            st.success(f"Recorded outcome: **{turn['outcome']}**"
                       + (f" — {turn['note']}" if turn["note"] else ""))


def render_conversation():
    for turn in st.session_state.turns:
        role = "user" if turn["speaker"] == Speaker.CUSTOMER.value else "assistant"
        avatar = "🧑" if role == "user" else "🎧"
        with st.chat_message(role, avatar=avatar):
            st.markdown(f"**{turn['speaker'].title()}:** {turn['text']}")
            trace = turn["trace"]
            if turn["speaker"] == Speaker.CUSTOMER.value and trace.event.assessed:
                flagged = trace.flagged
                label = "🚩 Vulnerability signal — see agent trace" if flagged else "🔍 Agent trace (no signal)"
                with st.expander(label, expanded=flagged):
                    render_trace(trace)
                if flagged:
                    render_guidance_card(turn)


# --------------------------------------------------------------------------- #
# Input modes
# --------------------------------------------------------------------------- #
def input_type_a_turn():
    with st.form("type_turn", clear_on_submit=True):
        c1, c2 = st.columns([1, 4])
        speaker = c1.selectbox("Speaker", ["customer", "handler"], label_visibility="collapsed")
        text = c2.text_input("Turn", placeholder="Type what was said…",
                             label_visibility="collapsed")
        if st.form_submit_button("Add turn", type="primary") and text.strip():
            add_turn(speaker, text)
            st.rerun()


def input_live_azure():
    az = azure_status(cfg)
    if not az["available"]:
        st.info(f"Live microphone needs Azure Speech. {az['reason']}")
        st.caption("Configure `.env` (AZURE_SPEECH_KEY / AZURE_SPEECH_REGION), install "
                   "`requirements-full.txt`, and unset `VCA_FORCE_SIMULATED`, then run locally.")
        return

    from vca.ingestion.azure_speech import AzureSpeechSource, LiveAzureRecognizer  # noqa: E402

    st.caption(f"Azure region: `{az['region']}` · speaking is attributed to the customer.")
    c1, c2, c3 = st.columns(3)

    if c1.button("🎙️ Capture one phrase", type="primary"):
        src = AzureSpeechSource("LIVE", cfg.azure_speech_key, cfg.azure_speech_region)
        try:
            with st.spinner("Listening…"):
                text = src.recognize_once()
            if text:
                add_turn("customer", text)
                st.rerun()
            else:
                st.warning("No speech recognised.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Azure Speech error: {exc}")

    ss = st.session_state
    if c2.button("▶️ Start continuous"):
        ss.live = LiveAzureRecognizer(cfg.azure_speech_key, cfg.azure_speech_region)
        try:
            ss.live.start()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not start: {exc}")
            ss.live = None
    if c3.button("⏹️ Stop") and ss.live:
        ss.live.stop()
        ss.live = None

    if ss.live and ss.live.running:
        st.success("Listening continuously…")
        if st.button("🔄 Fetch new speech"):
            for phrase in ss.live.drain():
                add_turn("customer", phrase)
            st.rerun()


def input_sample():
    sample_dir = cfg.path("data", "transcripts")
    samples = {p.stem: p for p in sorted(sample_dir.glob("*.json"))}
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox("Sample conversation", list(samples))
    if c2.button("Load sample", type="primary"):
        reset_conversation()
        data = load_conversation(samples[choice].read_text(encoding="utf-8"), "s.json")
        st.session_state.conversation_id = choice
        load_turns(data["turns"])
        st.rerun()


def input_upload():
    up = st.file_uploader("Upload a transcript", type=["json", "csv", "txt"])
    if up is not None and st.button("Process upload", type="primary"):
        reset_conversation()
        content = up.read().decode("utf-8", errors="replace")
        data = load_conversation(content, up.name)
        st.session_state.conversation_id = f"UPLOAD-{up.name}"
        load_turns(data["turns"])
        st.success(f"Loaded {len(data['turns'])} turns from {up.name}.")
        st.rerun()


# --------------------------------------------------------------------------- #
# Report tab
# --------------------------------------------------------------------------- #
def render_report():
    st.subheader("Portfolio-level fair-treatment report")
    st.caption("Aggregated from the immutable evidence log across all conversations.")
    d = build_report(pipeline.evidence).as_dict()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detections", d["total_detections"])
    c2.metric("Conversations flagged", d["conversations_flagged"])
    c3.metric("Acceptance rate", f"{d['acceptance_rate'] * 100:.0f}%")
    c4.metric("Evidence chain", "✅ Valid" if d["chain_valid"] else "❌ Broken")

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Detections by driver**")
        st.bar_chart(pd.DataFrame(
            {"count": d["detections_by_driver"]}
        ))
    with cc2:
        st.markdown("**Handler outcomes**")
        if d["outcomes"]:
            st.bar_chart(pd.DataFrame({"count": d["outcomes"]}))
        else:
            st.info("No outcomes recorded yet.")

    if not d["chain_valid"]:
        st.error(f"Tamper detected at record {d['first_bad_record']}.")

    with st.expander("Raw evidence records (hash-chained)"):
        records = pipeline.evidence.read_all()
        if records:
            st.dataframe(pd.DataFrame([
                {
                    "conversation": r["conversation_id"],
                    "turn": r["turn_index"],
                    "drivers": ", ".join(s["driver"] for s in r["detection"]["scores"] if s["triggered"]),
                    "adaptation": ", ".join(a["policy_reference"] for a in r["guidance"]["adaptations"]),
                    "outcome": r["outcome"]["action"],
                    "prev_hash": r["prev_hash"][:10] + "…",
                    "record_hash": r["record_hash"][:10] + "…",
                }
                for r in records
            ]), width="stretch")
        else:
            st.info("Record a decision on a flagged turn to populate the log.")

    if st.button("🗑️ Clear evidence log"):
        p = cfg.path(cfg.evidence["path"])
        if p.exists():
            p.unlink()
        st.rerun()


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
render_sidebar()
st.title("🛟 Vulnerable Customer Care Agent")
st.caption("Monitors regulated service conversations for vulnerability across four "
           "regulator-defined drivers, surfaces the firm's prescribed adaptation, and "
           "records an auditable evidence trail — advisory only.")

tab_converse, tab_report, tab_about = st.tabs(["💬 Converse", "📊 Evidence & report", "ℹ️ About"])

with tab_converse:
    left, right = st.columns([2, 3])
    with left:
        st.subheader("Input")
        mode = st.radio("Input mode", ["Type a turn", "Live microphone (Azure)",
                                       "Replay sample", "Upload transcript"],
                        label_visibility="collapsed")
        if mode == "Type a turn":
            input_type_a_turn()
        elif mode == "Live microphone (Azure)":
            input_live_azure()
        elif mode == "Replay sample":
            input_sample()
        else:
            input_upload()
        st.divider()
        if st.button("♻️ New conversation"):
            reset_conversation()
            st.rerun()
        st.caption(f"Conversation: `{st.session_state.conversation_id}` · "
                   f"{len(st.session_state.turns)} turns")
    with right:
        st.subheader("Conversation & agent trace")
        if st.session_state.turns:
            render_conversation()
        else:
            st.info("Add a turn, replay a sample, upload a transcript, or speak to begin.")

with tab_report:
    render_report()

with tab_about:
    st.markdown((ROOT / "docs" / "DESIGN.md").read_text(encoding="utf-8"))
