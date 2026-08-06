"""Streamlit UI for the Vulnerable Customer Care Agent.

Handler view: play a conversation turn by turn, see discreet advisory prompts as
they fire, record the outcome for each, then view the portfolio report and
verify the evidence chain.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vca.config import load_config  # noqa: E402
from vca.ingestion.simulated import SimulatedSource  # noqa: E402
from vca.pipeline import VCAPipeline  # noqa: E402
from vca.reporting.metrics import build_report  # noqa: E402
from vca.schemas import HandlerAction, Outcome, Speaker  # noqa: E402

st.set_page_config(page_title="Vulnerable Customer Care Agent", page_icon="🛟", layout="wide")


@st.cache_resource
def get_pipeline():
    cfg = load_config()
    return cfg, VCAPipeline.from_config(cfg)


cfg, pipeline = get_pipeline()

st.title("🛟 Vulnerable Customer Care Agent")
st.caption(
    "Advisory-only monitoring for regulated service conversations. "
    f"Classifier: **{type(pipeline.classifier).__name__}** · "
    f"Retriever: **{type(pipeline.advisor.retriever).__name__}**"
)

tab_live, tab_report = st.tabs(["Live conversation", "Portfolio report"])

with tab_live:
    transcripts = sorted(cfg.path("data", "transcripts").glob("*.json"))
    labels = {p.stem: p for p in transcripts}
    col_pick, col_run = st.columns([3, 1])
    choice = col_pick.selectbox("Conversation", list(labels))
    fresh = col_run.checkbox("Clear evidence first", value=False)

    if st.button("Run conversation", type="primary"):
        if fresh:
            ep = cfg.path(cfg.evidence["path"])
            if ep.exists():
                ep.unlink()

        source = SimulatedSource.from_file(labels[choice])
        # Handler is assumed to accept advisory guidance in this demo.
        events = list(
            pipeline.process(
                source,
                on_guidance=lambda g: Outcome(
                    action=HandlerAction.ACCEPTED, note="Acknowledged via UI."
                ),
            )
        )

        for e in events:
            u = e.utterance
            if u.speaker == Speaker.HANDLER:
                st.markdown(f"**Handler:** {u.text}")
            else:
                st.markdown(f"**Customer:** {u.text}")
            if e.guidance:
                drivers = ", ".join(d.value for d in e.guidance.detection.triggered_drivers)
                with st.container(border=True):
                    st.warning(f"Possible vulnerability signal — **{drivers}** (advisory only)")
                    for a in e.guidance.adaptations:
                        st.markdown(f"**{a.title}** `[{a.policy_reference}]`")
                        st.write(a.guidance)
                    st.caption("This is guidance, not an instruction. The handler decides and acts.")

with tab_report:
    st.subheader("Portfolio-level fair-treatment report")
    report = build_report(pipeline.evidence)
    d = report.as_dict()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Detections", d["total_detections"])
    c2.metric("Conversations flagged", d["conversations_flagged"])
    c3.metric("Acceptance rate", f"{d['acceptance_rate'] * 100:.0f}%")
    c4.metric("Evidence chain", "✅ Valid" if d["chain_valid"] else "❌ Broken")

    st.markdown("**Detections by driver**")
    by_driver = pd.DataFrame(
        {"driver": list(d["detections_by_driver"]), "count": list(d["detections_by_driver"].values())}
    ).set_index("driver")
    st.bar_chart(by_driver)

    st.markdown("**Handler outcomes**")
    st.write(d["outcomes"] or "No outcomes recorded yet.")

    if not d["chain_valid"]:
        st.error(f"Tamper detected at record {d['first_bad_record']}.")

    with st.expander("Raw evidence records"):
        records = pipeline.evidence.read_all()
        if records:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "conversation": r["conversation_id"],
                            "turn": r["turn_index"],
                            "drivers": ", ".join(
                                s["driver"] for s in r["detection"]["scores"] if s["triggered"]
                            ),
                            "adaptation": ", ".join(
                                a["policy_reference"] for a in r["guidance"]["adaptations"]
                            ),
                            "outcome": r["outcome"]["action"],
                            "record_hash": r["record_hash"][:12] + "…",
                        }
                        for r in records
                    ]
                ),
                use_container_width=True,
            )
        else:
            st.info("Run a conversation to populate the evidence log.")
