"""GuardianCX — Agentic Customer Experience for regulated financial services.

Streamlit multipage entry point. Bootstraps the database + policy index, seeds
synthetic conversations on first run, and wires the 11 pages via st.navigation.

Run:  streamlit run app.py   (from the guardiancx/ directory)
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make `config` and the `guardiancx` package importable regardless of CWD.
ROOT = Path(__file__).resolve().parent
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st  # noqa: E402

from guardiancx.database.db import init_engine  # noqa: E402
from guardiancx.database.repository import list_evidence  # noqa: E402
from guardiancx.rag.vector_store import ensure_ingested  # noqa: E402
from guardiancx.services.synthetic_data import list_conversations  # noqa: E402
from guardiancx.ui import components as C  # noqa: E402
from guardiancx.ui import views  # noqa: E402

st.set_page_config(page_title="GuardianCX", page_icon="🛡️", layout="wide")
C.inject_css()


@st.cache_resource
def bootstrap() -> dict:
    """One-time setup: DB, policy index, and seed data if the store is empty."""
    init_engine()
    chunks = ensure_ingested()
    if not list_evidence():
        from guardiancx.agents.graph import process_conversation

        for conv in list_conversations():
            process_conversation(conv)
    return {"chunks": chunks, "records": len(list_evidence())}


bootstrap()

PAGES = [
    st.Page(views.exec_dashboard, title="Executive Dashboard", icon="📊", default=True),
    st.Page(views.live_monitor, title="Live Conversation Monitor", icon="🎧"),
    st.Page(views.detection, title="Vulnerability Detection", icon="🧭"),
    st.Page(views.guidance_panel, title="AI Guidance Panel", icon="💬"),
    st.Page(views.approval_queue, title="Human Approval Queue", icon="🧑‍⚖️"),
    st.Page(views.policy_kb, title="Policy Knowledge Base", icon="📚"),
    st.Page(views.guardrails_dashboard, title="Guardrails Dashboard", icon="🛡️"),
    st.Page(views.audit_trail, title="Audit Trail", icon="🔏"),
    st.Page(views.customer_timeline_page, title="Customer Timeline", icon="👤"),
    st.Page(views.analytics, title="Analytics & Evaluation", icon="📈"),
    st.Page(views.settings_page, title="Settings", icon="⚙️"),
]

st.navigation(PAGES).run()
