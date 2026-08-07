"""Reusable Streamlit UI components + enterprise styling for GuardianCX."""
from __future__ import annotations

from typing import Any

import streamlit as st

BRAND = "#1f4e79"     # deep enterprise blue
ACCENT = "#0a8f6b"    # green
WARN = "#c98a00"
DANGER = "#c0392b"

_RISK_COLOR = {"low": ACCENT, "medium": WARN, "high": DANGER}
_SEV_COLOR = {"info": ACCENT, "warn": WARN, "block": DANGER}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 2rem; max-width: 1250px; }}
        .gx-hero {{
            background: linear-gradient(120deg, {BRAND} 0%, #143a5c 100%);
            color: #fff; padding: 1.1rem 1.4rem; border-radius: 12px; margin-bottom: 1rem;
        }}
        .gx-hero h1 {{ margin: 0; font-size: 1.35rem; }}
        .gx-hero p {{ margin: .25rem 0 0; opacity: .85; font-size: .9rem; }}
        .gx-card {{
            background: var(--background-color, #fff);
            border: 1px solid rgba(128,128,128,.2); border-radius: 12px;
            padding: 1rem 1.1rem; height: 100%;
        }}
        .gx-metric-label {{ font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; opacity: .7; }}
        .gx-metric-value {{ font-size: 1.7rem; font-weight: 700; line-height: 1.1; }}
        .gx-pill {{ display: inline-block; padding: .12rem .6rem; border-radius: 999px;
                    font-size: .75rem; font-weight: 600; color: #fff; }}
        .gx-chip {{ display:inline-block; padding:.1rem .5rem; border-radius:6px;
                    background:rgba(31,78,121,.12); font-size:.75rem; margin:.1rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="gx-hero"><h1>🛡️ {title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: Any, help_text: str = "") -> None:
    st.markdown(
        f'<div class="gx-card"><div class="gx-metric-label">{label}</div>'
        f'<div class="gx-metric-value">{value}</div>'
        f'<div style="font-size:.78rem;opacity:.6">{help_text}</div></div>',
        unsafe_allow_html=True,
    )


def pill(text: str, color: str) -> str:
    return f'<span class="gx-pill" style="background:{color}">{text}</span>'


def risk_pill(level: str) -> str:
    return pill(level.upper(), _RISK_COLOR.get(level, BRAND))


def severity_pill(sev: str) -> str:
    return pill(sev.upper(), _SEV_COLOR.get(sev, BRAND))


def guardrail_badges(results: list[dict]) -> None:
    """Render guardrail results as colored badges."""
    for r in results:
        color = _SEV_COLOR["info"] if r["passed"] else _SEV_COLOR.get(r["severity"], WARN)
        icon = "✅" if r["passed"] else ("⛔" if r["severity"] == "block" else "⚠️")
        st.markdown(
            f'{icon} {pill(r["name"], color)} '
            f'<span style="font-size:.82rem;opacity:.85">{r["detail"]}</span>',
            unsafe_allow_html=True,
        )


def agent_trace(trace: list[dict]) -> None:
    icons = {"conversation": "👂", "vulnerability": "🧭", "policy": "📚",
             "guidance": "💬", "compliance": "🛡️", "supervisor": "🧑‍⚖️", "evidence": "🔏"}
    for step in trace:
        st.markdown(f"{icons.get(step['agent'], '•')} **{step['agent'].title()}** — {step['summary']}")
