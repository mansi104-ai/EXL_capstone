"""Reusable Streamlit UI components and a formal enterprise theme for GuardianCX."""
from __future__ import annotations

from typing import Any

import streamlit as st

# Formal, muted enterprise palette.
NAVY = "#0b2e4f"
NAVY_2 = "#14496b"
ACCENT = "#0e7c66"     # teal-green
WARN = "#b7791f"       # amber
DANGER = "#b23a48"     # muted red
INK = "#1f2933"
MUTED = "#6b7280"

_RISK_COLOR = {"low": ACCENT, "medium": WARN, "high": DANGER}
_SEV_COLOR = {"info": ACCENT, "warn": WARN, "block": DANGER}
_STAGE_COLOR = {
    "conversation": NAVY_2, "vulnerability": "#5b6b7b", "policy": NAVY_2,
    "guidance": ACCENT, "compliance": "#5b6b7b", "supervisor": NAVY, "evidence": "#4a5568",
}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{ --gx-navy: {NAVY}; --gx-accent: {ACCENT}; }}
        html, body, [class*="css"] {{
            font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        .block-container {{ padding-top: 1.8rem; max-width: 1250px; }}
        section[data-testid="stSidebar"] {{ border-right: 1px solid rgba(0,0,0,.06); }}

        .gx-hero {{
            background: linear-gradient(120deg, {NAVY} 0%, {NAVY_2} 100%);
            color: #fff; padding: 1.25rem 1.5rem; border-radius: 6px;
            margin-bottom: 1.25rem; border-left: 4px solid {ACCENT};
        }}
        .gx-eyebrow {{ font-size: .68rem; letter-spacing: .16em; text-transform: uppercase;
                       opacity: .75; margin-bottom: .35rem; }}
        .gx-hero h1 {{ margin: 0; font-size: 1.4rem; font-weight: 600; letter-spacing: -.01em; }}
        .gx-hero p {{ margin: .35rem 0 0; opacity: .9; font-size: .92rem; max-width: 70ch; }}

        .gx-card {{
            background: var(--background-color, #fff);
            border: 1px solid rgba(15,23,42,.10); border-radius: 6px;
            padding: 1rem 1.1rem; height: 100%;
            box-shadow: 0 1px 2px rgba(15,23,42,.04);
        }}
        .gx-metric-label {{ font-size: .7rem; text-transform: uppercase; letter-spacing: .08em;
                            color: {MUTED}; margin-bottom: .3rem; }}
        .gx-metric-value {{ font-size: 1.75rem; font-weight: 700; line-height: 1.1; color: {INK}; }}
        .gx-metric-help {{ font-size: .74rem; color: {MUTED}; margin-top: .15rem; }}

        .gx-pill {{ display: inline-block; padding: .12rem .55rem; border-radius: 4px;
                    font-size: .7rem; font-weight: 600; letter-spacing: .03em;
                    text-transform: uppercase; color: #fff; }}
        .gx-tag {{ display:inline-block; padding:.05rem .45rem; border-radius:4px;
                   font-size:.66rem; font-weight:700; letter-spacing:.06em; text-transform:uppercase;
                   color:#fff; margin-right:.4rem; }}
        .gx-sublabel {{ font-size:.7rem; letter-spacing:.08em; text-transform:uppercase; color:{MUTED}; }}
        .gx-footer {{ margin-top: 2.5rem; padding-top: .8rem; border-top: 1px solid rgba(0,0,0,.08);
                      color: {MUTED}; font-size: .72rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    st.sidebar.markdown(
        f"""
        <div style="padding:.4rem .2rem 1rem;">
          <div style="font-size:1.15rem;font-weight:700;color:{NAVY};letter-spacing:-.01em;">
            Guardian<span style="color:{ACCENT}">CX</span></div>
          <div class="gx-sublabel">Vulnerable-customer assurance</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        '<div class="gx-footer">GuardianCX — internal decision-support tool. '
        'Advisory only; a human handler makes every customer-facing decision. '
        'All data shown is synthetic and for demonstration. Confidential.</div>',
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="gx-hero"><div class="gx-eyebrow">GuardianCX · Regulated Customer Experience</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: Any, help_text: str = "") -> None:
    st.markdown(
        f'<div class="gx-card"><div class="gx-metric-label">{label}</div>'
        f'<div class="gx-metric-value">{value}</div>'
        f'<div class="gx-metric-help">{help_text}</div></div>',
        unsafe_allow_html=True,
    )


def pill(text: str, color: str) -> str:
    return f'<span class="gx-pill" style="background:{color}">{text}</span>'


def risk_pill(level: str) -> str:
    return pill("risk: " + level, _RISK_COLOR.get(level, NAVY))


def guardrail_badges(results: list[dict]) -> None:
    for r in results:
        if r["passed"]:
            mark, color = "✓", ACCENT
        else:
            mark = "✕" if r["severity"] == "block" else "!"
            color = _SEV_COLOR.get(r["severity"], WARN)
        st.markdown(
            f'<span style="color:{color};font-weight:700">{mark}</span> '
            f'{pill(r["name"].replace("_", " "), color)} '
            f'<span style="font-size:.82rem;color:{MUTED}">{r["detail"]}</span>',
            unsafe_allow_html=True,
        )


def agent_trace(trace: list[dict]) -> None:
    for step in trace:
        color = _STAGE_COLOR.get(step["agent"], NAVY)
        st.markdown(
            f'<span class="gx-tag" style="background:{color}">{step["agent"]}</span>'
            f'<span style="font-size:.9rem">{step["summary"]}</span>',
            unsafe_allow_html=True,
        )
