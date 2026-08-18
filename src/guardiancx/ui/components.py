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
    "conversation": NAVY_2, "financial_context": "#3d5a80", "sentiment": "#6d597a",
    "vulnerability": "#5b6b7b", "policy": NAVY_2, "guidance": ACCENT,
    "compliance": "#5b6b7b", "supervisor": NAVY, "evidence": "#4a5568",
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

        /* --- live signal rail --- */
        .gx-meter-row {{ margin: .45rem 0 .6rem; }}
        .gx-meter-head {{ display:flex; justify-content:space-between; font-size:.72rem;
                          color:{MUTED}; letter-spacing:.04em; margin-bottom:.2rem; }}
        .gx-meter-head b {{ color:{INK}; font-variant-numeric: tabular-nums; }}
        .gx-meter {{ height:6px; border-radius:3px; background:rgba(15,23,42,.09); overflow:hidden; }}
        .gx-meter > i {{ display:block; height:100%; border-radius:3px; }}
        .gx-chip {{ display:inline-block; padding:.12rem .5rem; margin:0 .3rem .3rem 0;
                    border-radius:999px; font-size:.7rem; font-weight:600;
                    border:1px solid rgba(15,23,42,.18); color:{INK}; }}
        .gx-chip.on {{ background:{WARN}; border-color:{WARN}; color:#fff; }}
        .gx-chip.alert {{ background:{DANGER}; border-color:{DANGER}; color:#fff; }}
        .gx-chip.ok {{ background:{ACCENT}; border-color:{ACCENT}; color:#fff; }}
        .gx-rail-title {{ font-size:.7rem; letter-spacing:.1em; text-transform:uppercase;
                          color:{MUTED}; margin:.9rem 0 .35rem; font-weight:700; }}
        .gx-quiet {{ font-size:.78rem; color:{MUTED}; }}
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


# --------------------------------------------------------------------------- #
# Live signal rail
#
# The rail is what a handler actually watches during a call, so every element in
# it answers "what should I do differently right now?" — a number with no action
# attached does not earn its place.
# --------------------------------------------------------------------------- #
def meter(label: str, value: float, color: str = ACCENT, caption: str = "",
          fmt: str = "{:.2f}") -> None:
    """A labelled 0-1 bar."""
    pct = max(0.0, min(1.0, float(value))) * 100
    st.markdown(
        f'<div class="gx-meter-row">'
        f'<div class="gx-meter-head"><span>{label}</span><b>{fmt.format(value)}</b></div>'
        f'<div class="gx-meter"><i style="width:{pct:.1f}%;background:{color}"></i></div>'
        + (f'<div class="gx-metric-help">{caption}</div>' if caption else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def chips(items: list[str], tone: str = "") -> None:
    """A row of small labels — drivers, stress indicators, redaction categories."""
    if not items:
        st.markdown('<span class="gx-quiet">none</span>', unsafe_allow_html=True)
        return
    cls = f"gx-chip {tone}".strip()
    st.markdown(
        "".join(f'<span class="{cls}">{item}</span>' for item in items),
        unsafe_allow_html=True,
    )


def rail_title(text: str) -> None:
    st.markdown(f'<div class="gx-rail-title">{text}</div>', unsafe_allow_html=True)


def quiet(text: str) -> None:
    st.markdown(f'<span class="gx-quiet">{text}</span>', unsafe_allow_html=True)


def risk_color(level: str) -> str:
    return _RISK_COLOR.get(level, NAVY)


def scale_color(value: float) -> str:
    """Green below a third, amber to two thirds, red above — the same reading
    everywhere a 0-1 indicator appears."""
    if value >= 0.66:
        return DANGER
    if value >= 0.33:
        return WARN
    return ACCENT


def recommendation_card(decision) -> None:
    """The advisory guidance block — risk, approval state, adaptations, citations.

    Shared by the live monitor and the guidance panel so a recommendation looks
    and reads identically wherever a handler meets it.
    """
    rec = decision.recommendation
    if rec is None:
        return
    pending = decision.approval_status.value == "pending"
    with st.container(border=True):
        st.markdown(
            f"{risk_pill(decision.risk_level.value)} "
            f"{pill('approval: ' + decision.approval_status.value, WARN if pending else ACCENT)} "
            f"&nbsp; <b>advisory guidance</b>",
            unsafe_allow_html=True,
        )
        st.write(rec.summary)
        for adaptation in rec.adaptations:
            st.markdown(f"- {adaptation}")
        st.caption("Citations: " + ", ".join(rec.citations) +
                   f" · confidence {rec.confidence:.2f} · source {rec.source}")
