"""Themed Plotly charts for GuardianCX dashboards.

Design rules (kept deliberately small and consistent):
- Form follows the data's job: magnitude -> bar; composition/status -> bar with
  reserved status colors; identity -> categorical hues in a FIXED order.
- Categorical driver hues are assigned by entity in a fixed order, never cycled.
- Risk and approval use RESERVED status colors (good / warning / critical /
  muted) with text labels — never color alone.
- One axis; recessive grid; thin marks; direct value labels; no chart title
  (the page supplies the heading). Backgrounds/fonts come from Streamlit's theme
  (`st.plotly_chart(..., theme="streamlit")`), so charts read in light and dark.
"""
from __future__ import annotations

import plotly.graph_objects as go

# Categorical hues for the four drivers (fixed order, distinct hue families).
DRIVER_COLORS = {
    "health": "#2a9d8f",       # teal
    "life_events": "#4361a8",  # indigo
    "resilience": "#c77d2e",   # amber-orange
    "capability": "#7b5ea7",   # violet
}
# Reserved status colors.
RISK_COLORS = {"low": "#0e7c66", "medium": "#b7791f", "high": "#b23a48"}
OUTCOME_COLORS = {
    "approved": "#0e7c66", "rejected": "#b23a48",
    "pending": "#b7791f", "not_required": "#9aa4af",
}
# Journeys are ranked by the harm that follows from mishandling them, so the
# palette ranks with them: the four where a mistake causes real detriment take
# the warm end, routine servicing the muted end. Colour is never the only cue —
# the label always carries the meaning.
JOURNEY_COLORS = {
    "fraud_scam": "#b23a48",
    "bereavement_estate": "#9c4f6e",
    "gambling_harm": "#a05a2c",
    "arrears_collections": "#c77d2e",
    "affordability_shock": "#c79a2e",
    "forbearance_request": "#4361a8",
    "third_party_access": "#7b5ea7",
    "product_sale": "#2a9d8f",
    "complaint": "#5b6b7b",
    "general_servicing": "#9aa4af",
}
_ACCENT = "#0e7c66"
_GRID = "rgba(128,128,128,0.18)"

_BASE_LAYOUT = dict(
    margin=dict(l=8, r=12, t=8, b=8),
    height=270,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
    bargap=0.35,
)


def _hbar(labels, values, colors, value_suffix="") -> go.Figure:
    """Horizontal magnitude bar with direct value labels."""
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v}{value_suffix}" for v in values],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(**_BASE_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor=_GRID, zeroline=False,
                     showline=False, title_text="")
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False,
                     autorange="reversed")
    return fig


def driver_bar(counts: dict[str, int]) -> go.Figure:
    order = ["health", "life_events", "resilience", "capability"]
    labels = [d.replace("_", " ") for d in order]
    values = [counts.get(d, 0) for d in order]
    colors = [DRIVER_COLORS[d] for d in order]
    return _hbar(labels, values, colors)


def risk_bar(counts: dict[str, int]) -> go.Figure:
    order = ["high", "medium", "low"]  # most severe on top after reverse
    labels = [r.title() for r in order]
    values = [counts.get(r, 0) for r in order]
    colors = [RISK_COLORS[r] for r in order]
    return _hbar(labels, values, colors)


def outcome_bar(counts: dict[str, int]) -> go.Figure:
    order = [k for k in ("approved", "rejected", "pending", "not_required")
             if counts.get(k, 0) > 0] or ["not_required"]
    labels = [k.replace("_", " ").title() for k in order]
    values = [counts.get(k, 0) for k in order]
    colors = [OUTCOME_COLORS[k] for k in order]
    return _hbar(labels, values, colors)


def guardrail_bar(counts: dict[str, int]) -> go.Figure:
    """Single-hue magnitude bar of guardrail activations (times each flagged)."""
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k.replace("_", " ") for k, _ in items] or ["none"]
    values = [v for _, v in items] or [0]
    return _hbar(labels, values, [_ACCENT] * len(labels))


def score_gauge(value: float, title: str = "") -> go.Figure:
    """Compact bullet gauge for a 0–100% rate."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100], "visible": False},
            "bar": {"color": _ACCENT, "thickness": 0.7},
            "borderwidth": 0,
        },
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(margin=dict(l=8, r=8, t=8, b=0), height=140,
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def journey_bar(counts: dict[str, int]) -> go.Figure:
    """Turns by banking journey — where the caseload actually sits.

    Ordered by volume rather than by the taxonomy's order, because the question
    this chart answers is "what is this contact centre spending its day on?"
    """
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    labels = [k.replace("_", " ") for k, _ in ordered]
    values = [v for _, v in ordered]
    colors = [JOURNEY_COLORS.get(k, _ACCENT) for k, _ in ordered]
    fig = _hbar(labels, values, colors)
    fig.update_layout(height=max(270, 30 * len(labels) + 60))
    return fig


def distress_hist(values: list[float]) -> go.Figure:
    """Distribution of measured customer distress across assessed turns."""
    fig = go.Figure(go.Histogram(
        x=values, nbinsx=10, marker=dict(color=_ACCENT, line=dict(width=0)),
        hovertemplate="distress %{x}: %{y} turn(s)<extra></extra>",
    ))
    fig.update_layout(**_BASE_LAYOUT)
    fig.update_xaxes(range=[0, 1], showgrid=True, gridcolor=_GRID, zeroline=False,
                     title_text="distress")
    fig.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False, title_text="")
    return fig
