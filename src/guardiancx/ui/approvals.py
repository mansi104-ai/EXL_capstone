"""The Human Approval Queue.

This page is where the system's central claim is either true or a slogan.
GuardianCX recommends; a person decides. If deciding is a pair of buttons under
a one-line summary, the person is a rubber stamp and the claim is false — they
have no way to tell a routine adaptation from a bereavement crisis, and no
option except yes or no on wording they had no part in.

So the queue is built around three things a real reviewer needs.

**Why is this in front of me?** Every item leads with the Supervisor's routing
reasons — high risk, a guardrail block, low confidence, customer distress,
financial detriment already occurring. Those call for different attention, and a
queue that shows only "high" flattens them.

**What did the customer actually say?** The reviewer sees the masked utterance,
the journey, the obligations it engages, and the customer's measured state. A
recommendation read without its context is just plausible text.

**Say it like this instead.** The third action, and in practice the most common
one: approve with an amendment. A supervisor rarely thinks the advice is wrong —
they think it is nearly right and would put it differently. Forcing that into a
binary loses the most useful signal the firm has about its own model. The
original is never overwritten; both versions live in the record, because "what
did the model propose" and "what did we send" are different questions.

**Ordering is by harm, not arrival.** A bereaved customer waiting twenty minutes
outranks a routine affordability case that arrived first. Queues sorted by
timestamp systematically serve the least vulnerable customers first, because
their cases are quicker to clear.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from ..database.repository import list_evidence, list_pending, update_approval
from ..finance.taxonomy import JOURNEY_LABELS, REGULATORY_MAP, Journey
from . import components as C

# How the queue is ordered. Higher sorts first.
_RISK_WEIGHT = {"high": 3, "medium": 2, "low": 1}

# Routing reasons that mean a person is being harmed now, rather than being at
# risk of it. These jump the queue whatever the risk label says.
_URGENT_REASONS = {"customer distress", "acute financial detriment", "guardrail block"}


def _age_minutes(created_at: str) -> float:
    if not created_at:
        return 0.0
    try:
        stamp = datetime.fromisoformat(created_at)
    except ValueError:
        return 0.0
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return max(0.0, (datetime.now(timezone.utc) - stamp).total_seconds() / 60.0)


def _priority(item: dict[str, Any]) -> tuple:
    """Sort key: urgency first, then risk, then how long they have waited."""
    reasons = set(item.get("routing_reasons") or [])
    urgent = bool(reasons & _URGENT_REASONS)
    distress = float(item.get("sentiment_distress") or 0.0)
    return (
        urgent,
        _RISK_WEIGHT.get(item.get("risk_level", "low"), 0),
        distress,
        _age_minutes(item.get("created_at", "")),
    )


def approval_queue() -> None:
    C.hero(
        "Human Approval Queue",
        "Every recommendation the system would not use without a person's say-so. "
        "You can approve it, change the wording, or reject it — nothing reaches a "
        "customer until you do.",
    )

    pending = sorted(list_pending(), key=_priority, reverse=True)
    _queue_summary(pending)

    if not pending:
        st.success("Nothing is waiting for review.")
        _recent_decisions()
        return

    left, right = st.columns([3, 1])
    with right:
        reviewer = st.text_input("Reviewer", "supervisor",
                                 help="Recorded against every decision you make.")
        only_urgent = st.toggle("Urgent only", value=False,
                                help="Distress, detriment, or a blocked guardrail.")
    with left:
        st.caption("Ordered by harm, not by arrival — a distressed or bereaved "
                   "customer is reviewed before a routine case that queued earlier.")

    shown = [i for i in pending
             if not only_urgent or (set(i.get("routing_reasons") or []) & _URGENT_REASONS)]
    if not shown:
        st.info("No urgent items. Turn off the filter to see the rest of the queue.")
        return

    for item in shown:
        _review_card(item, reviewer)

    _recent_decisions()


def _queue_summary(pending: list[dict[str, Any]]) -> None:
    urgent = [i for i in pending
              if set(i.get("routing_reasons") or []) & _URGENT_REASONS]
    oldest = max((_age_minutes(i.get("created_at", "")) for i in pending), default=0.0)
    high = sum(1 for i in pending if i.get("risk_level") == "high")

    row = st.columns(4)
    with row[0]:
        C.metric_card("Awaiting review", len(pending), "nothing has reached a customer")
    with row[1]:
        C.metric_card("Urgent", len(urgent), "distress, detriment or a blocked guardrail")
    with row[2]:
        C.metric_card("High risk", high, "acute or sensitive circumstances")
    with row[3]:
        C.metric_card("Longest wait", f"{oldest:.0f} min" if oldest else "—",
                      "how long the customer has been waiting")
    if urgent:
        st.warning(f"{len(urgent)} case(s) involve a customer who is being harmed now, "
                   "not merely at risk. They are at the top of the queue.")


def _review_card(item: dict[str, Any], reviewer: str) -> None:
    record_id = item["record_id"]
    reasons = item.get("routing_reasons") or []
    urgent = bool(set(reasons) & _URGENT_REASONS)

    with st.container(border=True):
        # --- why this is here ------------------------------------------------
        head = (
            f"{C.risk_pill(item['risk_level'])} "
            f"{C.pill(item.get('channel', 'chat'), C.NAVY_2)} "
            f"&nbsp; <b>{item['conversation_id']}</b> · customer {item['customer_id']}"
        )
        st.markdown(head, unsafe_allow_html=True)
        if reasons:
            st.markdown("**Routed for review because:** " + ", ".join(reasons))
            C.chips(reasons, tone="alert" if urgent else "on")
        waited = _age_minutes(item.get("created_at", ""))
        st.caption(f"Waiting {waited:.0f} min · detected by {item.get('detection_source', '—')}")

        # --- the situation ---------------------------------------------------
        if item.get("masked_text"):
            st.markdown("**The customer said**")
            st.info(item["masked_text"])

        cols = st.columns(3)
        journey = item.get("journey") or ""
        with cols[0]:
            label = JOURNEY_LABELS.get(Journey(journey), journey) if journey in \
                Journey._value2member_map_ else "—"
            st.markdown(f"**Journey**  \n{label}")
        with cols[1]:
            drivers = item.get("triggered_drivers") or []
            st.markdown("**Drivers**  \n" + (", ".join(drivers) if drivers else "none"))
        with cols[2]:
            distress = float(item.get("sentiment_distress") or 0.0)
            emotion = item.get("sentiment_emotion") or "—"
            st.markdown(f"**Customer state**  \n{emotion} · distress {distress:.2f}")

        if journey in Journey._value2member_map_:
            obligations = REGULATORY_MAP.get(Journey(journey), [])
            if obligations:
                with st.expander("Obligations engaged"):
                    for obligation in obligations:
                        st.markdown(f"- {obligation}")

        # --- what the system proposed ---------------------------------------
        st.markdown("**Proposed guidance**")
        st.write(item.get("recommendation") or "—")
        st.caption(
            f"Citations: {', '.join(item.get('citations') or []) or '—'} · "
            f"confidence {float(item.get('model_confidence') or 0):.2f}"
        )

        flagged = [g for g in (item.get("guardrails") or []) if not g.get("passed", True)]
        if flagged:
            with st.expander(f"{len(flagged)} guardrail flag(s)", expanded=urgent):
                C.guardrail_badges(flagged)

        # --- the decision ----------------------------------------------------
        st.markdown("**Your decision**")
        amended = st.text_area(
            "Amend the wording (optional)",
            value="",
            key=f"amend_{record_id}",
            placeholder="Leave empty to approve as proposed. Anything you write here is "
                        "recorded as the version that was used.",
            height=80,
        )
        note = st.text_input("Note", key=f"note_{record_id}",
                             placeholder="Why you decided this — kept in the audit trail.")

        c1, c2, c3, _ = st.columns([1, 1.3, 1, 3])
        if c1.button("Approve", key=f"ap_{record_id}", type="primary"):
            update_approval(record_id, "approved", reviewer, note)
            st.rerun()
        if c2.button("Approve with changes", key=f"am_{record_id}",
                     disabled=not amended.strip()):
            update_approval(record_id, "approved", reviewer,
                            note or "Approved with amended wording.", amended.strip())
            st.rerun()
        if c3.button("Reject", key=f"rj_{record_id}"):
            update_approval(record_id, "rejected", reviewer, note)
            st.rerun()


def _recent_decisions() -> None:
    """What reviewers have been doing — and where they disagreed with the model.

    The amendment rate is the most honest quality signal the firm has: it is the
    rate at which a trained human looked at the system's wording and decided to
    change it.
    """
    st.divider()
    st.markdown("**Recent decisions**")
    rows = [r for r in list_evidence()
            if r.get("approval_status") in ("approved", "rejected")]
    if not rows:
        st.caption("No decisions recorded yet.")
        return

    amended = [r for r in rows if r.get("amended_recommendation")]
    approved = [r for r in rows if r["approval_status"] == "approved"]
    cols = st.columns(3)
    with cols[0]:
        C.metric_card("Decided", len(rows), "reviewed by a person")
    with cols[1]:
        C.metric_card("Approved", f"{len(approved) / len(rows) * 100:.0f}%",
                      "of decided recommendations")
    with cols[2]:
        C.metric_card("Amended", f"{len(amended) / len(rows) * 100:.0f}%",
                      "reviewer changed the wording")

    if amended:
        with st.expander("Where reviewers changed the wording", expanded=False):
            st.caption("Each row is a case where a person judged the system's phrasing "
                       "needed work. This is the queue's most useful output.")
            for row in amended[:10]:
                st.markdown(f"**{row['conversation_id']}** — {row.get('decided_by', '')}")
                st.caption(f"Proposed: {row.get('recommendation', '')}")
                st.success(f"Sent: {row['amended_recommendation']}")

    table = pd.DataFrame([
        {
            "conversation": r["conversation_id"],
            "risk": r["risk_level"],
            "decision": r["approval_status"],
            "amended": "yes" if r.get("amended_recommendation") else "",
            "by": r.get("decided_by", ""),
            "note": r.get("decision_note", ""),
        }
        for r in rows[:25]
    ])
    st.dataframe(table, width="stretch", hide_index=True)
