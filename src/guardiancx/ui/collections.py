"""The Collections Queue page — the outbound channel, and the calls it will not make.

Most queue consoles show the work outstanding. This one gives equal space to the
work being held back, because that is where the conduct evidence is. A Head of
Collections signing off this channel does not need reassurance that it dials; she
needs to see that at 21:00 it dials nothing, that the customer who asked to be
left alone has gone from the list, and that each of those is attributable to a
named rule rather than to a coincidence of timing.

So the page is built around three questions a reviewer actually asks:

* **What would you call right now, and under what strategy?**
* **What are you not calling, and which rule stopped it?** — the blocked list,
  with the gate and the retry time.
* **Who has told us to stop?** — the opt-out register, append-only, with the
  customer's own words.

The clock control at the top is not a debugging aid. Calling-hours behaviour is
invisible at any single moment: a reviewer looking at this page at 14:00 on a
Tuesday learns nothing about what happens on a Friday at midday. Being able to
move the clock is how the rule gets demonstrated rather than asserted.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta

import pandas as pd
import streamlit as st

from ..finance.accounts import get_customer
from ..outbound import optout, queue as q
from ..outbound.contact_rules import GST, describe_window, local_now, may_contact
from ..outbound.strategy import list_strategies
from ..utils import language as L
from . import components as C

_STATE_KEY = "collections_clock"


def _clock() -> datetime:
    """The moment the page is reasoning about — now, or a time the reviewer set."""
    override = st.session_state.get(_STATE_KEY)
    return override or local_now()


def _customer_label(customer_id: str) -> str:
    customer = get_customer(customer_id)
    return f"{customer.name} ({customer_id})" if customer else customer_id


def _language_cell(code: str) -> str:
    return L.display_name(code)


def _clock_controls() -> None:
    st.markdown("##### The clock")
    C.quiet("Calling-hours conduct cannot be demonstrated at a single moment. "
            "Move the clock to see the window open and close.")

    columns = st.columns([2, 2, 3, 3])
    with columns[0]:
        use_now = st.toggle("Use the real time", value=_STATE_KEY not in st.session_state)
    if use_now:
        st.session_state.pop(_STATE_KEY, None)
    else:
        with columns[1]:
            day = st.date_input("Date", value=local_now().date(), key="collections_date")
        with columns[2]:
            hour = st.slider("Hour (GST)", 0, 23, 10, key="collections_hour")
        with columns[3]:
            minute = st.select_slider("Minute", options=[0, 15, 30, 45], value=0,
                                      key="collections_minute")
        st.session_state[_STATE_KEY] = datetime.combine(
            day, time(hour, minute), tzinfo=GST)

    moment = _clock()
    decision = may_contact(moment)
    label = moment.strftime("%A %d %B, %H:%M")
    if decision.permitted:
        st.success(f"**{label} GST** — {decision.reason}.")
    else:
        nxt = (decision.next_permitted_at.strftime("%A %H:%M")
               if decision.next_permitted_at else "—")
        st.warning(f"**{label} GST** — no contact: {decision.reason}. "
                   f"Next permitted: {nxt}.")


def _seed_controls() -> None:
    """A reviewer needs something in the queue to look at."""
    with st.expander("Schedule a call", expanded=False):
        from ..finance.accounts import list_customers

        customers = list_customers()
        columns = st.columns([3, 3, 2])
        with columns[0]:
            chosen = st.selectbox(
                "Customer", customers, format_func=lambda c: f"{c.name} — {L.display_name(c.language)}",
                key="collections_customer")
        with columns[1]:
            strategy = st.selectbox(
                "Approved strategy", list_strategies(),
                format_func=lambda s: s.label(), key="collections_strategy")
        with columns[2]:
            days = st.number_input("Due in (days)", min_value=0, max_value=30, value=0,
                                   key="collections_due")
        st.caption(strategy.purpose)
        if st.button("Add to the queue", type="primary"):
            q.schedule(chosen.customer_id, strategy.name,
                       due_at=_clock() + timedelta(days=int(days)))
            st.rerun()


def _due_table(moment: datetime) -> None:
    ready = q.due(moment)
    st.markdown(f"##### Ready to dial · {len(ready)}")
    if not ready:
        C.quiet("Nothing is callable at this moment. The held list below says why.")
        return

    rows = [{
        "Customer": _customer_label(task.customer_id),
        "Language": _language_cell(task.language),
        "Strategy": task.treatment.label() if task.treatment else task.strategy,
        "Attempt": f"{task.attempts + 1} of "
                   f"{task.treatment.max_attempts if task.treatment else '?'}",
        "Due": task.due_at.strftime("%d %b %H:%M") if task.due_at else "",
    } for task, _ in ready]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _blocked_table(moment: datetime) -> None:
    held = q.blocked(moment)
    st.markdown(f"##### Held back · {len(held)}")
    C.quiet("The conduct evidence is here rather than in the list above: every "
            "call not made, and the rule that stopped it.")
    if not held:
        C.quiet("Nothing is being held.")
        return

    rows = [{
        "Customer": _customer_label(task.customer_id),
        "Strategy": task.strategy,
        "Gate": gate.gate.replace("_", " "),
        "Reason": gate.reason,
        "Retry": gate.retry_at.strftime("%a %d %b %H:%M") if gate.retry_at else "—",
    } for task, gate in held]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _register_table() -> None:
    rows = optout.opt_out_register()
    st.markdown(f"##### Opt-out register · {len(rows)}")
    C.quiet("Append-only. A withdrawal is a new row, because the question a "
            "reviewer asks is not \"are they opted out?\" but \"were they opted "
            "out on the day we called?\"")
    if not rows:
        C.quiet("No opt-outs recorded.")
        return

    st.dataframe(pd.DataFrame([{
        "Recorded": r["recorded_at"][:16].replace("T", " "),
        "Customer": _customer_label(r["customer_id"]),
        "Channel": r["channel"],
        "Status": "opted out" if r["opted_out"] else "withdrawn",
        "In their words": r["stated"],
        "Language": L.display_name(r["language"]),
    } for r in rows]), use_container_width=True, hide_index=True)


def _strategy_reference() -> None:
    with st.expander("The approved treatment strategies", expanded=False):
        C.quiet("A strategy is data, not a prompt. The contact limits are enforced "
                "by the queue, the clause allow-list by retrieval, the wording rule "
                "by the approved-wording guardrail, and the handover triggers by the "
                "supervisor — none of them by asking the model nicely.")
        for strategy in list_strategies():
            st.markdown(f"**{strategy.label()}** — {strategy.purpose}")
            clauses = (", ".join(sorted(strategy.allowed_clauses))
                       if strategy.allowed_clauses else "any clause for the journey")
            st.markdown(
                f"- at most **{strategy.max_attempts}** attempts, "
                f"**{strategy.min_hours_between_attempts}h** apart\n"
                f"- clauses: {clauses}\n"
                f"- approved wording only: **{'yes' if strategy.approved_wording_only else 'no'}**\n"
                f"- hands to a person on: {', '.join(strategy.handover_triggers)}"
            )


def collections_queue() -> None:
    C.hero("Collections Queue",
           "Scheduled, routine calls about an outstanding obligation — and every "
           "call the approved strategy would not let us place.")

    st.caption(f"Approved calling window: **{describe_window()}**.")

    _clock_controls()
    st.divider()
    _seed_controls()

    moment = _clock()
    _due_table(moment)
    st.divider()
    _blocked_table(moment)
    st.divider()
    _register_table()
    st.divider()
    _strategy_reference()
