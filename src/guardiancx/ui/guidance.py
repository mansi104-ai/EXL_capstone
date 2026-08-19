"""The Advisory Guidance panel.

The word that matters in the title is *advisory*. This page shows what the system
would tell a handler to do, and it is built so that a handler can decide whether
to trust it — which means showing the reasoning, the source, and the limits, not
just the conclusion.

Four things are on screen for every piece of guidance, in the order a handler
actually needs them:

1. **What to say.** The drafted, customer-facing reply, in the words the customer
   would hear. This is the thing the handler uses; everything else explains it.
2. **What it rests on.** The policy clauses retrieved, with their scores, and
   whether the match came from the banking journey or from similarity alone.
3. **What it must not do.** The obligations engaged by this journey and the
   actions prohibited on it — the constraints the guidance was written under.
4. **Whether it may be used yet.** Risk, approval state, and every guardrail
   result, passed and failed alike. A handler who sees only failures cannot tell
   the difference between "checked and clean" and "not checked".

Running it against a **real customer** rather than a bare utterance is what makes
the advice concrete: with an account selected, the guidance knows the payment is
£612.40 and the customer is three months behind, and the drafted reply says so.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ..agents.graph import process_turn
from ..agents.reply import build_reply_prompt, compose_reply
from ..finance.accounts import context_summary, list_customers
from ..finance.taxonomy import JOURNEY_LABELS, PRODUCT_LABELS, STRESS_LABELS
from ..guardrails.manager import get_guardrail_manager
from ..services.claude_client import EFFORT_REPLY, get_llm
from . import components as C

# Openers chosen to exercise a different journey each, so the page is useful
# before anyone types anything.
_EXAMPLES = {
    "Bereavement + arrears": "My husband died last month and I'm three months behind "
                             "on the mortgage.",
    "Job loss": "I lost my job on Friday and I can't make this month's payment.",
    "Scam": "Someone called saying they were from the bank and told me to move my "
            "money to a safe account.",
    "Gambling harm": "I've been gambling again and I've maxed out the card.",
    "Third-party request": "Can you give me my husband's credit card balance?",
    "Crisis": "I feel completely hopeless. I don't know how I'll carry on.",
}


def guidance_panel() -> None:
    C.hero(
        "Advisory Guidance",
        "What the system would advise a handler to say, why, and whether it may be "
        "used yet. Advisory only — a person decides every time.",
    )

    customers = list_customers()
    labels = {f"{c.name} · {c.customer_id}": c.customer_id for c in customers}

    top = st.columns([2, 3])
    with top[0]:
        chosen = st.selectbox("Caller", list(labels),
                              help="The guidance is grounded in this customer's accounts.")
        customer_id = labels[chosen]
    with top[1]:
        example = st.selectbox("Start from an example", ["— write my own —", *_EXAMPLES])

    default = _EXAMPLES.get(example, "")
    text = st.text_area("What the customer said", value=default, height=90,
                        key=f"guide_text_{example}")

    with st.expander("What we hold for this caller"):
        st.code(context_summary(customer_id), language="text")
        st.caption("Figures are masked here exactly as they are masked in the reply — "
                   "an account is confirmed by its last four digits, never in full.")

    if st.button("Generate guidance", type="primary") and text.strip():
        with st.spinner("Ten agents working…"):
            st.session_state["guide_state"] = process_turn(
                "ADVISORY", customer_id, 0, "customer", text.strip())

    state = st.session_state.get("guide_state")
    if not state:
        st.info("Pick a caller and an example, then generate the guidance.")
        _portfolio()
        return

    _render(state)
    _portfolio()


def _render(state: dict) -> None:
    decision = state.get("decision")
    account = state.get("account_request")
    context = state.get("financial_context")
    sentiment = state.get("sentiment")

    # --- 1. what to say ----------------------------------------------------
    st.markdown("### What to say")
    reply = _drafted_reply(state)
    st.success(reply)

    clarity = get_guardrail_manager().run_reply(
        reply,
        journey=context.journey if context else None,
        account_refused=bool(account and account.refused),
    )
    for result in clarity.results:
        if result.passed:
            st.caption(f"✓ {result.name.replace('_', ' ')} — {result.detail}")
        else:
            st.warning(f"{result.name.replace('_', ' ')} — {result.detail}")

    if account is not None and account.refused:
        st.error("**This request was refused.** The caller asked about an account held "
                 "by someone else. The refusal stands regardless of the relationship "
                 "or the circumstances — offer the alternative instead.")

    # --- 2. the situation ---------------------------------------------------
    st.markdown("### The situation")
    cols = st.columns(3)
    with cols[0]:
        if context is not None:
            st.markdown(f"**Journey**  \n{JOURNEY_LABELS.get(context.journey, '—')}")
            st.caption(f"{PRODUCT_LABELS.get(context.product, '—')} · governed by "
                       f"{context.sourcebook}")
    with cols[1]:
        assessment = state.get("assessment")
        triggered = [d.value.replace("_", " ") for d in assessment.triggered] if assessment else []
        st.markdown("**Vulnerability drivers**")
        C.chips(triggered or [], tone="on")
    with cols[2]:
        if sentiment is not None and sentiment.source != "skipped":
            st.markdown(f"**Customer state**  \n{sentiment.label}")
            C.meter("distress", sentiment.distress, C.scale_color(sentiment.distress))

    if context is not None and context.stress_indicators:
        st.markdown("**Financial detriment evidenced**")
        C.chips([STRESS_LABELS.get(s, s.value) for s in context.stress_indicators],
                tone="alert" if context.acute else "on")

    if account is not None and account.asked and account.facts:
        st.markdown("**Account facts the handler may state**")
        for fact in account.facts:
            st.markdown(f"- {fact}")

    # --- 3. the constraints -------------------------------------------------
    if context is not None and (context.obligations or context.prohibited):
        st.markdown("### What this journey requires")
        left, right = st.columns(2)
        with left:
            st.markdown("**Obligations engaged**")
            for obligation in context.obligations:
                st.markdown(f"- {obligation}")
        with right:
            if context.prohibited:
                st.markdown("**Must not be offered**")
                for item in context.prohibited:
                    st.markdown(f"- {item}")
                st.caption("Enforced by the `prohibited_action` guardrail, not only "
                           "requested in the prompt.")

    # --- 4. grounding and clearance ----------------------------------------
    st.markdown("### What it rests on")
    retrieved = state.get("retrieved", [])
    if retrieved:
        st.dataframe(
            pd.DataFrame([
                {
                    "clause": c.policy_reference,
                    "title": c.title,
                    "score": c.score,
                    "matched on journey": "yes" if c.journey_match else "",
                    "approved wording": len(c.offers),
                }
                for c in retrieved
            ]),
            width="stretch", hide_index=True,
        )
        st.caption("Retrieval is filtered by vulnerability driver and re-ranked by "
                   "banking journey, so a bereaved customer settling an estate and a "
                   "bereaved customer reporting a scam get different clauses.")
    else:
        st.caption("No policy retrieved — nothing triggered a lookup, so the guidance "
                   "makes no policy claim.")

    if decision is not None:
        st.markdown("### Whether it may be used")
        st.markdown(
            f"{C.risk_pill(decision.risk_level.value)} "
            f"{C.pill('approval: ' + decision.approval_status.value, C.WARN if decision.approval_status.value == 'pending' else C.ACCENT)}",
            unsafe_allow_html=True,
        )
        if decision.routing_reasons:
            st.markdown("**Routed for human approval because:** "
                        + ", ".join(decision.routing_reasons))
        if decision.recommendation:
            with st.expander("The handler-facing recommendation in full", expanded=False):
                C.recommendation_card(decision)
        with st.expander("Every guardrail, passed and failed"):
            C.guardrail_badges([r.model_dump() for r in decision.guardrails.results])
        with st.expander("Agent trace"):
            C.agent_trace(state.get("trace", []))


def _drafted_reply(state: dict) -> str:
    """The customer-facing draft, from the model where one is configured."""
    decision = state.get("decision")
    llm = get_llm()
    if llm.available:
        from ..agents.prompts import HANDLER_REPLY_SYSTEM

        prompt = build_reply_prompt(
            state.get("masked_text", ""), decision,
            context=state.get("financial_context"),
            sentiment=state.get("sentiment"),
            retrieved=state.get("retrieved"),
            account=state.get("account_request"),
        )
        drafted = llm.text(HANDLER_REPLY_SYSTEM, prompt, max_tokens=220,
                           effort=EFFORT_REPLY)
        if drafted and drafted.strip():
            return drafted.strip()
    return compose_reply(
        decision, state.get("financial_context"), state.get("sentiment"),
        retrieved=state.get("retrieved"), account=state.get("account_request"),
    )


def _portfolio() -> None:
    from ..database.repository import list_evidence

    st.divider()
    st.markdown("**Recent guidance across the portfolio**")
    rows = [r for r in list_evidence() if r.get("recommendation")]
    if not rows:
        st.caption("No guidance recorded yet.")
        return
    st.dataframe(
        pd.DataFrame([
            {
                "conversation": r["conversation_id"],
                "journey": r.get("journey", ""),
                "risk": r["risk_level"],
                "guidance": r["recommendation"],
                "citations": ", ".join(r.get("citations") or []),
                "confidence": round(float(r.get("model_confidence") or 0), 2),
                "approval": r["approval_status"],
            }
            for r in rows[:20]
        ]),
        width="stretch", hide_index=True,
    )
