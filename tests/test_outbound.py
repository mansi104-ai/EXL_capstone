"""Tests for the outbound collections channel.

Three things are being protected here, and none of them is testable by reading
the code:

* **A call is never placed outside the approved window.** The failure mode is a
  batch job that runs when the batch finishes rather than when the customer is
  awake, and it is invisible until it is in a recording.
* **An opt-out is honoured on the next call, not the next batch.** The register
  is consulted per task rather than per run, so a request made mid-call on one
  line stops a call already queued on another.
* **A refusal says which rule refused it.** A queue that quietly holds work back
  is indistinguishable from a queue that is broken, and neither is auditable.

The database is redirected to a temporary file for the whole module. The opt-out
register is append-only by design, so a test that wrote to the real one would
leave a customer opted out for every later run.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

from guardiancx.outbound.contact_rules import (
    GST,
    ContactWindow,
    describe_window,
    may_contact,
)
from guardiancx.outbound.strategy import MANDATORY_HANDOVER, get_strategy, list_strategies


@pytest.fixture(scope="module", autouse=True)
def _isolated_database(tmp_path_factory):
    from guardiancx.database import db

    previous = os.environ.get("GUARDIANCX_DATABASE_URL")
    path = tmp_path_factory.mktemp("outbound") / "outbound.db"
    os.environ["GUARDIANCX_DATABASE_URL"] = f"sqlite:///{path}"

    from config.settings import get_settings

    get_settings.cache_clear()
    db.reset_engine()
    yield
    if previous is None:
        os.environ.pop("GUARDIANCX_DATABASE_URL", None)
    else:
        os.environ["GUARDIANCX_DATABASE_URL"] = previous
    get_settings.cache_clear()
    db.reset_engine()


@pytest.fixture(autouse=True)
def _empty_queue():
    from guardiancx.outbound import queue as q

    q.clear_queue()
    yield
    q.clear_queue()


# Fixed points in the week, so a test never depends on when it is run.
MONDAY_10 = datetime(2026, 9, 7, 10, 0, tzinfo=GST)
FRIDAY_1230 = datetime(2026, 9, 11, 12, 30, tzinfo=GST)
SATURDAY_11 = datetime(2026, 9, 12, 11, 0, tzinfo=GST)
SUNDAY_11 = datetime(2026, 9, 13, 11, 0, tzinfo=GST)


# --------------------------------------------------------------------------- #
# Calling hours
# --------------------------------------------------------------------------- #
def test_a_call_inside_the_window_is_permitted():
    assert may_contact(MONDAY_10).permitted


@pytest.mark.parametrize("moment, expected", [
    (datetime(2026, 9, 7, 8, 59, tzinfo=GST), "before 09:00"),
    (datetime(2026, 9, 7, 20, 0, tzinfo=GST), "after 20:00"),
    (SUNDAY_11, "the weekend"),
    (FRIDAY_1230, "Friday prayers"),
])
def test_calls_outside_the_window_are_refused_with_a_reason(moment, expected):
    decision = may_contact(moment)
    assert not decision.permitted
    assert expected in decision.reason


def test_the_uae_weekend_permits_saturday_and_not_sunday():
    """Not a detail. Porting a Saturday-Sunday weekend into this market silently
    removes a day on which customers are more reachable than on a working day."""
    assert may_contact(SATURDAY_11).permitted
    assert not may_contact(SUNDAY_11).permitted


def test_a_refusal_says_when_to_come_back():
    decision = may_contact(datetime(2026, 9, 7, 6, 0, tzinfo=GST))
    assert decision.next_permitted_at == datetime(2026, 9, 7, 9, 0, tzinfo=GST)


def test_after_hours_rolls_to_the_next_permitted_morning():
    decision = may_contact(datetime(2026, 9, 12, 21, 0, tzinfo=GST))   # Saturday night
    # Sunday is a no-contact day, so the next opening is Monday.
    assert decision.next_permitted_at == datetime(2026, 9, 14, 9, 0, tzinfo=GST)


def test_friday_prayers_resume_at_the_end_of_the_window_not_the_next_day():
    decision = may_contact(FRIDAY_1230)
    assert decision.next_permitted_at == datetime(2026, 9, 11, 13, 30, tzinfo=GST)


def test_a_naive_datetime_is_read_as_local_time():
    """Gulf Standard Time, not UTC. Reading it as UTC shifts every window by four
    hours, in the direction that permits calls at five in the morning."""
    assert may_contact(datetime(2026, 9, 7, 10, 0)).permitted
    assert not may_contact(datetime(2026, 9, 7, 6, 0)).permitted


def test_a_public_holiday_blocks_contact():
    window = ContactWindow(holidays=frozenset({MONDAY_10.date()}))
    decision = may_contact(MONDAY_10, window)
    assert not decision.permitted
    assert "holiday" in decision.reason


def test_the_window_describes_itself_for_the_record():
    assert "09:00" in describe_window() and "Friday prayers" in describe_window()


# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #
def test_every_strategy_hands_over_on_hardship_and_distress():
    """The brief routes disputes, hardship and vulnerability to a person. A
    strategy that could switch that off would be the one bug worth having."""
    for strategy in list_strategies():
        for trigger in MANDATORY_HANDOVER:
            assert trigger in strategy.handover_triggers, strategy.name


def test_a_strategy_scopes_the_clauses_it_may_draw_on():
    strategy = get_strategy("document_expiry")
    assert strategy.permits_clause("VP-J13")
    # A document-expiry call must not become a collections call.
    assert not strategy.permits_clause("VP-J1")


def test_a_strategy_with_no_allow_list_permits_any_clause():
    from guardiancx.outbound.strategy import TreatmentStrategy
    from guardiancx.finance.taxonomy import Journey

    open_strategy = TreatmentStrategy(
        name="t", version="1.0", journey=Journey.GENERAL_SERVICING, purpose="")
    assert open_strategy.permits_clause("anything")


# --------------------------------------------------------------------------- #
# The queue and its gates
# --------------------------------------------------------------------------- #
def test_a_due_task_inside_the_window_is_offered():
    from guardiancx.outbound import queue as q

    q.schedule("CUST-6612", "early_arrears_d1_7", due_at=MONDAY_10)
    offered = q.due(MONDAY_10)
    assert [t.customer_id for t, _ in offered] == ["CUST-6612"]


def test_the_call_opens_in_the_language_on_the_customer_record():
    """An outbound call chooses a voice before anyone has spoken. Defaulting to
    English is how a multilingual channel becomes an English one."""
    from guardiancx.outbound import queue as q

    assert q.schedule("CUST-6612", "early_arrears_d1_7").language == "ur"
    assert q.schedule("CUST-7788", "early_arrears_d1_7").language == "ar"
    assert q.schedule("CUST-2205", "early_arrears_d1_7").language == "fil"


def test_a_task_outside_calling_hours_is_held_with_the_reason():
    from guardiancx.outbound import queue as q

    q.schedule("CUST-6612", "early_arrears_d1_7", due_at=SUNDAY_11)
    assert q.due(SUNDAY_11) == []
    held = q.blocked(SUNDAY_11)
    assert len(held) == 1
    assert held[0][1].gate == "calling_hours"


def test_an_opt_out_stops_the_call_and_is_recorded_as_an_opt_out():
    """The gate order is load-bearing. Reporting "outside calling hours" for a
    customer who has withdrawn consent implies the bank means to try tomorrow."""
    from guardiancx.outbound import optout, queue as q

    q.schedule("CUST-8830", "early_arrears_d1_7", due_at=MONDAY_10)
    assert len(q.due(MONDAY_10)) == 1

    optout.record_opt_out("CUST-8830", stated="stop calling me")
    assert q.due(MONDAY_10) == []
    held = q.blocked(MONDAY_10)
    assert held[0][1].gate == "opt_out"


def test_the_attempt_limit_counts_history_not_intent():
    from guardiancx.outbound import queue as q

    task = q.schedule("CUST-2205", "early_arrears_d1_7", due_at=MONDAY_10)
    strategy = get_strategy("early_arrears_d1_7")

    later = MONDAY_10
    for _ in range(strategy.max_attempts):
        q.record_attempt(task.task_id, now=later)
        later = later + timedelta(hours=strategy.min_hours_between_attempts + 1)
        # Keep it inside the window on the following days.
        later = later.replace(hour=10, minute=0)

    held = [g for t, g in q.blocked(later) if t.task_id == task.task_id]
    assert held and held[0].gate == "attempts"


def test_a_second_attempt_waits_the_required_gap():
    from guardiancx.outbound import queue as q

    task = q.schedule("CUST-2205", "early_arrears_d1_7", due_at=MONDAY_10)
    q.record_attempt(task.task_id, now=MONDAY_10)

    soon = MONDAY_10 + timedelta(hours=2)
    held = [g for t, g in q.blocked(soon) if t.task_id == task.task_id]
    assert held and held[0].gate == "attempts"
    assert held[0].retry_at is not None


def test_a_task_not_yet_due_is_not_offered():
    from guardiancx.outbound import queue as q

    q.schedule("CUST-2205", "early_arrears_d1_7", due_at=MONDAY_10 + timedelta(days=2))
    assert q.due(MONDAY_10) == []


def test_an_unknown_strategy_never_dials():
    from guardiancx.outbound import queue as q

    q.schedule("CUST-2205", "no_such_strategy", due_at=MONDAY_10)
    held = q.blocked(MONDAY_10)
    assert held[0][1].gate == "strategy"


def test_an_opt_out_cancels_the_work_as_well_as_blocking_it():
    """Blocking alone leaves the customer showing as outstanding work, which is
    how a queue keeps re-presenting somebody who asked to be left alone."""
    from guardiancx.outbound import optout, queue as q

    q.schedule("CUST-4471", "early_arrears_d8_30", due_at=MONDAY_10)
    optout.record_opt_out("CUST-4471", stated="take me off your list")
    cancelled = q.cancel_for_customer("CUST-4471", "customer opted out")
    assert cancelled == 1
    assert q.pending_tasks() == []


# --------------------------------------------------------------------------- #
# Hearing the opt-out
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("text", [
    "please stop calling me",
    "don't call me again",
    "take me off your list",
    "remove my number from your system",
    "I want to opt out",
    "no more calls please",
])
def test_a_plain_english_opt_out_is_heard(text):
    from guardiancx.outbound.optout import detect_opt_out

    assert detect_opt_out(text) is not None


@pytest.mark.parametrize("text, language", [
    ("لا تتصلوا بي مرة أخرى", "ar"),
    ("مجھے کال نہ کریں", "ur"),
    ("मुझे कॉल मत करो", "hi"),
])
def test_an_opt_out_is_heard_in_the_language_it_was_said_in(text, language):
    """The sentence a customer most wants understood is the one they are most
    likely to say in their own language."""
    from guardiancx.outbound.optout import detect_opt_out

    assert detect_opt_out(text, language) is not None


@pytest.mark.parametrize("text", [
    "can you stop the interest while we sort this out",
    "please stop the letters",
    "I can't talk right now",
    "call me back tomorrow",
    "don't call me at work, use my mobile",
])
def test_asking_for_help_is_not_read_as_an_opt_out(text):
    """A false positive here costs the customer their route back to the bank.
    A false negative costs one more call."""
    from guardiancx.outbound.optout import detect_opt_out

    assert detect_opt_out(text) is None


def test_an_opt_out_can_be_withdrawn_and_the_later_row_wins():
    from guardiancx.outbound import optout

    optout.record_opt_out("CUST-9001", stated="stop calling me")
    assert optout.is_opted_out("CUST-9001")
    optout.withdraw_opt_out("CUST-9001", stated="actually please do call about the plan")
    assert not optout.is_opted_out("CUST-9001")


def test_the_register_keeps_the_history_rather_than_overwriting_it():
    """"Were they opted out on the day we called?" is the question a reviewer
    asks, and only an append-only register can answer it."""
    from guardiancx.outbound import optout

    optout.record_opt_out("CUST-9002", stated="stop calling me")
    optout.withdraw_opt_out("CUST-9002", stated="you can call me again")
    rows = optout.opt_out_register("CUST-9002")
    assert len(rows) == 2
    assert [r["opted_out"] for r in rows] == [False, True]   # newest first


def test_a_blanket_opt_out_covers_every_channel():
    from guardiancx.outbound import optout

    optout.record_opt_out("CUST-9003", channel="all", stated="stop contacting me")
    assert optout.is_opted_out("CUST-9003", "voice")
    assert optout.is_opted_out("CUST-9003", "sms")
