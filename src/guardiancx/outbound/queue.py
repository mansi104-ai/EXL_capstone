"""The outbound queue — a scheduled obligation becomes a call, or explains why not.

This is the piece that makes the channel outbound. Everything else in the system
answers a customer who has already rung; here, a due obligation raises the call.

The shape is deliberately unglamorous: a task is created when something falls
due, and before it can be dialled it has to pass every gate in `GATES`. A task
that fails a gate is never silently dropped — it comes back with a reason and,
where the block is temporal, a time to try again.

**Gates, not checks.** The difference matters. A check is something the dialler
consults and may forget to; a gate is something the dialler cannot get past. So
`due()` is the only way to obtain a task to call, it applies every gate, and
there is no second path that skips them. The opt-out gate in particular is
positioned before the calling-hours gate for a reason that shows up in the
evidence: a customer who has opted out should be recorded as opted out, not as
"outside calling hours", because the second reason implies the bank intends to
try again tomorrow.

**Why the register is consulted per task rather than per batch.** An opt-out
recorded three minutes ago, mid-call, on another line, must stop the next call.
Caching the register at the top of a batch run is exactly the optimisation that
produces a call placed twelve minutes after the customer asked for it to stop.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Optional

from ..database.db import session_scope
from ..database.models import CallTask as CallTaskRow
from ..database.repository import audit
from ..finance.accounts import get_customer
from ..utils.language import normalise as normalise_language
from ..utils.logging import get_logger
from .contact_rules import GST, ContactDecision, local_now, may_contact
from .optout import is_opted_out
from .strategy import TreatmentStrategy, get_strategy

log = get_logger("outbound.queue")

PENDING = "pending"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
CANCELLED = "cancelled"


@dataclass
class Task:
    """A scheduled contact attempt, as the dialler sees it."""

    task_id: str
    customer_id: str
    strategy: str
    language: str
    due_at: datetime
    state: str = PENDING
    attempts: int = 0
    last_attempt_at: Optional[datetime] = None
    account_id: str = ""
    detail: dict = None

    @property
    def treatment(self) -> Optional[TreatmentStrategy]:
        return get_strategy(self.strategy)


@dataclass(frozen=True)
class GateResult:
    """Why a task may or may not be dialled right now."""

    allowed: bool
    gate: str
    reason: str
    retry_at: Optional[datetime] = None

    def __bool__(self) -> bool:
        return self.allowed


# --------------------------------------------------------------------------- #
# The gates
# --------------------------------------------------------------------------- #
def _gate_strategy(task: Task, now: datetime) -> GateResult:
    if task.treatment is None:
        return GateResult(False, "strategy",
                          f"no approved strategy named {task.strategy!r}")
    return GateResult(True, "strategy", f"under {task.treatment.label()}")


def _gate_consent(task: Task, now: datetime) -> GateResult:
    """The opt-out register. First, and never cached."""
    strategy = task.treatment
    channel = strategy.channel if strategy else "voice"
    if is_opted_out(task.customer_id, channel):
        return GateResult(False, "opt_out",
                          "the customer has asked not to be contacted")
    return GateResult(True, "opt_out", "no opt-out on the register")


def _gate_attempts(task: Task, now: datetime) -> GateResult:
    strategy = task.treatment
    if strategy is None:
        return GateResult(False, "attempts", "no strategy")
    if task.attempts >= strategy.max_attempts:
        return GateResult(
            False, "attempts",
            f"{task.attempts} of {strategy.max_attempts} attempts already made")
    if task.last_attempt_at is not None:
        last = task.last_attempt_at
        if last.tzinfo is None:
            last = last.replace(tzinfo=GST)
        earliest = last + timedelta(hours=strategy.min_hours_between_attempts)
        if now < earliest:
            return GateResult(
                False, "attempts",
                f"less than {strategy.min_hours_between_attempts}h since the last attempt",
                earliest)
    return GateResult(True, "attempts", f"attempt {task.attempts + 1}")


def _gate_due(task: Task, now: datetime) -> GateResult:
    due = task.due_at
    if due.tzinfo is None:
        due = due.replace(tzinfo=GST)
    if now < due:
        return GateResult(False, "due", "not yet due", due)
    return GateResult(True, "due", "due")


def _gate_hours(task: Task, now: datetime) -> GateResult:
    strategy = task.treatment
    window = strategy.window if strategy else None
    decision: ContactDecision = may_contact(now, window)
    return GateResult(bool(decision), "calling_hours", decision.reason,
                      decision.next_permitted_at)


# Order is meaningful — see the module docstring. The first gate to refuse is the
# reason recorded, so the most fundamental reason has to be asked first.
GATES: list[Callable[[Task, datetime], GateResult]] = [
    _gate_strategy,
    _gate_consent,
    _gate_due,
    _gate_attempts,
    _gate_hours,
]


def check_gates(task: Task, now: Optional[datetime] = None) -> GateResult:
    """Run every gate in order; return the first refusal, or the last pass."""
    moment = local_now(now)
    result = GateResult(True, "none", "no gates ran")
    for gate in GATES:
        result = gate(task, moment)
        if not result.allowed:
            return result
    return result


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _to_task(row: CallTaskRow) -> Task:
    return Task(
        task_id=row.task_id, customer_id=row.customer_id, strategy=row.strategy,
        language=row.language, due_at=row.due_at, state=row.state,
        attempts=row.attempts, last_attempt_at=row.last_attempt_at,
        account_id=row.account_id, detail=dict(row.detail or {}),
    )


def schedule(customer_id: str, strategy: str, *, due_at: Optional[datetime] = None,
             account_id: str = "", language: str = "", detail: Optional[dict] = None) -> Task:
    """Put a contact obligation on the queue.

    The language defaults to the customer's record rather than to English. An
    outbound call has to pick a voice before anyone has spoken, and picking
    English by default is how a multilingual channel quietly becomes an English
    one with a translation feature.
    """
    if language:
        code = normalise_language(language)
    else:
        customer = get_customer(customer_id)
        code = normalise_language(customer.language if customer else "en")

    task_id = uuid.uuid4().hex[:12]
    when = due_at or local_now()
    with session_scope() as session:
        session.add(CallTaskRow(
            task_id=task_id, customer_id=customer_id, strategy=strategy,
            language=code, account_id=account_id, due_at=when,
            state=PENDING, detail=dict(detail or {}),
        ))
    audit("outbound.scheduled", detail={
        "task_id": task_id, "customer_id": customer_id,
        "strategy": strategy, "language": code, "due_at": when.isoformat(),
    })
    return Task(task_id=task_id, customer_id=customer_id, strategy=strategy,
                language=code, due_at=when, account_id=account_id,
                detail=dict(detail or {}))


#: States in which a task is still the queue's business. `IN_PROGRESS` belongs
#: here: an attempt having been made is not the same as the obligation being
#: discharged. A call that rang out is exactly the case the retry gap exists for,
#: and dropping the task from the queue the moment it was dialled would mean the
#: attempt limit could never be reached — every attempt would look like the
#: first one to a queue that had already forgotten the task.
OPEN_STATES = (PENDING, IN_PROGRESS)


def pending_tasks() -> list[Task]:
    """Every task the queue still owes an outcome on, soonest due first."""
    with session_scope() as session:
        rows = (session.query(CallTaskRow)
                .filter(CallTaskRow.state.in_(OPEN_STATES))
                .order_by(CallTaskRow.due_at.asc()).all())
        return [_to_task(r) for r in rows]


def due(now: Optional[datetime] = None) -> list[tuple[Task, GateResult]]:
    """Every pending task that passes every gate, with the gate result that let it."""
    moment = local_now(now)
    out = []
    for task in pending_tasks():
        result = check_gates(task, moment)
        if result.allowed:
            out.append((task, result))
    return out


def blocked(now: Optional[datetime] = None) -> list[tuple[Task, GateResult]]:
    """Every pending task that a gate is holding back, and which gate.

    The console shows this next to the due list. A queue that only displays what
    it is about to do hides the more interesting half — the calls it is *not*
    making, and the rule that stopped each one.
    """
    moment = local_now(now)
    out = []
    for task in pending_tasks():
        result = check_gates(task, moment)
        if not result.allowed:
            out.append((task, result))
    return out


def record_attempt(task_id: str, *, outcome: str = "",
                   conversation_id: str = "", now: Optional[datetime] = None) -> None:
    """Log that the call was placed. Counts against the strategy's attempt limit.

    The task stays open. Placing a call is not the same as reaching anyone, and
    the obligation is discharged by `close`, not by dialling.
    """
    moment = local_now(now)
    with session_scope() as session:
        row = session.query(CallTaskRow).filter(
            CallTaskRow.task_id == task_id).one_or_none()
        if row is None:
            return
        row.attempts += 1
        row.last_attempt_at = moment
        row.state = IN_PROGRESS
        if outcome:
            row.outcome = outcome
        if conversation_id:
            row.conversation_id = conversation_id
    audit("outbound.attempted", conversation_id=conversation_id,
          detail={"task_id": task_id, "outcome": outcome})


def close(task_id: str, *, outcome: str, state: str = COMPLETED,
          conversation_id: str = "") -> None:
    """Finish with a task — the call happened, or the obligation no longer stands."""
    with session_scope() as session:
        row = session.query(CallTaskRow).filter(
            CallTaskRow.task_id == task_id).one_or_none()
        if row is None:
            return
        row.state = state
        row.outcome = outcome
        if conversation_id:
            row.conversation_id = conversation_id
    audit("outbound.closed", conversation_id=conversation_id,
          detail={"task_id": task_id, "outcome": outcome, "state": state})


def cancel_for_customer(customer_id: str, reason: str) -> int:
    """Drop every pending task for a customer. Returns how many.

    Called when an opt-out is recorded mid-call. The opt-out gate would refuse
    these tasks anyway, but leaving them pending means the queue keeps offering
    a customer who has asked to be left alone, and the console keeps showing
    them as work outstanding. Honouring the request means removing the work.
    """
    with session_scope() as session:
        rows = (session.query(CallTaskRow)
                .filter(CallTaskRow.customer_id == customer_id)
                .filter(CallTaskRow.state.in_([PENDING, IN_PROGRESS])).all())
        for row in rows:
            row.state = CANCELLED
            row.outcome = reason
        count = len(rows)
    if count:
        audit("outbound.cancelled", detail={"customer_id": customer_id,
                                            "reason": reason, "tasks": count})
    return count


def clear_queue() -> None:
    """Empty the queue. Tests and the seed script only."""
    with session_scope() as session:
        session.query(CallTaskRow).delete()
