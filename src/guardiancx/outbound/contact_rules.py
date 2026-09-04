"""When the bank is allowed to call, and when it is not.

Permitted calling hours are the part of a treatment strategy that is easiest to
state and easiest to breach, because nothing about a queue naturally knows what
time it is where the customer lives. A dialler that runs when the batch finishes
will call at 03:00, and the recording will prove it.

So the window is a gate in front of the dialler rather than a note in a runbook,
and it answers two questions: *may I call now*, and if not, *when may I*. The
second matters as much as the first — a task that is merely refused gets retried
in a loop; a task that is told when to come back reschedules itself.

**Time zone.** All of this is Gulf Standard Time, UTC+04:00, expressed here as a
fixed offset rather than through the tz database. That is not a shortcut: the UAE
has not observed daylight saving, and a fixed offset is therefore exactly correct
while also removing a dependency on `tzdata` being installed on the host. If this
system is ever pointed at a market with DST, this constant is the thing to
change, and it should become a real zone rather than a different number.

**What is regulation and what is the bank's own strategy.** The CBUAE Consumer
Protection Standards require that collections contact be at reasonable times and
free of harassment; they do not publish a clock. The specific window below —
09:00 to 20:00, no contact on the UAE Sunday, and nothing during Friday
congregational prayers — is Gulf Union Bank's approved treatment strategy, which
is what the agent is actually bound by. The distinction is worth keeping visible:
a reviewer asking "why 20:00?" should be pointed at the strategy document, not at
a regulation that does not say it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional

# Gulf Standard Time. No daylight saving, so a fixed offset is exact.
GST = timezone(timedelta(hours=4), name="GST")

# The approved window, local time.
WINDOW_OPEN = time(9, 0)
WINDOW_CLOSE = time(20, 0)

# Friday congregational prayers. Contact during this window is not merely
# unwelcome, it is contact the customer cannot answer.
FRIDAY = 4                      # Python weekday(): Monday is 0
JUMUAH_START = time(12, 0)
JUMUAH_END = time(13, 30)

# The UAE weekend is Saturday and Sunday. The bank's strategy permits contact on
# a Saturday, when many customers are more reachable than on a working day, and
# forbids it on a Sunday.
NO_CONTACT_WEEKDAYS = {6}       # Sunday


@dataclass(frozen=True)
class ContactWindow:
    """The approved calling window. Overridable so a strategy can be stricter."""

    open_at: time = WINDOW_OPEN
    close_at: time = WINDOW_CLOSE
    no_contact_weekdays: frozenset[int] = frozenset(NO_CONTACT_WEEKDAYS)
    # Public holidays — Eid, National Day. Supplied by the caller rather than
    # hardcoded, because the Islamic holidays move each year and a stale table
    # baked into source is worse than no table at all.
    holidays: frozenset[date] = field(default_factory=frozenset)
    respect_jumuah: bool = True


@dataclass(frozen=True)
class ContactDecision:
    """May we call, and if not, when."""

    permitted: bool
    reason: str
    next_permitted_at: Optional[datetime] = None

    def __bool__(self) -> bool:
        return self.permitted


def local_now(now: Optional[datetime] = None) -> datetime:
    """`now` in Gulf Standard Time, whatever it arrived as.

    A naive datetime is read as already-local rather than as UTC. That choice
    matters: tests and the console both pass local wall-clock times, and
    silently treating those as UTC would shift every one of them four hours and
    make the window look wrong in exactly the way that is hardest to notice.
    """
    now = now or datetime.now(GST)
    if now.tzinfo is None:
        return now.replace(tzinfo=GST)
    return now.astimezone(GST)


def _blocked_day(moment: datetime, window: ContactWindow) -> Optional[str]:
    if moment.date() in window.holidays:
        return "a public holiday"
    if moment.weekday() in window.no_contact_weekdays:
        return "the weekend"
    return None


def _next_open(moment: datetime, window: ContactWindow) -> datetime:
    """The next instant contact becomes permitted, at or after `moment`."""
    candidate = moment
    for _ in range(30):          # a month of lookahead is more than enough
        blocked = _blocked_day(candidate, window)
        if blocked:
            candidate = datetime.combine(candidate.date() + timedelta(days=1),
                                         window.open_at, tzinfo=GST)
            continue
        if candidate.timetz().replace(tzinfo=None) < window.open_at:
            candidate = datetime.combine(candidate.date(), window.open_at, tzinfo=GST)
            continue
        if candidate.timetz().replace(tzinfo=None) >= window.close_at:
            candidate = datetime.combine(candidate.date() + timedelta(days=1),
                                         window.open_at, tzinfo=GST)
            continue
        if (window.respect_jumuah and candidate.weekday() == FRIDAY
                and JUMUAH_START <= candidate.time() < JUMUAH_END):
            candidate = datetime.combine(candidate.date(), JUMUAH_END, tzinfo=GST)
            continue
        return candidate
    return candidate


def may_contact(now: Optional[datetime] = None,
                window: Optional[ContactWindow] = None) -> ContactDecision:
    """Is outbound contact permitted at this moment?

    Returns the reason either way. A refusal that does not say why cannot be
    audited, and a queue that cannot explain its own inaction looks identical to
    a queue that is broken.
    """
    window = window or ContactWindow()
    moment = local_now(now)

    blocked = _blocked_day(moment, window)
    if blocked:
        return ContactDecision(
            False, f"outside the approved window — {blocked}",
            _next_open(moment, window))

    clock = moment.time()
    if clock < window.open_at:
        return ContactDecision(
            False,
            f"before {window.open_at.strftime('%H:%M')} Gulf Standard Time",
            _next_open(moment, window))
    if clock >= window.close_at:
        return ContactDecision(
            False,
            f"after {window.close_at.strftime('%H:%M')} Gulf Standard Time",
            _next_open(moment, window))
    if (window.respect_jumuah and moment.weekday() == FRIDAY
            and JUMUAH_START <= clock < JUMUAH_END):
        return ContactDecision(
            False, "during Friday prayers", _next_open(moment, window))

    return ContactDecision(True, "within the approved calling window")


def describe_window(window: Optional[ContactWindow] = None) -> str:
    """One line a reviewer can read, for the console and the evidence record."""
    window = window or ContactWindow()
    days = "Monday to Saturday" if window.no_contact_weekdays == frozenset({6}) else "permitted days"
    tail = ", not during Friday prayers" if window.respect_jumuah else ""
    return (f"{window.open_at.strftime('%H:%M')}–{window.close_at.strftime('%H:%M')} "
            f"Gulf Standard Time, {days}{tail}")
