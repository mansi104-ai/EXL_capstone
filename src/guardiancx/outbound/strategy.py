"""Approved treatment strategies — the thing the agent is not allowed to deviate from.

The brief this channel was built for asks for a voice channel that "follows
approved treatment strategies without deviation". A strategy is not a prompt and
not a tone of voice. It is a named, versioned, signed-off object that fixes, for
one situation:

* **when** the customer may be contacted, and how often;
* **what may be said** — which policy clauses the call may draw on, and whether
  the agent may speak anything other than their approved wording;
* **what ends the call** — the signals that stop the strategy and hand to a
  person.

Encoding it as data rather than as instructions in a prompt is the whole point.
A prompt is a request that a model usually honours; a strategy is a gate that
runs whether the model cooperates or not. Everything below is checked in code:
the clause allow-list is enforced by retrieval, the wording rule by
`guardrails.wording`, the contact limits by the queue, and the handover triggers
by the supervisor.

**The handover triggers are the load-bearing part.** The brief is explicit that
disputes, hardship claims and vulnerability signals go to a human. That is not a
degraded outcome to be minimised — for a collections call it is frequently the
*correct* outcome, and a strategy tuned to avoid it would be a strategy tuned to
keep vulnerable customers talking to a machine. So every strategy here hands over
on hardship and on distress, and none of them can be configured not to.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..finance.taxonomy import Journey
from .contact_rules import ContactWindow

# Signals that end a strategy and hand the call to a person. Every strategy
# carries these; a strategy may add to them and may not remove them.
MANDATORY_HANDOVER: tuple[str, ...] = (
    "dispute",            # the customer disputes the debt or the amount
    "hardship",           # a hardship or forbearance claim
    "vulnerability",      # any vulnerability driver triggered
    "distress",           # measured distress, whatever the words said
    "bereavement",
    "complaint",
    "third_party",        # someone other than the customer is on the line
)


@dataclass(frozen=True)
class TreatmentStrategy:
    """One approved strategy. Immutable — a change is a new version, not an edit."""

    name: str
    version: str
    journey: Journey
    purpose: str                       # one line, for the console and the record

    # Contact limits. These are limits on *history*, which is why the queue is
    # persistent: "three attempts" means three ever, not three since the process
    # last restarted.
    max_attempts: int = 3
    min_hours_between_attempts: int = 24
    channel: str = "voice"
    window: ContactWindow = field(default_factory=ContactWindow)

    # What may be said. `allowed_clauses` is an allow-list of policy references;
    # empty means "any clause the retrieval step returns for this journey".
    allowed_clauses: frozenset[str] = frozenset()
    # When true, every offer put to the customer must be approved wording,
    # verbatim. Derived or model-composed offers are blocked.
    approved_wording_only: bool = True

    extra_handover: tuple[str, ...] = ()

    @property
    def handover_triggers(self) -> tuple[str, ...]:
        return MANDATORY_HANDOVER + tuple(self.extra_handover)

    def permits_clause(self, reference: str) -> bool:
        return not self.allowed_clauses or reference in self.allowed_clauses

    def label(self) -> str:
        return f"{self.name} v{self.version}"


# --------------------------------------------------------------------------- #
# The registry
#
# Four strategies, matching the routine obligations the brief names: a missed
# payment, an early-arrears follow-up, a payment-plan check-in, and an expiring
# identity document. Each is deliberately narrow. A strategy that covers
# "collections" in general is a strategy nobody can sign off, because the right
# thing to say seven days after a missed instalment is not the right thing to say
# ninety days later.
# --------------------------------------------------------------------------- #
STRATEGIES: dict[str, TreatmentStrategy] = {
    "early_arrears_d1_7": TreatmentStrategy(
        name="early_arrears_d1_7",
        version="1.0",
        journey=Journey.ARREARS_COLLECTIONS,
        purpose=("A first, single missed instalment within seven days. The call "
                 "establishes what happened before it establishes what is owed — "
                 "on a salary-transfer facility the commonest answer is that the "
                 "salary did not arrive."),
        max_attempts=2,
        min_hours_between_attempts=48,
        allowed_clauses=frozenset({"VP-J1", "VP-J14", "VP-J15", "VP-J16"}),
    ),
    "early_arrears_d8_30": TreatmentStrategy(
        name="early_arrears_d8_30",
        version="1.0",
        journey=Journey.ARREARS_COLLECTIONS,
        purpose=("Arrears between eight and thirty days. The call may discuss a "
                 "plan and must state the credit-reporting position before the "
                 "customer agrees to anything."),
        max_attempts=3,
        min_hours_between_attempts=24,
        allowed_clauses=frozenset({"VP-J1", "VP-J2", "VP-J3", "VP-J4",
                                   "VP-J14", "VP-J15", "VP-J16"}),
    ),
    "payment_plan_check_in": TreatmentStrategy(
        name="payment_plan_check_in",
        version="1.0",
        journey=Journey.FORBEARANCE_REQUEST,
        purpose=("A scheduled review of an arrangement already in place. The "
                 "review date exists so a temporary adaptation is revisited "
                 "rather than left to lapse."),
        max_attempts=2,
        min_hours_between_attempts=72,
        allowed_clauses=frozenset({"VP-J1", "VP-J2", "VP-J3", "VP-J13", "VP-J16"}),
    ),
    "document_expiry": TreatmentStrategy(
        name="document_expiry",
        version="1.0",
        journey=Journey.GENERAL_SERVICING,
        purpose=("An Emirates ID or passport on file is about to expire. Purely "
                 "administrative, and it must not become a collections call: "
                 "the customer did not ask to discuss their balance."),
        max_attempts=2,
        min_hours_between_attempts=72,
        allowed_clauses=frozenset({"VP-J13"}),
        # A document call that finds arrears has found a different conversation,
        # and it is not this one's to have.
        extra_handover=("arrears",),
    ),
}


def get_strategy(name: str) -> Optional[TreatmentStrategy]:
    return STRATEGIES.get(name)


def list_strategies() -> list[TreatmentStrategy]:
    return list(STRATEGIES.values())
