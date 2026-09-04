"""Outbound collections — the channel that starts the call.

Everything else in GuardianCX answers a customer who has already rung. This
package is the other direction: a due obligation raises a call, and the approved
treatment strategy decides whether it may be placed, what may be said on it, and
when to stop trying.

Four modules, each owning one question:

* `strategy` — *what is approved*: the named, versioned treatment strategies,
  their contact limits, their clause allow-lists, and the signals that hand the
  call to a person.
* `contact_rules` — *may we call now*: the permitted window in Gulf Standard
  Time, and when it next opens.
* `optout` — *has the customer told us to stop*: the durable register, and the
  detector that hears the request mid-call in the language it was said in.
* `queue` — *the gates*: the only route from a scheduled obligation to a dialled
  call, and the reason recorded whenever a task does not pass.
"""
from .contact_rules import ContactWindow, describe_window, may_contact
from .optout import detect_opt_out, is_opted_out, opt_out_register, record_opt_out
from .queue import (
    Task,
    blocked,
    cancel_for_customer,
    check_gates,
    close,
    due,
    pending_tasks,
    record_attempt,
    schedule,
)
from .strategy import STRATEGIES, TreatmentStrategy, get_strategy, list_strategies

__all__ = [
    "ContactWindow", "may_contact", "describe_window",
    "detect_opt_out", "is_opted_out", "record_opt_out", "opt_out_register",
    "Task", "schedule", "due", "blocked", "check_gates", "record_attempt",
    "close", "cancel_for_customer", "pending_tasks",
    "TreatmentStrategy", "STRATEGIES", "get_strategy", "list_strategies",
]
