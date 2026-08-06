"""Portfolio-level reporting over the evidence log.

Turns the immutable evidence trail into fair-treatment metrics the firm can show
a supervisor: how many vulnerable customers were identified, across which
drivers, and whether guidance was consistently acted on.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from ..evidence.store import EvidenceStore
from ..schemas import Driver, HandlerAction


@dataclass
class PortfolioReport:
    total_detections: int = 0
    conversations_flagged: int = 0
    detections_by_driver: dict[str, int] = field(default_factory=dict)
    outcomes: dict[str, int] = field(default_factory=dict)
    acceptance_rate: float = 0.0
    chain_valid: bool = True
    first_bad_record: str | None = None

    def as_dict(self) -> dict:
        return {
            "total_detections": self.total_detections,
            "conversations_flagged": self.conversations_flagged,
            "detections_by_driver": self.detections_by_driver,
            "outcomes": self.outcomes,
            "acceptance_rate": round(self.acceptance_rate, 3),
            "chain_valid": self.chain_valid,
            "first_bad_record": self.first_bad_record,
        }


def build_report(store: EvidenceStore) -> PortfolioReport:
    records = store.read_all()
    driver_counts: Counter = Counter()
    outcome_counts: Counter = Counter()
    conversations: set[str] = set()

    for rec in records:
        conversations.add(rec["conversation_id"])
        for s in rec["detection"]["scores"]:
            if s["triggered"]:
                driver_counts[s["driver"]] += 1
        outcome_counts[rec["outcome"]["action"]] += 1

    # Ensure every driver appears, even at zero.
    for d in Driver:
        driver_counts.setdefault(d.value, 0)

    accepted = outcome_counts.get(HandlerAction.ACCEPTED.value, 0)
    responded = sum(
        outcome_counts.get(a.value, 0)
        for a in (HandlerAction.ACCEPTED, HandlerAction.DISMISSED, HandlerAction.MODIFIED)
    )
    acceptance = accepted / responded if responded else 0.0

    ok, bad = store.verify_chain()

    return PortfolioReport(
        total_detections=len(records),
        conversations_flagged=len(conversations),
        detections_by_driver=dict(sorted(driver_counts.items())),
        outcomes=dict(outcome_counts),
        acceptance_rate=acceptance,
        chain_valid=ok,
        first_bad_record=bad,
    )
