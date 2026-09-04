"""Seed GuardianCX: ingest policies and run the synthetic conversations through
the agent pipeline so the dashboards, approval queue, and audit trail have data.

Usage:  python scripts/seed_data.py [--fresh]
"""
import argparse
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from guardiancx.agents.graph import process_conversation  # noqa: E402
from guardiancx.database.db import init_engine  # noqa: E402
from guardiancx.database.repository import clear_all, list_evidence, list_pending  # noqa: E402
from guardiancx.outbound import queue as outbound_queue  # noqa: E402
from guardiancx.outbound.contact_rules import local_now  # noqa: E402
from guardiancx.rag.vector_store import ensure_ingested  # noqa: E402
from guardiancx.services.synthetic_data import list_conversations  # noqa: E402

# Outbound obligations to put on the queue, so the Collections Queue page has
# something real to reason about on a first run. Spread across the calling
# window and the languages in the book, because a queue that is entirely English
# and entirely due right now demonstrates neither of the things this channel is
# for.
SEED_TASKS = [
    ("CUST-6612", "early_arrears_d1_7", 0),      # Urdu — salary-transfer arrears
    ("CUST-7788", "early_arrears_d8_30", 0),     # Arabic — instalment overdue
    ("CUST-4471", "payment_plan_check_in", 2),   # Hindi — arrangement review
    ("CUST-2205", "document_expiry", 5),         # Filipino — Emirates ID expiring
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh", action="store_true", help="Clear evidence + audit first.")
    args = parser.parse_args()

    init_engine()
    if args.fresh:
        clear_all()
    chunks = ensure_ingested()
    print(f"Policy chunks indexed: {chunks}")

    for conv in list_conversations():
        states = process_conversation(conv)
        flagged = sum(1 for s in states if s.get("decision") and s["decision"].assessment.triggered)
        print(f"  {conv['conversation_id']} ({conv['customer_name']}): "
              f"{len(states)} turns, {flagged} flagged")

    if args.fresh:
        outbound_queue.clear_queue()
    for customer_id, strategy, in_days in SEED_TASKS:
        due = local_now() + timedelta(days=in_days)
        task = outbound_queue.schedule(customer_id, strategy, due_at=due)
        print(f"  queued {customer_id} · {strategy} · opens in "
              f"{task.language} · due {due:%d %b %H:%M}")

    print(f"Evidence records: {len(list_evidence())} | pending approvals: {len(list_pending())}")
    print(f"Outbound queue: {len(outbound_queue.pending_tasks())} task(s)")


if __name__ == "__main__":
    main()
