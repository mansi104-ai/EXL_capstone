"""Seed GuardianCX: ingest policies and run the synthetic conversations through
the agent pipeline so the dashboards, approval queue, and audit trail have data.

Usage:  python scripts/seed_data.py [--fresh]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from guardiancx.agents.graph import process_conversation  # noqa: E402
from guardiancx.database.db import init_engine  # noqa: E402
from guardiancx.database.repository import clear_all, list_evidence, list_pending  # noqa: E402
from guardiancx.rag.vector_store import ensure_ingested  # noqa: E402
from guardiancx.services.synthetic_data import list_conversations  # noqa: E402


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

    print(f"Evidence records: {len(list_evidence())} | pending approvals: {len(list_pending())}")


if __name__ == "__main__":
    main()
