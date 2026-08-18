"""CLI: evaluate RAG retrieval quality against the labelled test set.

Runs the test set twice — filtered by vulnerability driver alone, and with
journey-aware re-ranking as the Policy Retrieval agent actually queries — so
the contribution of the finance layer is measured rather than asserted.

Usage:  python scripts/eval_rag.py [--k 3]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from guardiancx.rag.evaluation import compare_journey_reranking, load_testset  # noqa: E402
from guardiancx.rag.vector_store import get_vector_store  # noqa: E402
from guardiancx.services.embeddings import get_embedder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    testset = load_testset()
    reports = compare_journey_reranking(testset, k=args.k)
    report = reports["journey_aware"]

    print(f"Embedder: {get_embedder().backend}  |  Vector store: {get_vector_store().backend}")
    print(f"RAG evaluation over {report.n} queries (k={report.k}):\n")
    print(f"  {'configuration':16} {'hit@k':>7} {'prec@k':>7} {'recall@k':>9} {'MRR':>7}")
    for name in ("driver_only", "journey_aware"):
        summary = reports[name].as_summary()
        print(f"  {name:16} {summary['hit_rate@k']:>7.3f} {summary['precision@k']:>7.3f} "
              f"{summary['recall@k']:>9.3f} {summary['mrr']:>7.3f}")
    delta = report.mrr - reports["driver_only"].mrr
    print(f"\n  Journey-aware re-ranking moves MRR by {delta:+.3f}.")

    print("\n  hit_rate by driver (journey-aware):")
    for d, hr in sorted(report.per_driver.items()):
        print(f"    {d:14} {hr:.3f}")

    misses = [r for r in report.results if not r.hit]
    if misses:
        print(f"\n  {len(misses)} miss(es):")
        for r in misses:
            print(f"    - [{r.driver}/{r.journey}] expected {r.expected}, got {r.retrieved}")
            print(f"      query: {r.query}")
    else:
        print("\n  All queries retrieved an expected clause. [PASS]")


if __name__ == "__main__":
    main()
