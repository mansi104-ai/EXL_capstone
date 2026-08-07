"""CLI: evaluate RAG retrieval quality against the labelled test set.

Usage:  python scripts/eval_rag.py [--k 3]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from guardiancx.rag.evaluation import evaluate_rag, load_testset  # noqa: E402
from guardiancx.rag.vector_store import get_vector_store  # noqa: E402
from guardiancx.services.embeddings import get_embedder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    testset = load_testset()
    report = evaluate_rag(testset, k=args.k)

    print(f"Embedder: {get_embedder().backend}  |  Vector store: {get_vector_store().backend}")
    print(f"RAG evaluation over {report.n} queries (k={report.k}):")
    for key, val in report.as_summary().items():
        print(f"  {key:14} {val}")
    print("  hit_rate by driver:")
    for d, hr in sorted(report.per_driver.items()):
        print(f"    {d:14} {hr:.3f}")

    misses = [r for r in report.results if not r.hit]
    if misses:
        print(f"\n  {len(misses)} miss(es):")
        for r in misses:
            print(f"    - [{r.driver}] expected {r.expected}, got {r.retrieved}")
            print(f"      query: {r.query}")
    else:
        print("\n  All queries retrieved an expected clause. [PASS]")


if __name__ == "__main__":
    main()
