"""RAG retrieval evaluation for GuardianCX.

Given a labelled test set of (query, driver, expected clause references), measure
how well the vector store retrieves the correct policy clause. This is how we
*prove* retrieval quality (the bereavement→VP-L1 class of bug) instead of
spot-checking.

Metrics (at cut-off k):
- hit_rate@k : fraction of queries where at least one expected clause is in top-k.
- precision@k: mean over queries of (relevant retrieved) / k.
- recall@k   : mean over queries of (relevant retrieved) / (expected).
- MRR        : mean reciprocal rank of the first relevant clause.

Retrieval is scoped to the labelled driver, mirroring how the Policy Retrieval
agent queries in production.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..utils.types import Driver
from .vector_store import VectorStore, ensure_ingested, get_vector_store


def default_testset_path() -> Path:
    # repo_root/eval/rag_testset.jsonl  (this file: src/guardiancx/rag/evaluation.py)
    return Path(__file__).resolve().parents[3] / "eval" / "rag_testset.jsonl"


def load_testset(path: str | Path | None = None) -> list[dict]:
    p = Path(path) if path else default_testset_path()
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


@dataclass
class QueryResult:
    query: str
    driver: str
    expected: list[str]
    retrieved: list[str]
    hit: bool
    reciprocal_rank: float
    precision: float
    recall: float


@dataclass
class EvalReport:
    k: int
    n: int
    hit_rate: float
    precision_at_k: float
    recall_at_k: float
    mrr: float
    per_driver: dict[str, float] = field(default_factory=dict)  # driver -> hit_rate
    results: list[QueryResult] = field(default_factory=list)

    def as_summary(self) -> dict:
        return {
            "queries": self.n,
            "k": self.k,
            "hit_rate@k": round(self.hit_rate, 3),
            "precision@k": round(self.precision_at_k, 3),
            "recall@k": round(self.recall_at_k, 3),
            "mrr": round(self.mrr, 3),
        }


def evaluate_rag(
    testset: list[dict],
    store: Optional[VectorStore] = None,
    k: int = 3,
) -> EvalReport:
    ensure_ingested()
    store = store or get_vector_store()

    results: list[QueryResult] = []
    driver_hits: dict[str, list[int]] = {}

    for item in testset:
        driver = Driver(item["driver"]) if item.get("driver") else None
        expected = set(item["expected"])
        chunks = store.query(item["query"], driver=driver, top_k=k)
        retrieved = [c.policy_reference for c in chunks]

        relevant = [r for r in retrieved if r in expected]
        hit = bool(relevant)
        rr = 0.0
        for rank, ref in enumerate(retrieved, start=1):
            if ref in expected:
                rr = 1.0 / rank
                break
        precision = len(relevant) / k if k else 0.0
        recall = len(relevant) / len(expected) if expected else 0.0

        results.append(QueryResult(
            query=item["query"], driver=item.get("driver", ""),
            expected=sorted(expected), retrieved=retrieved,
            hit=hit, reciprocal_rank=rr, precision=precision, recall=recall,
        ))
        driver_hits.setdefault(item.get("driver", ""), []).append(int(hit))

    n = len(results) or 1
    report = EvalReport(
        k=k, n=len(results),
        hit_rate=sum(r.hit for r in results) / n,
        precision_at_k=sum(r.precision for r in results) / n,
        recall_at_k=sum(r.recall for r in results) / n,
        mrr=sum(r.reciprocal_rank for r in results) / n,
        per_driver={d: sum(v) / len(v) for d, v in driver_hits.items()},
        results=results,
    )
    return report
