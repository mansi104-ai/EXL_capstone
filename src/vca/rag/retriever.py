"""Adaptation retriever over the firm's vulnerability policy.

Given a detected driver and the triggering utterance, retrieve the most relevant
policy clause(s) and return them as :class:`Adaptation` objects.

Two backends:
  * EmbeddingRetriever  -- sentence-transformers + FAISS (semantic).
  * KeywordRetriever    -- token-overlap fallback, no heavy deps.
The factory picks embeddings if available, else the fallback, so the pipeline
always runs.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from ..schemas import Adaptation, Driver
from .policy import PolicySection, parse_policy


def _sections_to_adaptation(section: PolicySection, score: float) -> Adaptation:
    return Adaptation(
        driver=section.driver,
        title=section.title,
        guidance=section.body,
        policy_reference=section.code,
        retrieval_score=round(float(score), 4),
    )


class BaseRetriever:
    def __init__(self, sections: list[PolicySection], top_k: int = 3):
        self.sections = sections
        self.top_k = top_k

    def _candidates(self, driver: Driver | None) -> list[PolicySection]:
        if driver is None:
            return self.sections
        scoped = [s for s in self.sections if s.driver == driver]
        return scoped or self.sections

    def retrieve(self, query: str, driver: Driver | None = None, top_k: int | None = None) -> list[Adaptation]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Keyword / token-overlap fallback
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[a-z]+")
_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is",
    "are", "i", "im", "my", "me", "you", "it", "that", "this", "so", "at", "as",
    "be", "have", "has", "do", "not", "no", "if", "we", "they", "them",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 2]


class KeywordRetriever(BaseRetriever):
    def __init__(self, sections: list[PolicySection], top_k: int = 3):
        super().__init__(sections, top_k)
        # Precompute TF vectors + IDF for cosine similarity.
        self._docs = [Counter(_tokens(s.text)) for s in sections]
        df: Counter = Counter()
        for d in self._docs:
            for term in d:
                df[term] += 1
        n = max(len(self._docs), 1)
        self._idf = {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}

    def _vec(self, counter: Counter) -> dict[str, float]:
        return {t: c * self._idf.get(t, 1.0) for t, c in counter.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        common = set(a) & set(b)
        num = sum(a[t] * b[t] for t in common)
        da = math.sqrt(sum(v * v for v in a.values()))
        db = math.sqrt(sum(v * v for v in b.values()))
        return num / (da * db) if da and db else 0.0

    def retrieve(self, query: str, driver: Driver | None = None, top_k: int | None = None) -> list[Adaptation]:
        k = top_k or self.top_k
        qvec = self._vec(Counter(_tokens(query)))
        scored = []
        for i, section in enumerate(self.sections):
            if driver is not None and section.driver != driver:
                continue
            score = self._cosine(qvec, self._vec(self._docs[i]))
            scored.append((score, section))
        if not scored:
            scored = [(0.0, s) for s in self.sections]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [_sections_to_adaptation(s, sc) for sc, s in scored[:k]]


# --------------------------------------------------------------------------- #
# Semantic embedding retriever
# --------------------------------------------------------------------------- #
class EmbeddingRetriever(BaseRetriever):
    def __init__(self, sections: list[PolicySection], embed_model: str, top_k: int = 3):
        super().__init__(sections, top_k)
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(embed_model)
        self._emb = self._model.encode(
            [s.text for s in sections], normalize_embeddings=True, convert_to_numpy=True
        )

    def retrieve(self, query: str, driver: Driver | None = None, top_k: int | None = None) -> list[Adaptation]:
        import numpy as np

        k = top_k or self.top_k
        q = self._model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self._emb @ q  # cosine (normalized)
        idxs = list(range(len(self.sections)))
        if driver is not None:
            scoped = [i for i in idxs if self.sections[i].driver == driver]
            idxs = scoped or idxs
        idxs.sort(key=lambda i: float(sims[i]), reverse=True)
        return [_sections_to_adaptation(self.sections[i], float(sims[i])) for i in idxs[:k]]


def load_retriever(policy_path: str | Path, embed_model: str, top_k: int = 3, prefer: str = "auto") -> BaseRetriever:
    sections = parse_policy(policy_path)
    if prefer in ("auto", "embedding"):
        try:
            return EmbeddingRetriever(sections, embed_model=embed_model, top_k=top_k)
        except Exception:  # noqa: BLE001 - fall back if deps/model unavailable
            if prefer == "embedding":
                raise
    return KeywordRetriever(sections, top_k=top_k)
