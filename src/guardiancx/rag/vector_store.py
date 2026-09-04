"""Vector store for GuardianCX RAG.

Primary backend: ChromaDB (persistent). Fallback: an in-memory cosine store so
retrieval works with no native dependency. Both use the shared Embedder (Azure
OpenAI or hashing fallback), so the whole RAG path degrades gracefully.

**Journey-aware retrieval.** Similarity alone is not enough in this domain. A
bereaved customer in a collections call and a bereaved customer reporting a scam
produce near-identical embeddings and need completely different clauses. So a
query carries the classified journey alongside the text, and clauses tagged for
that journey are boosted above ones that merely read similarly. The boost is
applied after retrieval rather than as a hard filter, so a clause that is right
on the merits is never excluded for lacking a tag.
"""
from __future__ import annotations

import json
import math
from typing import Any, Optional

from config.settings import get_settings

from ..finance.taxonomy import Journey
from ..services.embeddings import get_embedder
from ..utils.logging import get_logger
from ..utils.types import Driver, PolicyChunk
from .chunking import Chunk, chunk_policy_dir

log = get_logger("rag.vector_store")

# How much a journey tag match is worth against a cosine score in 0-1. Large
# enough to reorder near-ties, small enough that a much better semantic match
# still wins.
JOURNEY_BOOST = 0.15


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(y * y for y in b)) or 1.0
    return num / (da * db)


class VectorStore:
    def __init__(self):
        self.embedder = get_embedder()
        self.backend = "in-memory"
        self._chroma = None
        self._mem: list[dict[str, Any]] = []
        # Collection name is keyed to the embedding space, so switching embedders
        # (e.g. hashing -> sentence-transformers) uses a fresh collection instead
        # of mixing incompatible vectors.
        collection = "guardiancx_policies_" + self.embedder.signature.replace("-", "_")
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(get_settings().chroma_dir))
            self._chroma = client.get_or_create_collection(
                collection, metadata={"hnsw:space": "cosine"},
            )
            self.backend = "chromadb"
        except Exception as exc:  # noqa: BLE001
            log.warning("ChromaDB unavailable, using in-memory store: %s", exc)

    # --- ingest ----------------------------------------------------------
    def count(self) -> int:
        if self._chroma is not None:
            try:
                return self._chroma.count()
            except Exception:  # noqa: BLE001
                return 0
        return len(self._mem)

    def ingest_chunks(self, chunks: list[Chunk]) -> int:
        embeds = self.embedder.embed([c.embed_text for c in chunks])
        # Chroma metadata values must be scalars, so the journey list travels as
        # a comma-joined string and is split back out on read.
        metadatas = [
            {"ref": c.ref, "title": c.title,
             "driver": c.driver.value if c.driver else "",
             "journeys": ",".join(c.journey_values),
             # Chroma metadata must be scalar, so the approved wordings travel as
             # one string and are split back out on read.
             "offers": " || ".join(c.offers),
             # The non-English wordings travel as JSON for the same reason. A
             # separator-joined form would have to survive Arabic and Urdu text
             # containing the separator; JSON already answers that question.
             "offers_i18n": json.dumps(c.offers_by_language, ensure_ascii=False)
                            if c.offers_by_language else "",
             "source": c.source}
            for c in chunks
        ]
        if self._chroma is not None:
            self._chroma.upsert(
                ids=[c.ref for c in chunks],
                embeddings=embeds,
                documents=[c.text for c in chunks],
                metadatas=metadatas,
            )
        else:
            self._mem = []
            for meta, c, e in zip(metadatas, chunks, embeds):
                self._mem.append({**meta, "text": c.text, "embed": e})
        return len(chunks)

    def ingest_dir(self, directory) -> int:
        return self.ingest_chunks(chunk_policy_dir(directory))

    # --- query -----------------------------------------------------------
    @staticmethod
    def _to_chunk(meta: dict, document: str, score: float) -> PolicyChunk:
        raw = meta.get("journeys") or ""
        offers = meta.get("offers") or ""
        i18n_raw = meta.get("offers_i18n") or ""
        try:
            i18n = json.loads(i18n_raw) if i18n_raw else {}
        except (TypeError, ValueError):
            # A malformed metadata blob must not take the call down: the clause
            # is still correct, it just loses its translations for this turn.
            i18n = {}
        return PolicyChunk(
            policy_reference=meta.get("ref", ""),
            title=meta.get("title", ""),
            driver=Driver(meta["driver"]) if meta.get("driver") else None,
            text=document,
            score=round(score, 4),
            journeys=[j for j in raw.split(",") if j],
            offers=[o.strip() for o in offers.split("||") if o.strip()],
            offers_by_language={k: list(v) for k, v in i18n.items() if v},
        )

    def query(self, text: str, driver: Optional[Driver] = None, top_k: int = 3,
              journey: Optional[Journey] = None) -> list[PolicyChunk]:
        """Retrieve the most relevant clauses.

        When a `journey` is supplied, more candidates are fetched than requested
        and re-scored with the journey boost before the top_k is taken — the
        clause that is right for *this situation* gets to overtake the one that
        merely reads similarly.
        """
        qvec = self.embedder.embed_one(text)
        fetch = top_k * 3 if journey else top_k
        results: list[PolicyChunk] = []

        if self._chroma is not None:
            # Clauses with no driver are *cross-cutting* — recording duties, the
            # ban on selling into vulnerability, complaint handling. They apply
            # to every driver by definition, so a driver filter has to admit them;
            # filtering on the driver alone makes them permanently unreachable.
            where = ({"$or": [{"driver": driver.value}, {"driver": ""}]}
                     if driver else None)
            try:
                res = self._chroma.query(
                    query_embeddings=[qvec], n_results=fetch, where=where,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Chroma query failed: %s", exc)
                return []
            metas = (res.get("metadatas") or [[]])[0]
            docs = (res.get("documents") or [[]])[0]
            dists = (res.get("distances") or [[]])[0] or [0.0] * len(metas)
            results = [self._to_chunk(meta, doc, 1.0 - float(dist))
                       for meta, doc, dist in zip(metas, docs, dists)]
        else:
            scored = []
            for item in self._mem:
                if driver and item["driver"] not in (driver.value, ""):
                    continue  # cross-cutting clauses (no driver) always qualify
                scored.append((_cosine(qvec, item["embed"]), item))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [self._to_chunk(item, item["text"], sc)
                       for sc, item in scored[:fetch]]

        if journey is None:
            return results[:top_k]

        for chunk in results:
            if journey.value in chunk.journeys:
                chunk.journey_match = True
                chunk.score = round(min(1.0, chunk.score + JOURNEY_BOOST), 4)
        results.sort(key=lambda c: c.score, reverse=True)
        return results[:top_k]


_STORE: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _STORE
    if _STORE is None:
        _STORE = VectorStore()
    return _STORE


def ensure_ingested() -> int:
    """Ingest the policy corpus, re-ingesting when the files on disk have changed
    (e.g. a new policy document was added). Returns the indexed chunk count."""
    from config.settings import POLICY_DIR

    from .chunking import chunk_policy_dir

    store = get_vector_store()
    expected = len(chunk_policy_dir(POLICY_DIR))
    if store.count() != expected:
        store.ingest_dir(POLICY_DIR)
    return store.count()
