"""Vector store for GuardianCX RAG.

Primary backend: ChromaDB (persistent). Fallback: an in-memory cosine store so
retrieval works with no native dependency. Both use the shared Embedder (Azure
OpenAI or hashing fallback), so the whole RAG path degrades gracefully.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from config.settings import get_settings

from ..services.embeddings import get_embedder
from ..utils.logging import get_logger
from ..utils.types import Driver, PolicyChunk
from .chunking import Chunk, chunk_policy_dir

log = get_logger("rag.vector_store")
_COLLECTION = "guardiancx_policies"


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
        try:
            import chromadb

            client = chromadb.PersistentClient(path=str(get_settings().chroma_dir))
            self._chroma = client.get_or_create_collection(_COLLECTION)
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
        if self._chroma is not None:
            self._chroma.upsert(
                ids=[c.ref for c in chunks],
                embeddings=embeds,
                documents=[c.text for c in chunks],
                metadatas=[
                    {"ref": c.ref, "title": c.title,
                     "driver": c.driver.value if c.driver else "",
                     "source": c.source}
                    for c in chunks
                ],
            )
        else:
            self._mem = []
            for c, e in zip(chunks, embeds):
                self._mem.append({
                    "ref": c.ref, "title": c.title,
                    "driver": c.driver.value if c.driver else "",
                    "source": c.source, "text": c.text, "embed": e,
                })
        return len(chunks)

    def ingest_dir(self, directory) -> int:
        return self.ingest_chunks(chunk_policy_dir(directory))

    # --- query -----------------------------------------------------------
    def query(self, text: str, driver: Optional[Driver] = None, top_k: int = 3) -> list[PolicyChunk]:
        qvec = self.embedder.embed_one(text)
        if self._chroma is not None:
            where = {"driver": driver.value} if driver else None
            try:
                res = self._chroma.query(
                    query_embeddings=[qvec], n_results=top_k, where=where,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Chroma query failed: %s", exc)
                return []
            out: list[PolicyChunk] = []
            metas = (res.get("metadatas") or [[]])[0]
            docs = (res.get("documents") or [[]])[0]
            dists = (res.get("distances") or [[]])[0] or [0.0] * len(metas)
            for meta, doc, dist in zip(metas, docs, dists):
                out.append(PolicyChunk(
                    policy_reference=meta.get("ref", ""),
                    title=meta.get("title", ""),
                    driver=Driver(meta["driver"]) if meta.get("driver") else None,
                    text=doc,
                    score=round(1.0 - float(dist), 4),
                ))
            return out
        # in-memory
        scored = []
        for item in self._mem:
            if driver and item["driver"] != driver.value:
                continue
            scored.append((_cosine(qvec, item["embed"]), item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            PolicyChunk(
                policy_reference=item["ref"], title=item["title"],
                driver=Driver(item["driver"]) if item["driver"] else None,
                text=item["text"], score=round(sc, 4),
            )
            for sc, item in scored[:top_k]
        ]


_STORE: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _STORE
    if _STORE is None:
        _STORE = VectorStore()
    return _STORE


def ensure_ingested() -> int:
    """Ingest the policy corpus if the store is empty. Returns chunk count."""
    from config.settings import POLICY_DIR

    store = get_vector_store()
    if store.count() == 0:
        store.ingest_dir(POLICY_DIR)
    return store.count()
