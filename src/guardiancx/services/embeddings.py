"""Embeddings for GuardianCX RAG.

Backend priority (best available wins):
  1. **Azure OpenAI** embeddings (`text-embedding-3-small` by default).
  2. **sentence-transformers** (local `all-MiniLM-L6-v2`) — semantic quality
     with no external service.
  3. **Hashing** fallback — deterministic bag-of-tokens; keeps the app running
     with zero dependencies, but only matches exact shared words (low recall).

All backends return unit-norm vectors so cosine similarity is comparable. The
`signature` (backend + dimensionality) lets the vector store detect when the
embedder changed and re-index cleanly.
"""
from __future__ import annotations

import hashlib
import math
from typing import Optional

from config.settings import Settings, get_settings

from ..utils.logging import get_logger

log = get_logger("services.embeddings")

_FALLBACK_DIM = 384
_ST_MODEL = "all-MiniLM-L6-v2"


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class Embedder:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.backend = "hashing"
        self._azure = None
        self._st = None
        self._dim: Optional[int] = None

        if self.settings.azure_embeddings_enabled:
            try:
                from openai import AzureOpenAI

                self._azure = AzureOpenAI(
                    api_key=self.settings.azure_openai_api_key,
                    azure_endpoint=self.settings.azure_openai_endpoint,
                    api_version=self.settings.azure_openai_api_version,
                )
                self.backend = "azure-openai"
            except Exception as exc:  # noqa: BLE001
                log.warning("Azure embeddings unavailable: %s", exc)

        if self._azure is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._st = SentenceTransformer(_ST_MODEL)
                self.backend = "sentence-transformers"
            except Exception as exc:  # noqa: BLE001
                log.warning("sentence-transformers unavailable, using hashing: %s", exc)

    # --- embedding -------------------------------------------------------
    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._azure is not None:
            try:
                resp = self._azure.embeddings.create(
                    model=self.settings.azure_openai_embedding_deployment, input=texts,
                )
                return [_normalize(d.embedding) for d in resp.data]
            except Exception as exc:  # noqa: BLE001
                log.warning("Azure embed failed, falling back: %s", exc)
        if self._st is not None:
            vecs = self._st.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
            return [v.tolist() for v in vecs]
        return [self._hash_embed(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed_one("dimension probe"))
        return self._dim

    @property
    def signature(self) -> str:
        """Stable id of the embedding space; changes when backend/dim changes."""
        return f"{self.backend}-{self.dim}"

    @staticmethod
    def _hash_embed(text: str) -> list[float]:
        vec = [0.0] * _FALLBACK_DIM
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % _FALLBACK_DIM] += 1.0
        return _normalize(vec)


_EMBEDDER: Embedder | None = None


def get_embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder()
    return _EMBEDDER
