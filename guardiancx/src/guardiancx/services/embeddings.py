"""Embeddings for GuardianCX RAG.

Primary: Azure OpenAI embeddings (`text-embedding-3-small` by default).
Fallback: a deterministic local hashing embedder so the vector store and RAG
work with zero external dependencies. Both return unit-norm vectors so cosine
similarity is comparable.
"""
from __future__ import annotations

import hashlib
import math
from typing import Optional

from config.settings import Settings, get_settings

from ..utils.logging import get_logger

log = get_logger("services.embeddings")

_FALLBACK_DIM = 384


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class Embedder:
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self._client = None
        self.backend = "hashing"
        if self.settings.azure_embeddings_enabled:
            try:
                from openai import AzureOpenAI

                self._client = AzureOpenAI(
                    api_key=self.settings.azure_openai_api_key,
                    azure_endpoint=self.settings.azure_openai_endpoint,
                    api_version=self.settings.azure_openai_api_version,
                )
                self.backend = "azure-openai"
            except Exception as exc:  # noqa: BLE001
                log.warning("Azure embeddings unavailable, using hashing fallback: %s", exc)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._client is not None:
            try:
                resp = self._client.embeddings.create(
                    model=self.settings.azure_openai_embedding_deployment,
                    input=texts,
                )
                return [_normalize(d.embedding) for d in resp.data]
            except Exception as exc:  # noqa: BLE001
                log.warning("Azure embed call failed, using hashing fallback: %s", exc)
        return [self._hash_embed(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    @staticmethod
    def _hash_embed(text: str) -> list[float]:
        """Deterministic bag-of-hashed-tokens embedding (no dependencies)."""
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
