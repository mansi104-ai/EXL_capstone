"""Selects the best available classifier.

Prefers the fine-tuned transformer if a trained model exists and torch is
importable; otherwise falls back to the keyword classifier so the pipeline
always runs.
"""
from __future__ import annotations

from ..config import Config, load_config
from .base import BaseClassifier
from .keyword import KeywordClassifier


def load_classifier(cfg: Config | None = None, prefer: str = "auto") -> BaseClassifier:
    cfg = cfg or load_config()
    threshold = float(cfg.classifier.get("threshold", 0.5))
    model_dir = cfg.path(cfg.classifier["model_dir"])

    if prefer == "keyword":
        return KeywordClassifier(threshold=threshold)

    if prefer in ("auto", "transformer"):
        try:
            import torch  # noqa: F401

            from .transformer import TransformerClassifier

            if TransformerClassifier.is_available(model_dir):
                return TransformerClassifier(
                    model_dir=model_dir,
                    threshold=threshold,
                    max_length=int(cfg.classifier.get("max_length", 128)),
                )
        except ImportError:
            pass
        if prefer == "transformer":
            raise RuntimeError(
                f"No fine-tuned model at {model_dir} (or torch missing). "
                "Run scripts/train_classifier.py first."
            )

    return KeywordClassifier(threshold=threshold)
