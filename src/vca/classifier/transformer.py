"""Fine-tuned transformer classifier (inference).

Loads a DistilBERT model fine-tuned for multi-label vulnerability-driver
detection (see classifier/train.py) and produces per-driver probabilities.
Torch/transformers are imported lazily.
"""
from __future__ import annotations

from pathlib import Path

from .base import DRIVER_ORDER, BaseClassifier


class TransformerClassifier(BaseClassifier):
    def __init__(self, model_dir: str | Path, threshold: float = 0.5, max_length: int = 128):
        super().__init__(threshold=threshold)
        self.model_dir = str(model_dir)
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._torch = None

    @staticmethod
    def is_available(model_dir: str | Path) -> bool:
        """True if a fine-tuned model appears to exist at model_dir."""
        p = Path(model_dir)
        return p.exists() and any(p.glob("*.safetensors") or p.glob("pytorch_model.bin"))

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self._model.eval()

    def predict_scores(self, text: str) -> list[float]:
        self._load()
        torch = self._torch
        enc = self._tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self._model(**enc).logits[0]
        probs = torch.sigmoid(logits).tolist()
        # Model is configured with labels in DRIVER_ORDER (see train.py).
        return [float(probs[i]) for i in range(len(DRIVER_ORDER))]
