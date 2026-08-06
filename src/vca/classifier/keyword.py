"""Keyword/heuristic fallback classifier.

Used when no fine-tuned transformer is available (e.g. before training, or when
torch/transformers are not installed). Deterministic and dependency-free, so the
end-to-end pipeline, demo and tests always run. Scores are coarse but sensible.
"""
from __future__ import annotations

import re

from .base import DRIVER_ORDER, BaseClassifier

# Indicative terms per driver. Matched as whole-word, case-insensitive.
KEYWORDS: dict[str, list[str]] = {
    "health": [
        "cancer", "chemo", "terminal", "illness", "ill", "disabled", "disability",
        "anxiety", "depression", "mental health", "not coping", "hospital",
        "stroke", "medication", "diagnosed", "heart condition", "memory",
    ],
    "life_events": [
        "passed away", "died", "death", "bereave", "bereaved", "funeral",
        "lost my job", "redundant", "redundancy", "divorce", "separated",
        "separation", "executor", "estate", "caring", "carer", "hours were cut",
        "income has", "made redundant",
    ],
    "resilience": [
        "can't afford", "cannot afford", "behind", "debts", "debt", "no savings",
        "struggling financially", "struggling", "borrowing", "hand to mouth",
        "heating and eating", "money for food", "owe money", "can't keep up",
        "hardship", "living hand",
    ],
    "capability": [
        "don't understand", "not good with computers", "confusing", "confused",
        "plain english", "a bit lost", "isn't my first language",
        "first language", "power of attorney", "daughter usually helps",
        "son to be able", "helps me with", "explain that",
    ],
}


def _match_score(text: str, terms: list[str]) -> float:
    low = text.lower()
    hits = 0
    for term in terms:
        pattern = r"\b" + re.escape(term.lower()) + r"\b"
        if re.search(pattern, low):
            hits += 1
    if hits == 0:
        return 0.0
    # Saturating score: 1 hit -> 0.75, 2+ -> ~0.9+.
    return min(0.6 + 0.15 * hits, 0.98)


class KeywordClassifier(BaseClassifier):
    def predict_scores(self, text: str) -> list[float]:
        return [_match_score(text, KEYWORDS[d.value]) for d in DRIVER_ORDER]
