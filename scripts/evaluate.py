"""Evaluate the active classifier on the validation set.

Reports per-driver precision / recall / F1 and micro-averaged totals. Works with
whichever classifier the factory selects (fine-tuned transformer if trained,
else the keyword fallback), so you can compare the two.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vca.classifier.base import DRIVER_ORDER  # noqa: E402
from vca.classifier.factory import load_classifier  # noqa: E402
from vca.config import load_config  # noqa: E402


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the VCA classifier.")
    parser.add_argument("--prefer", default="auto", choices=["auto", "transformer", "keyword"])
    args = parser.parse_args()

    cfg = load_config()
    clf = load_classifier(cfg, prefer=args.prefer)
    print(f"Evaluating: {type(clf).__name__}\n")

    val_path = cfg.path("data", "labeled", "val.jsonl")
    rows = [json.loads(l) for l in val_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    stats = {d: {"tp": 0, "fp": 0, "fn": 0} for d in DRIVER_ORDER}
    for row in rows:
        det = clf.detect("eval", 0, row["text"])
        pred = {s.driver: s.triggered for s in det.scores}
        for d in DRIVER_ORDER:
            gold = bool(row["labels"][d.value])
            if pred[d] and gold:
                stats[d]["tp"] += 1
            elif pred[d] and not gold:
                stats[d]["fp"] += 1
            elif not pred[d] and gold:
                stats[d]["fn"] += 1

    print(f"{'driver':14} {'prec':>6} {'rec':>6} {'f1':>6}")
    tot = {"tp": 0, "fp": 0, "fn": 0}
    for d in DRIVER_ORDER:
        s = stats[d]
        p, r, f = _prf(s["tp"], s["fp"], s["fn"])
        print(f"{d.value:14} {p:6.2f} {r:6.2f} {f:6.2f}")
        for k in tot:
            tot[k] += s[k]
    p, r, f = _prf(tot["tp"], tot["fp"], tot["fn"])
    print(f"{'-'*34}")
    print(f"{'micro-avg':14} {p:6.2f} {r:6.2f} {f:6.2f}")


if __name__ == "__main__":
    main()
