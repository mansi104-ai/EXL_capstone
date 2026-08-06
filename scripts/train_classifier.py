"""CLI: fine-tune the vulnerability-driver classifier."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vca.classifier.train import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the VCA driver classifier.")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()
    train(epochs=args.epochs)


if __name__ == "__main__":
    main()
