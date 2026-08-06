"""CLI: build/verify the policy retrieval index.

Parses the policy and (if sentence-transformers is available) warms the
embedding retriever. Prints the parsed clauses so you can confirm coverage.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vca.config import load_config  # noqa: E402
from vca.rag.retriever import load_retriever  # noqa: E402


def main() -> None:
    cfg = load_config()
    policy_path = cfg.path(cfg.rag["policy_path"])
    retriever = load_retriever(
        policy_path=policy_path,
        embed_model=cfg.rag["embed_model"],
        top_k=int(cfg.rag.get("top_k", 3)),
    )
    print(f"Retriever: {type(retriever).__name__}")
    print(f"Parsed {len(retriever.sections)} policy clauses:")
    for s in retriever.sections:
        print(f"  [{s.driver.value:12}] {s.code} — {s.title}")


if __name__ == "__main__":
    main()
