"""CLI: run the VCA pipeline end-to-end over sample conversations.

Streams each conversation, prints the transcript with advisory prompts inline,
records evidence, then prints the portfolio report and verifies the chain.
"""
import argparse
import sys
from pathlib import Path

# Windows terminals default to cp1252; force UTF-8 so guidance glyphs render.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vca.config import load_config  # noqa: E402
from vca.pipeline import VCAPipeline  # noqa: E402
from vca.reporting.metrics import build_report  # noqa: E402
from vca.schemas import HandlerAction, Outcome, Speaker  # noqa: E402

BLUE, YELLOW, GREEN, DIM, RESET = "\033[94m", "\033[93m", "\033[92m", "\033[2m", "\033[0m"


def _auto_outcome(guidance):
    # In the demo, assume the handler accepts the advisory guidance.
    return Outcome(action=HandlerAction.ACCEPTED, note="Acknowledged in demo run.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the VCA end-to-end demo.")
    parser.add_argument(
        "--transcripts", nargs="*", default=None,
        help="Transcript JSON files. Defaults to all in data/transcripts.",
    )
    parser.add_argument("--fresh", action="store_true", help="Clear the evidence log first.")
    args = parser.parse_args()

    cfg = load_config()
    evidence_path = cfg.path(cfg.evidence["path"])
    if args.fresh and evidence_path.exists():
        evidence_path.unlink()

    transcripts = args.transcripts or sorted(
        str(p) for p in cfg.path("data", "transcripts").glob("*.json")
    )
    pipeline = VCAPipeline.from_config(cfg)
    print(f"Classifier: {type(pipeline.classifier).__name__}  |  "
          f"Retriever: {type(pipeline.advisor.retriever).__name__}\n")

    for tpath in transcripts:
        conv_id = Path(tpath).stem
        print(f"{BLUE}=== Conversation {conv_id} ==={RESET}")
        for event in pipeline.process_transcript(tpath, conversation_id=conv_id,
                                                  on_guidance=_auto_outcome):
            u = event.utterance
            who = "Customer" if u.speaker == Speaker.CUSTOMER else "Handler "
            print(f"  {DIM}{who}:{RESET} {u.text}")
            if event.guidance:
                for line in event.guidance.message.splitlines():
                    print(f"    {YELLOW}▶ {line}{RESET}")
        print()

    store = pipeline.evidence
    report = build_report(store)
    print(f"{GREEN}=== Portfolio report ==={RESET}")
    for k, v in report.as_dict().items():
        print(f"  {k}: {v}")
    ok = "VALID" if report.chain_valid else f"BROKEN at {report.first_bad_record}"
    print(f"  evidence chain: {ok}")


if __name__ == "__main__":
    main()
