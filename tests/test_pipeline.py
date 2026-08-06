from pathlib import Path

from vca.classifier.keyword import KeywordClassifier
from vca.evidence.store import EvidenceStore
from vca.guidance.advisor import HandlerAdvisor
from vca.ingestion.simulated import SimulatedSource
from vca.pipeline import VCAPipeline
from vca.rag.policy import parse_policy
from vca.rag.retriever import KeywordRetriever
from vca.schemas import HandlerAction, Outcome, Speaker

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "data" / "policy" / "vulnerability_policy.md"
CONV = ROOT / "data" / "transcripts" / "CONV-001.json"


def _pipeline(tmp_path):
    clf = KeywordClassifier(threshold=0.5)
    advisor = HandlerAdvisor(KeywordRetriever(parse_policy(POLICY), top_k=1))
    evidence = EvidenceStore(tmp_path / "ev.jsonl")
    return VCAPipeline(clf, advisor, evidence)


def test_handler_utterances_not_classified(tmp_path):
    pipe = _pipeline(tmp_path)
    source = SimulatedSource.from_file(CONV)
    events = list(pipe.process(source, on_guidance=lambda g: Outcome(action=HandlerAction.ACCEPTED)))
    for e in events:
        if e.utterance.speaker == Speaker.HANDLER:
            assert e.guidance is None


def test_guidance_is_advisory_only(tmp_path):
    pipe = _pipeline(tmp_path)
    source = SimulatedSource.from_file(CONV)
    flagged = [e for e in pipe.process(source, on_guidance=lambda g: Outcome())
               if e.guidance is not None]
    assert flagged
    for e in flagged:
        assert e.guidance.advisory_only is True
        assert e.record is not None


def test_neutral_conversation_produces_no_records(tmp_path):
    pipe = _pipeline(tmp_path)
    source = SimulatedSource.from_file(ROOT / "data" / "transcripts" / "CONV-003.json")
    events = list(pipe.process(source))
    assert all(e.guidance is None for e in events)
    assert pipe.evidence.read_all() == []
