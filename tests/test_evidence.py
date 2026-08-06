import json

from vca.classifier.keyword import KeywordClassifier
from vca.evidence.store import EvidenceStore
from vca.guidance.advisor import HandlerAdvisor
from vca.rag.policy import parse_policy
from vca.rag.retriever import KeywordRetriever
from vca.schemas import HandlerAction, Outcome
from pathlib import Path

POLICY = Path(__file__).resolve().parents[1] / "data" / "policy" / "vulnerability_policy.md"


def _advisor():
    return HandlerAdvisor(KeywordRetriever(parse_policy(POLICY), top_k=1))


def _record(store, text):
    clf = KeywordClassifier(threshold=0.5)
    det = clf.detect("C", 0, text)
    guid = _advisor().advise(det)
    return store.append(det, guid, Outcome(action=HandlerAction.ACCEPTED))


def test_chain_valid_after_appends(tmp_path):
    store = EvidenceStore(tmp_path / "ev.jsonl")
    _record(store, "My husband passed away.")
    _record(store, "I can't afford the payment, no savings.")
    ok, bad = store.verify_chain()
    assert ok and bad is None
    assert len(store.read_all()) == 2


def test_tampering_breaks_chain(tmp_path):
    path = tmp_path / "ev.jsonl"
    store = EvidenceStore(path)
    _record(store, "My husband passed away.")
    _record(store, "I lost my job on Friday.")

    # Tamper with the first record's outcome.
    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    rec["outcome"]["action"] = "dismissed"
    lines[0] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, bad = store.verify_chain()
    assert not ok
    assert bad is not None
