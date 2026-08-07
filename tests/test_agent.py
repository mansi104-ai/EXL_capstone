from pathlib import Path

from vca.agent import CareAgent, Step
from vca.classifier.keyword import KeywordClassifier
from vca.evidence.store import EvidenceStore
from vca.guidance.advisor import HandlerAdvisor
from vca.pipeline import VCAPipeline
from vca.rag.policy import parse_policy
from vca.rag.retriever import KeywordRetriever
from vca.schemas import HandlerAction, Outcome, Speaker, Utterance

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "data" / "policy" / "vulnerability_policy.md"


def _agent(tmp_path):
    clf = KeywordClassifier(threshold=0.5)
    advisor = HandlerAdvisor(KeywordRetriever(parse_policy(POLICY), top_k=1))
    evidence = EvidenceStore(tmp_path / "ev.jsonl")
    return CareAgent(VCAPipeline(clf, advisor, evidence)), evidence


def _cust(text):
    return Utterance(conversation_id="C", turn_index=0, speaker=Speaker.CUSTOMER, text=text)


def test_assess_does_not_write_evidence(tmp_path):
    agent, evidence = _agent(tmp_path)
    trace = agent.run_turn(_cust("My husband passed away last month."))
    assert trace.flagged
    assert evidence.read_all() == []  # assessment alone writes nothing


def test_trace_has_expected_steps_when_flagged(tmp_path):
    agent, _ = _agent(tmp_path)
    trace = agent.run_turn(_cust("I can't afford this and I have no savings."))
    steps = [s.step for s in trace.steps]
    assert Step.PERCEIVE in steps
    assert Step.ASSESS in steps
    assert Step.RETRIEVE in steps
    assert Step.ADVISE in steps
    assert Step.DECIDE in steps


def test_handler_turn_not_assessed(tmp_path):
    agent, _ = _agent(tmp_path)
    u = Utterance(conversation_id="C", turn_index=0, speaker=Speaker.HANDLER, text="Hello")
    trace = agent.run_turn(u)
    assert not trace.event.assessed
    assert not trace.flagged


def test_record_after_decision_writes_evidence(tmp_path):
    agent, evidence = _agent(tmp_path)
    trace = agent.run_turn(_cust("My husband passed away last month."))
    ev = trace.event
    agent.pipeline.record_outcome(ev.detection, ev.guidance,
                                  Outcome(action=HandlerAction.ACCEPTED))
    records = evidence.read_all()
    assert len(records) == 1
    assert records[0]["outcome"]["action"] == "accepted"
