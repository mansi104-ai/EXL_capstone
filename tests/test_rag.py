from pathlib import Path

from vca.rag.policy import parse_policy
from vca.rag.retriever import KeywordRetriever
from vca.schemas import Driver

POLICY = Path(__file__).resolve().parents[1] / "data" / "policy" / "vulnerability_policy.md"


def test_policy_parses_all_drivers():
    sections = parse_policy(POLICY)
    drivers = {s.driver for s in sections}
    assert drivers == set(Driver)
    assert len(sections) >= 12


def test_no_horizontal_rule_leaks_into_body():
    for s in parse_policy(POLICY):
        assert "---" not in s.body


def test_retrieval_is_driver_scoped():
    retriever = KeywordRetriever(parse_policy(POLICY), top_k=1)
    res = retriever.retrieve("husband passed away", driver=Driver.LIFE_EVENTS)
    assert res
    assert res[0].driver == Driver.LIFE_EVENTS
    assert res[0].policy_reference == "VP-L1"
