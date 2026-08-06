from vca.classifier.keyword import KeywordClassifier
from vca.schemas import Driver


def test_keyword_detects_bereavement():
    clf = KeywordClassifier(threshold=0.5)
    d = clf.detect("C", 0, "My husband passed away last month.")
    assert Driver.LIFE_EVENTS in d.triggered_drivers


def test_keyword_detects_multiple_drivers():
    clf = KeywordClassifier(threshold=0.5)
    d = clf.detect("C", 0, "I struggle with my memory since the stroke and my daughter usually helps me.")
    assert Driver.HEALTH in d.triggered_drivers
    assert Driver.CAPABILITY in d.triggered_drivers


def test_neutral_utterance_triggers_nothing():
    clf = KeywordClassifier(threshold=0.5)
    d = clf.detect("C", 0, "Can I order a new debit card please?")
    assert not d.any_triggered


def test_scores_within_bounds():
    clf = KeywordClassifier(threshold=0.5)
    d = clf.detect("C", 0, "I can't afford this, I have no savings.")
    for s in d.scores:
        assert 0.0 <= s.score <= 1.0
