from guardiancx.rag.evaluation import evaluate_rag, load_testset


def test_testset_loads():
    ts = load_testset()
    assert len(ts) >= 20
    assert all({"query", "driver", "expected"} <= set(row) for row in ts)


def test_rag_retrieval_quality_meets_bar():
    """Semantic retrieval should reliably surface the correct clause. Guards
    against the class of regression where bereavement returned unrelated clauses."""
    report = evaluate_rag(load_testset(), k=3)
    assert report.hit_rate >= 0.85, f"hit-rate too low: {report.hit_rate}"
    assert report.mrr >= 0.7, f"MRR too low: {report.mrr}"


def test_bereavement_specifically_retrieves_vp_l1():
    ts = [{"query": "my husband passed away last month", "driver": "life_events",
           "expected": ["VP-L1"]}]
    report = evaluate_rag(ts, k=3)
    assert report.results[0].hit, report.results[0].retrieved
