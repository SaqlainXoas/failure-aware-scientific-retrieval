import pytest

from retrieval.evaluate import evaluate_query, mrr_at_k, ndcg_at_k, recall_at_k

RANKED = ["d1", "d2", "d3", "d4", "d5"]
QRELS_BINARY = {"d2": 1, "d4": 1}
RELEVANT_BINARY = {"d2", "d4"}


def test_recall_at_k_hand_calculated():
    assert recall_at_k(RANKED, RELEVANT_BINARY, 3) == pytest.approx(0.5)
    assert recall_at_k(RANKED, RELEVANT_BINARY, 5) == pytest.approx(1.0)


def test_mrr_at_k_hand_calculated():
    assert mrr_at_k(RANKED, RELEVANT_BINARY, 10) == pytest.approx(0.5)


def test_ndcg_at_k_binary_hand_calculated():
    assert ndcg_at_k(RANKED, QRELS_BINARY, 5) == pytest.approx(0.6509209298071326)


def test_ndcg_at_k_graded_relevance():
    # Verifies gain calculation uses the actual grade, not just membership.
    qrels = {"d1": 2, "d3": 1}
    ranked = ["d1", "d2", "d3"]
    assert ndcg_at_k(ranked, qrels, 3) == pytest.approx(0.9502344167898356)


def test_zero_grade_qrel_is_not_relevant():
    # A grade-0 entry (annotated, judged not relevant) must contribute nothing —
    # evaluate_query's relevant-doc derivation and nDCG's gain must both ignore it.
    with_zero_grade = {"d2": 0, "d4": 1}
    without_entry = {"d4": 1}
    assert evaluate_query(RANKED, with_zero_grade) == evaluate_query(RANKED, without_entry)


def test_no_relevant_docs_returns_zero_no_crash():
    assert recall_at_k(RANKED, set(), 10) == 0.0
    assert mrr_at_k(RANKED, set(), 10) == 0.0
    assert ndcg_at_k(RANKED, {}, 10) == 0.0
    assert ndcg_at_k(RANKED, {"d1": 0, "d2": 0}, 10) == 0.0


def test_fewer_ranked_docs_than_k_does_not_crash():
    short_ranking = ["d2", "d1"]  # relevant doc already ranked first -> a perfect ranking
    assert recall_at_k(short_ranking, {"d2"}, 50) == pytest.approx(1.0)
    assert mrr_at_k(short_ranking, {"d2"}, 50) == pytest.approx(1.0)
    assert ndcg_at_k(short_ranking, {"d2": 1}, 50) == pytest.approx(1.0)


def test_perfect_ranking_scores_one():
    ranked = ["d1", "d2"]
    qrels = {"d1": 1, "d2": 1}
    relevant = {"d1", "d2"}
    assert recall_at_k(ranked, relevant, 2) == pytest.approx(1.0)
    assert mrr_at_k(ranked, relevant, 2) == pytest.approx(1.0)
    assert ndcg_at_k(ranked, qrels, 2) == pytest.approx(1.0)


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_raises(k):
    with pytest.raises(ValueError):
        recall_at_k(RANKED, RELEVANT_BINARY, k)
    with pytest.raises(ValueError):
        mrr_at_k(RANKED, RELEVANT_BINARY, k)
    with pytest.raises(ValueError):
        ndcg_at_k(RANKED, QRELS_BINARY, k)


def test_evaluate_query_returns_all_metrics():
    result = evaluate_query(RANKED, QRELS_BINARY)
    assert set(result) == {"recall@5", "recall@10", "recall@50", "mrr@10", "ndcg@10"}
    assert result["recall@5"] == pytest.approx(1.0)
    assert result["mrr@10"] == pytest.approx(0.5)
