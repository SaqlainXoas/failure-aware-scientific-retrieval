import pytest

from retrieval.evaluate import (
    CATEGORIES,
    candidate_success_50,
    decomposition_metrics,
    final_success_10,
    label_transition,
)


def test_already_successful():
    gold = {"g"}
    first_stage = ["g", "a", "b"]
    reranked = ["a", "g", "b"]
    assert label_transition(gold, first_stage, reranked, candidate_depth=50) == "already_successful"


def test_rescued_by_reranker():
    gold = {"g"}
    first_stage = ["a", "b", "c", "d", "e", "f", "h", "i", "j", "k", "g"]  # g at rank 11: outside top 10
    reranked = ["g", "a", "b", "c", "d", "e", "f", "h", "i", "j", "k"]  # g at rank 1: inside top 10
    assert label_transition(gold, first_stage, reranked, candidate_depth=50) == "rescued_by_reranker"


def test_degraded_by_reranker():
    gold = {"g"}
    first_stage = ["g", "a", "b", "c", "d", "e", "f", "h", "i", "j"]  # g at rank 1: inside top 10
    reranked = ["a", "b", "c", "d", "e", "f", "h", "i", "j", "k", "g"]  # g at rank 11: outside top 10
    assert label_transition(gold, first_stage, reranked, candidate_depth=50) == "degraded_by_reranker"


def test_unchanged_failure():
    gold = {"g"}
    first_stage = ["a", "b", "c", "d", "e", "f", "h", "i", "j", "k", "g"]  # g at rank 11
    reranked = ["b", "a", "c", "d", "e", "f", "h", "i", "j", "k", "g"]  # g at rank 11
    assert label_transition(gold, first_stage, reranked, candidate_depth=50) == "unchanged_failure"


def test_no_opportunity():
    gold = {"missing"}
    first_stage = ["a", "b", "c"]
    reranked = ["a", "b", "c"]
    assert label_transition(gold, first_stage, reranked, candidate_depth=50) == "no_opportunity"


def test_candidate_success_and_final_success_helpers():
    assert candidate_success_50({"g"}, ["a", "g", "b"]) is True
    assert candidate_success_50({"g"}, ["a", "b"]) is False
    assert final_success_10({"g"}, ["g", "a"]) is True
    assert final_success_10({"g"}, ["a", "b"]) is False


def test_labels_partition_all_queries():
    cases = {
        "q_already": (
            {"g"},
            ["g", "a", "b"],
            ["a", "g", "b"],
        ),
        "q_rescued": (
            {"g"},
            ["a", "b", "c", "d", "e", "f", "h", "i", "j", "k", "g"],
            ["g", "a", "b", "c", "d", "e", "f", "h", "i", "j", "k"],
        ),
        "q_degraded": (
            {"g"},
            ["g", "a", "b", "c", "d", "e", "f", "h", "i", "j"],
            ["a", "b", "c", "d", "e", "f", "h", "i", "j", "k", "g"],
        ),
        "q_unchanged": (
            {"g"},
            ["a", "b", "c", "d", "e", "f", "h", "i", "j", "k", "g"],
            ["b", "a", "c", "d", "e", "f", "h", "i", "j", "k", "g"],
        ),
        "q_no_opportunity": ({"missing"}, ["a", "b", "c"], ["a", "b", "c"]),
    }
    labels = [
        label_transition(gold, first_stage, reranked, candidate_depth=50)
        for gold, first_stage, reranked in cases.values()
    ]
    assert len(labels) == len(cases)
    assert set(labels) == set(CATEGORIES)


def test_candidate_failure_never_rescued():
    gold = {"missing"}
    first_stage = ["a"] * 1  # gold entirely outside top-50
    reranked = ["a"]
    label = label_transition(gold, first_stage, reranked, candidate_depth=50)
    assert label == "no_opportunity"
    assert label != "rescued_by_reranker"


def test_decomposition_metrics_rates():
    labels = (
        ["already_successful"] * 2
        + ["rescued_by_reranker"] * 1
        + ["degraded_by_reranker"] * 1
        + ["unchanged_failure"] * 1
        + ["no_opportunity"] * 5
    )
    metrics = decomposition_metrics(labels)

    n = 10
    assert metrics["candidate_set_failure_rate"] == pytest.approx(5 / n)
    assert metrics["reranking_failure_rate"] == pytest.approx(2 / n)
    assert metrics["final_success_rate"] == pytest.approx(3 / n)
    assert metrics["opportunity_rate"] == pytest.approx(5 / n)
    assert metrics["rescue_rate"] == pytest.approx(1 / n)
    assert metrics["degradation_rate"] == pytest.approx(1 / n)
    assert metrics["conditional_conversion_rate"] == pytest.approx(3 / 5)


def test_decomposition_metrics_zero_opportunity_edge_case():
    labels = ["no_opportunity"] * 4
    metrics = decomposition_metrics(labels)
    assert metrics["conditional_conversion_rate"] == 0.0


def test_decomposition_counts_sum_to_query_count():
    # Transition labels must partition the query set, and the saved
    # artifact has to make that checkable rather than only exposing rates.
    labels = (
        ["already_successful"] * 7
        + ["rescued_by_reranker"] * 3
        + ["degraded_by_reranker"] * 2
        + ["unchanged_failure"] * 4
        + ["no_opportunity"] * 9
    )
    metrics = decomposition_metrics(labels)

    assert set(metrics["counts"]) == set(CATEGORIES)
    assert sum(metrics["counts"].values()) == len(labels)
    assert metrics["n_queries"] == len(labels)
    assert metrics["candidate_set_failure_rate"] + metrics["reranking_failure_rate"] + metrics[
        "final_success_rate"
    ] == pytest.approx(1.0)


def test_share_of_failures_from_candidate_set():
    # The headline quantity of the study: of everything that failed, how much the reranker
    # never had a chance to fix. 5 no_opportunity out of 5 + 1 + 2 = 8 total failures.
    labels = (
        ["no_opportunity"] * 5
        + ["degraded_by_reranker"] * 1
        + ["unchanged_failure"] * 2
        + ["already_successful"] * 10
    )
    metrics = decomposition_metrics(labels)

    assert metrics["n_final_failures"] == 8
    assert metrics["share_of_failures_from_candidate_set"] == pytest.approx(5 / 8)


def test_share_of_failures_is_zero_when_nothing_failed():
    metrics = decomposition_metrics(["already_successful"] * 3)

    assert metrics["n_final_failures"] == 0
    assert metrics["share_of_failures_from_candidate_set"] == 0.0


def test_decomposition_rejects_unknown_transition_label():
    with pytest.raises(ValueError, match="Unknown transition"):
        decomposition_metrics(["already_successful", "not_a_real_transition"])
