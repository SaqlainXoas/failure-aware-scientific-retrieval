import math

import pytest

from retrieval.confidence import (
    COMMON_FEATURES,
    HYBRID_FEATURES,
    confidence_metrics,
    extract_features,
    fit_calibrator,
    predict_proba,
    raw_baseline_scores,
    risk_coverage_curve,
    select_thresholds,
    selective_results_at_coverage,
)


def _rows(query_id, doc_scores):
    return [
        {"query_id": query_id, "doc_id": doc_id, "rank": rank, "score": score}
        for rank, (doc_id, score) in enumerate(doc_scores, start=1)
    ]


def test_extract_features_schema():
    first_stage = _rows("q1", [("a", 10.0), ("b", 8.0), ("c", 2.0)])
    reranked = _rows("q1", [("a", 3.0), ("b", 1.0), ("c", 0.5)])

    features = extract_features(first_stage, reranked, rerank_depth=50)

    assert set(features) == {"q1"}
    assert set(features["q1"]) == set(COMMON_FEATURES)


def test_reranker_score_features():
    first_stage = _rows("q1", [("a", 1.0), ("b", 0.5)])
    reranked = _rows("q1", [("a", 4.0), ("b", 2.0), ("c", 1.0), ("d", 1.0), ("e", 1.0), ("f", 0.0)])

    features = extract_features(first_stage, reranked, rerank_depth=50)["q1"]

    assert features["reranker_top1_score"] == pytest.approx(4.0)
    assert features["reranker_top1_top2_margin"] == pytest.approx(2.0)
    top5 = [4.0, 2.0, 1.0, 1.0, 1.0]
    assert features["reranker_top5_mean"] == pytest.approx(sum(top5) / 5)
    mean = sum(top5) / 5
    expected_std = math.sqrt(sum((x - mean) ** 2 for x in top5) / 5)
    assert features["reranker_top5_std"] == pytest.approx(expected_std)


def test_reranker_features_fewer_than_five_candidates_no_nan():
    first_stage = _rows("q1", [("a", 1.0)])
    reranked = _rows("q1", [("a", 4.0)])

    features = extract_features(first_stage, reranked, rerank_depth=50)["q1"]

    assert features["reranker_top1_score"] == pytest.approx(4.0)
    assert features["reranker_top1_top2_margin"] == 0.0
    assert features["reranker_top5_mean"] == pytest.approx(4.0)
    assert features["reranker_top5_std"] == 0.0
    for value in features.values():
        assert not math.isnan(value)
        assert not math.isinf(value)


def test_first_stage_normalization():
    first_stage = _rows("q1", [("a", 10.0), ("b", 6.0), ("c", 0.0)])
    reranked = _rows("q1", [("a", 1.0), ("b", 0.5), ("c", 0.1)])

    features = extract_features(first_stage, reranked, rerank_depth=50)["q1"]

    assert features["first_stage_top1_score_norm"] == pytest.approx(1.0)
    assert features["first_stage_top1_top2_margin_norm"] == pytest.approx(0.4)


def test_first_stage_normalization_constant_scores_returns_midpoint():
    first_stage = _rows("q1", [("a", 5.0), ("b", 5.0)])
    reranked = _rows("q1", [("a", 1.0), ("b", 1.0)])

    features = extract_features(first_stage, reranked, rerank_depth=50)["q1"]

    assert features["first_stage_top1_score_norm"] == pytest.approx(0.5)
    assert features["first_stage_top1_top2_margin_norm"] == pytest.approx(0.0)


def test_rank_correlation_perfect_agreement():
    first_stage = _rows("q1", [("a", 3.0), ("b", 2.0), ("c", 1.0)])
    reranked = _rows("q1", [("a", 9.0), ("b", 5.0), ("c", 1.0)])

    features = extract_features(first_stage, reranked, rerank_depth=50)["q1"]

    assert features["first_stage_rerank_rank_correlation"] == pytest.approx(1.0)


def test_rank_correlation_full_reversal():
    first_stage = _rows("q1", [("a", 3.0), ("b", 2.0), ("c", 1.0)])
    reranked = _rows("q1", [("c", 9.0), ("b", 5.0), ("a", 1.0)])

    features = extract_features(first_stage, reranked, rerank_depth=50)["q1"]

    assert features["first_stage_rerank_rank_correlation"] == pytest.approx(-1.0)


def test_rank_correlation_fewer_than_two_shared_docs_is_zero():
    first_stage = _rows("q1", [("a", 3.0)])
    reranked = _rows("q1", [("a", 9.0)])

    features = extract_features(first_stage, reranked, rerank_depth=50)["q1"]

    assert features["first_stage_rerank_rank_correlation"] == 0.0


def test_top1_in_reranked_top3():
    first_stage = _rows("q1", [("a", 3.0), ("b", 2.0)])
    reranked_in = _rows("q1", [("z", 9.0), ("a", 5.0), ("y", 1.0)])
    reranked_out = _rows("q1", [("z", 9.0), ("y", 5.0), ("x", 1.0), ("a", 0.1)])

    assert extract_features(first_stage, reranked_in, rerank_depth=50)["q1"]["first_stage_top1_in_reranked_top3"] == 1.0
    assert extract_features(first_stage, reranked_out, rerank_depth=50)["q1"]["first_stage_top1_in_reranked_top3"] == 0.0


def test_hybrid_overlap_feature():
    first_stage = _rows("q1", [("a", 3.0)])
    reranked = _rows("q1", [("a", 9.0)])
    bm25_rows = _rows("q1", [(f"d{i}", float(11 - i)) for i in range(1, 12)])  # d1..d11, top10 = d1..d10
    dense_rows = _rows("q1", [("d5", 9.0), ("d6", 8.0), ("d99", 1.0)])  # top10 = d5, d6, d99

    features = extract_features(
        first_stage, reranked, rerank_depth=50, raw_rows={"bm25": bm25_rows, "dense": dense_rows}
    )["q1"]

    assert set(features) == set(COMMON_FEATURES) | set(HYBRID_FEATURES)
    assert features["hybrid_bm25_dense_top10_overlap"] == pytest.approx(2 / 10)


def test_no_qrel_derived_features():
    all_features = set(COMMON_FEATURES) | set(HYBRID_FEATURES)
    forbidden_terms = {"gold", "qrel", "relevant", "success", "label"}
    for name in all_features:
        assert not any(term in name for term in forbidden_terms), name


def test_raw_baseline_scores_is_top1_reranker_score():
    reranked = _rows("q1", [("a", 4.0), ("b", 1.0)]) + _rows("q2", [("c", 2.0)])

    baseline = raw_baseline_scores(reranked)

    assert baseline == {"q1": 4.0, "q2": 2.0}


def _synthetic_dataset(n=40, seed=0):
    import random

    rng = random.Random(seed)
    features_by_query = {}
    labels_by_query = {}
    for i in range(n):
        qid = f"q{i}"
        top1 = rng.uniform(0, 1)
        label = top1 > 0.5
        features_by_query[qid] = {
            "reranker_top1_score": top1,
            "reranker_top1_top2_margin": rng.uniform(0, 0.3),
            "reranker_top5_mean": top1 * 0.8,
            "reranker_top5_std": rng.uniform(0, 0.1),
            "first_stage_top1_score_norm": rng.uniform(0, 1),
            "first_stage_top1_top2_margin_norm": rng.uniform(0, 0.3),
            "first_stage_rerank_rank_correlation": rng.uniform(-1, 1),
            "first_stage_top1_in_reranked_top3": float(rng.random() > 0.3),
        }
        labels_by_query[qid] = label
    return features_by_query, labels_by_query


def test_fit_calibrator_only_uses_train_data(monkeypatch):
    train_features, train_labels = _synthetic_dataset(seed=1)
    dev_features, dev_labels = _synthetic_dataset(seed=2)

    import sklearn.pipeline

    original_fit = sklearn.pipeline.Pipeline.fit
    seen_lengths = []

    def spy_fit(self, X, y, **kwargs):
        seen_lengths.append(len(X))
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(sklearn.pipeline.Pipeline, "fit", spy_fit)

    calibrator = fit_calibrator(train_features, train_labels, COMMON_FEATURES)

    assert seen_lengths == [len(train_features)]

    dev_probs = predict_proba(calibrator, dev_features, COMMON_FEATURES)
    assert set(dev_probs) == set(dev_features)
    assert all(0.0 <= p <= 1.0 for p in dev_probs.values())


def test_predict_proba_does_not_refit():
    train_features, train_labels = _synthetic_dataset(seed=1)
    calibrator = fit_calibrator(train_features, train_labels, COMMON_FEATURES)
    coef_before = calibrator.named_steps["classifier"].coef_.copy()

    dev_features, dev_labels = _synthetic_dataset(seed=2)
    predict_proba(calibrator, dev_features, COMMON_FEATURES)

    coef_after = calibrator.named_steps["classifier"].coef_
    assert (coef_before == coef_after).all()


def test_confidence_metrics_single_class_returns_none_not_error():
    probs = {"q1": 0.9, "q2": 0.8}
    labels = {"q1": True, "q2": True}

    metrics = confidence_metrics(probs, labels)

    assert metrics["auroc"] is None
    assert metrics["auprc"] is None
    assert metrics["brier"] is not None


def test_select_thresholds_full_coverage_is_min_score():
    probs = {"q1": 0.9, "q2": 0.5, "q3": 0.1}
    labels = {"q1": True, "q2": False, "q3": True}

    thresholds = select_thresholds(probs, labels, coverage_levels=(1.0,))

    assert thresholds["1.0"] == pytest.approx(0.1)


def test_selective_results_at_coverage_keeps_most_confident():
    probs = {"q1": 0.9, "q2": 0.5, "q3": 0.1}
    labels = {"q1": True, "q2": False, "q3": True}

    results = selective_results_at_coverage(probs, labels, coverage_levels=(1 / 3,))

    assert results[str(1 / 3)]["n_kept"] == 1
    assert results[str(1 / 3)]["success_rate"] == 1.0


def test_risk_coverage_curve_monotonic_coverage():
    probs = {"q1": 0.9, "q2": 0.5, "q3": 0.1}
    labels = {"q1": True, "q2": False, "q3": True}

    curve = risk_coverage_curve(probs, labels)

    assert [p["coverage"] for p in curve] == [pytest.approx(1 / 3), pytest.approx(2 / 3), pytest.approx(1.0)]
    assert curve[0]["risk"] == 0.0
    assert curve[-1]["risk"] == pytest.approx(1 / 3)
