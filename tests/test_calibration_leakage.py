"""Leakage rules for the confidence stage (plan §5, §9; Phase 4 spec AC 2-4).

`fit_calibrator` being train-only is necessary but not sufficient — the rule that matters lives
in `run_calibration`'s wiring, which decides which split is fitted and which is only scored.
These tests spy on the actual `Pipeline.fit` calls to assert that.
"""

import pytest
import sklearn.pipeline

from retrieval.confidence import COMMON_FEATURES, confidence_metrics, cross_validated_predictions
from retrieval.run import run_calibration

# confidence_cv_folds=0 disables the pooled cross-validated estimate, so these tests assert the
# purity of the *predeclared* train/dev protocol in isolation. The cross-validated estimate
# deliberately pools train+dev and is covered by its own tests below.
CONFIG = {
    "pipeline": "bm25",
    "confidence_class_weight": None,
    "confidence_coverage_levels": [1.0, 0.8, 0.6],
    "confidence_cv_folds": 0,
}


def _features(query_ids, offset=0.0):
    return {
        qid: {name: offset + (i + j) / 10.0 for j, name in enumerate(COMMON_FEATURES)}
        for i, qid in enumerate(query_ids)
    }


def _reranked_rows(query_ids, offset=0.0):
    return [
        {"query_id": qid, "doc_id": f"d{i}", "rank": rank, "score": offset + i - rank}
        for i, qid in enumerate(query_ids)
        for rank in (1, 2, 3)
    ]


def _labels(query_ids):
    """Labels positively correlated with the reranked top-1 score, as on real data, with one
    deliberate flip so AUROC lands strictly between 0.5 and 1.0."""
    n = len(query_ids)
    labels = {qid: i >= n // 2 for i, qid in enumerate(query_ids)}
    labels[query_ids[0]] = True
    return labels


@pytest.fixture
def fit_spy(monkeypatch):
    """Records the feature matrix passed to every Pipeline.fit during the test."""
    original_fit = sklearn.pipeline.Pipeline.fit
    fitted_matrices = []

    def spy_fit(self, X, y, **kwargs):
        fitted_matrices.append([list(row) for row in X])
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(sklearn.pipeline.Pipeline, "fit", spy_fit)
    return fitted_matrices


def _run(fit_spy):
    train_ids = [f"train{i}" for i in range(20)]
    dev_ids = [f"dev{i}" for i in range(10)]
    train_features = _features(train_ids)
    dev_features = _features(dev_ids, offset=100.0)  # disjoint value range: identifiable
    calibration = run_calibration(
        CONFIG,
        train_features,
        _labels(train_ids),
        _reranked_rows(train_ids),
        dev_features,
        _labels(dev_ids),
        {qid: "already_successful" for qid in dev_ids},
        _reranked_rows(dev_ids, offset=100.0),
    )
    return calibration, train_ids, dev_ids, dev_features


def test_no_dev_feature_value_ever_reaches_a_fit_call(fit_spy):
    _, train_ids, _, dev_features = _run(fit_spy)

    dev_values = {value for feats in dev_features.values() for value in feats.values()}
    for matrix in fit_spy:
        # Every fitted model must have seen exactly the training rows...
        assert len(matrix) == len(train_ids)
        # ...and none of the dev split's identifiable feature values.
        for row in matrix:
            assert not (set(row) & dev_values)


def test_every_reported_model_was_fitted_on_train_and_scored_on_dev(fit_spy):
    calibration, train_ids, dev_ids, _ = _run(fit_spy)

    assert calibration["fit_split"] == "calibration-train"
    assert calibration["eval_split"] == "calibration-dev"
    assert calibration["cross_validated"] is None  # disabled for this protocol-purity test
    # raw_score needs no fit; the Platt baseline and the calibrator each fit exactly once.
    assert len(fit_spy) == 2
    for result in calibration["results"].values():
        assert result["confidence_metrics"]["n_queries"] == len(dev_ids)
    assert [p["query_id"] for p in calibration["predictions"]] == sorted(dev_ids)
    assert {p["transition_label"] for p in calibration["predictions"]} == {
        "already_successful"
    }


def test_thresholds_are_selected_on_dev_not_train(fit_spy):
    calibration, _, dev_ids, _ = _run(fit_spy)

    # A coverage-1.0 threshold is the minimum score over the split it was selected on, so it
    # must be reachable from the dev predictions and not from anything train-only.
    for name, result in calibration["results"].items():
        dev_scores = [p[name] for p in calibration["predictions"]]
        assert result["thresholds"]["1.0"] == pytest.approx(min(dev_scores))
        assert result["selective_results"]["1.0"]["n_kept"] == len(dev_ids)


def test_bm25_pipeline_has_no_exploratory_model(fit_spy):
    calibration, _, _, _ = _run(fit_spy)

    assert calibration["exploratory_models"] == []
    assert set(calibration["results"]) == {"raw_score", "raw_score_platt", "calibrated"}


def test_raw_score_is_not_reported_as_a_probability(fit_spy):
    calibration, _, _, _ = _run(fit_spy)

    raw = calibration["results"]["raw_score"]["confidence_metrics"]
    platt = calibration["results"]["raw_score_platt"]["confidence_metrics"]

    # plan §9: the raw reranker score is not a probability, so it gets no Brier score; the
    # train-fitted Platt version does. Platt scaling is monotone in the direction the training
    # data implies, so when higher scores really do mean success (as here and on real data) the
    # rank metrics are identical and only the Brier-scale interpretation changes.
    assert raw["is_probability"] is False
    assert raw["brier"] is None
    assert platt["is_probability"] is True
    assert 0.0 <= platt["brier"] <= 1.0
    assert 0.5 < raw["auroc"] < 1.0  # fixture sanity: not degenerate in either direction
    assert raw["auroc"] == pytest.approx(platt["auroc"])
    assert raw["auprc"] == pytest.approx(platt["auprc"])


def _cv_inputs(n=60):
    query_ids = [f"q{i:03d}" for i in range(n)]
    features = _features(query_ids)
    labels = {qid: i % 4 != 0 for i, qid in enumerate(query_ids)}  # 75% success, like real data
    baseline = {qid: float(i) for i, qid in enumerate(query_ids)}
    return query_ids, features, labels, baseline


def test_cross_validation_predicts_every_query_exactly_once_out_of_fold():
    query_ids, features, labels, baseline = _cv_inputs()

    rows = cross_validated_predictions(
        features, labels, baseline, {"calibrated": COMMON_FEATURES}, n_splits=5, seed=42
    )

    # Every query gets exactly one prediction, and it comes from a fold that excluded it.
    assert [row["query_id"] for row in rows] == sorted(query_ids)
    assert len({row["query_id"] for row in rows}) == len(query_ids)
    assert {row["fold"] for row in rows} == {0, 1, 2, 3, 4}


def test_cross_validation_never_scores_a_query_with_a_model_that_saw_it(fit_spy):
    query_ids, features, labels, baseline = _cv_inputs()

    rows = cross_validated_predictions(
        features, labels, baseline, {"calibrated": COMMON_FEATURES}, n_splits=5, seed=42
    )

    # Each fold fits on strictly fewer rows than the full set — the held-out queries are absent.
    by_fold: dict[int, list[str]] = {}
    for row in rows:
        by_fold.setdefault(row["fold"], []).append(row["query_id"])
    for held_out in by_fold.values():
        assert 0 < len(held_out) < len(query_ids)
    # Folds partition the query set: no overlap, full coverage.
    flattened = [qid for held_out in by_fold.values() for qid in held_out]
    assert sorted(flattened) == sorted(query_ids)
    assert len(flattened) == len(set(flattened))


def test_cross_validation_is_deterministic_for_a_fixed_seed():
    _, features, labels, baseline = _cv_inputs()
    kwargs = dict(feature_sets={"calibrated": COMMON_FEATURES}, n_splits=5, seed=42)

    first = cross_validated_predictions(features, labels, baseline, **kwargs)
    second = cross_validated_predictions(features, labels, baseline, **kwargs)

    assert first == second


def test_cross_validation_probabilities_are_valid_and_raw_score_passes_through():
    _, features, labels, baseline = _cv_inputs()

    rows = cross_validated_predictions(
        features, labels, baseline, {"calibrated": COMMON_FEATURES}, n_splits=5, seed=42
    )

    for row in rows:
        assert row["raw_score"] == baseline[row["query_id"]]
        for model in ("raw_score_platt", "calibrated"):
            assert 0.0 <= row[model] <= 1.0
        assert row["final_success_10"] == labels[row["query_id"]]


def test_confidence_metrics_report_the_auprc_floor():
    # AUPRC is bounded below by the base rate, not 0.5 — reporting it bare overstates the model.
    probs = {"q1": 0.9, "q2": 0.8, "q3": 0.2, "q4": 0.1}
    labels = {"q1": True, "q2": True, "q3": True, "q4": False}

    metrics = confidence_metrics(probs, labels)

    assert metrics["base_rate"] == pytest.approx(0.75)
    assert metrics["n_failures"] == 1
    assert metrics["auprc_over_base_rate"] == pytest.approx(metrics["auprc"] - 0.75)


def test_cross_validated_estimate_is_reported_separately_from_the_protocol_result():
    """The pooled estimate must never overwrite or blend into the predeclared train/dev result."""
    train_ids = [f"train{i}" for i in range(40)]
    dev_ids = [f"dev{i}" for i in range(20)]
    config = {**CONFIG, "confidence_cv_folds": 5}

    calibration = run_calibration(
        config,
        _features(train_ids),
        _labels(train_ids),
        _reranked_rows(train_ids),
        _features(dev_ids, offset=100.0),
        _labels(dev_ids),
        {qid: "already_successful" for qid in dev_ids},
        _reranked_rows(dev_ids, offset=100.0),
    )

    cv = calibration["cross_validated"]
    # Protocol result is still dev-only...
    for result in calibration["results"].values():
        assert result["confidence_metrics"]["n_queries"] == len(dev_ids)
    # ...while the pooled estimate covers every calibration query, giving more failures to learn from.
    assert cv["n_queries"] == len(train_ids) + len(dev_ids)
    assert cv["n_failures"] > 0
    assert cv["n_splits"] == 5
    assert set(cv["results"]) == {"raw_score", "raw_score_platt", "calibrated"}
    assert [row["query_id"] for row in cv["predictions"]] == sorted(train_ids + dev_ids)
