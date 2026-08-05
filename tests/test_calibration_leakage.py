"""Leakage rules for the confidence stage (plan §5, §9; Phase 4 spec AC 2-4).

`fit_calibrator` being train-only is necessary but not sufficient — the rule that matters lives
in `run_calibration`'s wiring, which decides which split is fitted and which is only scored.
These tests spy on the actual `Pipeline.fit` calls to assert that.
"""

import pytest
import sklearn.pipeline

from retrieval.confidence import COMMON_FEATURES
from retrieval.run import run_calibration

CONFIG = {
    "pipeline": "bm25",
    "confidence_class_weight": None,
    "confidence_coverage_levels": [1.0, 0.8, 0.6],
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
