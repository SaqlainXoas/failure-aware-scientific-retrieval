"""Phase 5 bootstrap, plotting, selection, provenance, and leakage checks."""

import json
from collections import Counter

import numpy as np
import pytest

import retrieval.analysis as analysis_module
from retrieval.analysis import (
    build_bootstrap_artifact,
    load_failure_case_data,
    render_bootstrap_markdown,
    select_failure_cases,
)
from retrieval.confidence import (
    bootstrap_mean_comparison,
    bootstrap_score_comparison,
    paired_query_bootstrap,
)
from retrieval.plots import (
    plot_failure_breakdown,
    plot_first_stage_performance,
    plot_hybrid_risk_coverage,
    plot_reliability,
    plot_reranking_transitions,
    reliability_bins,
)


def test_paired_bootstrap_is_deterministic_and_shares_query_samples():
    calls_a = []
    calls_b = []

    def statistic_a(query_ids):
        calls_a.append(tuple(query_ids))
        return sum(int(query_id) for query_id in query_ids) / len(query_ids)

    def statistic_b(query_ids):
        calls_b.append(tuple(query_ids))
        return 2.0 * statistic_a(query_ids)

    first = paired_query_bootstrap(
        ["3", "1", "2"], statistic_a, statistic_b, n_resamples=8, seed=42
    )
    second = paired_query_bootstrap(
        ["3", "1", "2"],
        lambda ids: sum(map(int, ids)) / len(ids),
        lambda ids: 2.0 * sum(map(int, ids)) / len(ids),
        n_resamples=8,
        seed=42,
    )

    # statistic_b calls statistic_a internally, so compare each B call to the immediately
    # preceding A-side call made by paired_query_bootstrap.
    paired_a_calls = calls_a[::2]
    assert paired_a_calls == calls_b
    assert any(len(set(sample)) < len(sample) for sample in calls_b[1:])
    assert first == second


def test_bootstrap_difference_direction_and_percentile_fixture():
    values_a = {"q1": 0.0, "q2": 0.0, "q3": 0.0}
    values_b = {"q1": 0.0, "q2": 1.0, "q3": 2.0}
    result = bootstrap_mean_comparison(
        values_a,
        values_b,
        n_resamples=6,
        confidence_level=0.5,
        seed=42,
    )

    rng = np.random.default_rng(42)
    ordered = ["q1", "q2", "q3"]
    expected = []
    for _ in range(6):
        indices = rng.integers(0, 3, size=3)
        expected.append(np.mean([values_b[ordered[index]] for index in indices]))
    lower, upper = np.percentile(expected, [25.0, 75.0])

    assert result["point_estimate_a"] == 0.0
    assert result["point_estimate_b"] == 1.0
    assert result["difference"] == 1.0
    assert result["ci_lower"] == pytest.approx(lower)
    assert result["ci_upper"] == pytest.approx(upper)


def test_score_bootstrap_supports_auprc_and_brier():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    labels = {"q1": True, "q2": False, "q3": True, "q4": False}
    raw = {"q1": 3.0, "q2": 2.0, "q3": 1.0, "q4": 0.0}
    calibrated = {"q1": 0.9, "q2": 0.2, "q3": 0.8, "q4": 0.1}
    platt = {"q1": 0.75, "q2": 0.4, "q3": 0.7, "q4": 0.3}

    auprc = bootstrap_score_comparison(
        labels,
        raw,
        calibrated,
        metric="auprc",
        side_a_is_probability=False,
        side_b_is_probability=True,
        n_resamples=10,
        seed=42,
    )
    brier = bootstrap_score_comparison(
        labels,
        platt,
        calibrated,
        metric="brier",
        side_a_is_probability=True,
        side_b_is_probability=True,
        n_resamples=10,
        seed=42,
    )

    query_ids = sorted(labels)
    y_true = [labels[qid] for qid in query_ids]
    assert auprc["point_estimate_a"] == pytest.approx(
        sklearn_metrics.average_precision_score(y_true, [raw[qid] for qid in query_ids])
    )
    assert auprc["point_estimate_b"] == pytest.approx(
        sklearn_metrics.average_precision_score(
            y_true, [calibrated[qid] for qid in query_ids]
        )
    )
    assert brier["point_estimate_a"] == pytest.approx(
        np.mean([(platt[qid] - labels[qid]) ** 2 for qid in query_ids])
    )
    assert brier["point_estimate_b"] == pytest.approx(
        np.mean([(calibrated[qid] - labels[qid]) ** 2 for qid in query_ids])
    )


def test_score_bootstrap_refuses_raw_brier_and_invalid_inputs():
    labels = {"q1": True, "q2": False}
    raw = {"q1": 4.0, "q2": -2.0}
    probability = {"q1": 0.8, "q2": 0.2}

    with pytest.raises(ValueError, match="probability scores"):
        bootstrap_score_comparison(
            labels,
            raw,
            probability,
            metric="brier",
            side_a_is_probability=False,
            side_b_is_probability=True,
        )
    with pytest.raises(ValueError, match="identical query IDs"):
        bootstrap_mean_comparison({"q1": 1.0}, {"q2": 1.0})
    with pytest.raises(ValueError, match="finite"):
        bootstrap_mean_comparison({"q1": float("nan")}, {"q1": 1.0})


def _selection_predictions():
    rows = []
    query_number = 1
    for transition, count in (
        ("no_opportunity", 7),
        ("unchanged_failure", 8),
        ("rescued_by_reranker", 5),
        ("degraded_by_reranker", 5),
    ):
        for index in range(count):
            rows.append(
                {
                    "query_id": f"q{query_number:02d}",
                    "calibrated": 0.02 * query_number,
                    "final_success_10": transition == "rescued_by_reranker",
                    "transition_label": transition,
                }
            )
            query_number += 1
    # These are the two highest incorrect confidences and are removed before transition quotas.
    rows[0]["calibrated"] = 0.99
    rows[7]["calibrated"] = 0.98
    return rows


def test_failure_case_selection_is_deterministic_unique_and_quota_bound():
    predictions = _selection_predictions()
    first = select_failure_cases(predictions)
    second = select_failure_cases(list(reversed(predictions)))

    assert first == second
    assert len(first) == len({row["query_id"] for row in first}) == 12
    assert [row["query_id"] for row in first[:2]] == ["q01", "q08"]
    assert Counter(row["selection_group"] for row in first) == {
        "high_confidence_incorrect": 2,
        "candidate_set_failure": 3,
        "reranking_failure": 3,
        "reranker_rescue": 2,
        "reranker_degradation": 2,
    }
    no_opportunity = [
        row for row in first if row["selection_group"] == "candidate_set_failure"
    ]
    eligible = sorted(
        (
            row
            for row in predictions
            if row["transition_label"] == "no_opportunity" and row["query_id"] != "q01"
        ),
        key=lambda row: (row["calibrated"], row["query_id"]),
    )
    expected_indices = [
        int(np.floor((j + 0.5) * len(eligible) / 3))
        for j in range(3)
    ]
    assert [row["quantile_index"] for row in no_opportunity] == expected_indices


def test_reliability_bins_are_fixed_and_include_probability_one():
    predictions = [
        {"raw_score_platt": 0.0, "final_success_10": False},
        {"raw_score_platt": 0.09, "final_success_10": True},
        {"raw_score_platt": 0.2, "final_success_10": False},
        {"raw_score_platt": 1.0, "final_success_10": True},
    ]

    points = reliability_bins(predictions, "raw_score_platt", n_bins=10)

    assert [point["bin"] for point in points] == [0, 2, 9]
    assert points[0]["count"] == 2
    assert points[-1]["mean_confidence"] == 1.0
    assert points[-1]["upper"] == 1.0


def test_five_figures_generate_from_small_saved_fixtures(tmp_path):
    metadata = {"split": "calibration-dev", "source_runs": {"fixture": "run"}}
    first_stage = [
        {"pipeline": name, "recall@10": 0.5, "recall@50": 0.8, "ndcg@10": 0.6, "n_queries": 4}
        for name in ("bm25", "dense_bge", "hybrid_rrf")
    ]
    decomposition = [
        {
            "pipeline": name,
            "candidate_set_failure_rate": 0.1,
            "reranking_failure_rate": 0.2,
            "final_success_rate": 0.7,
        }
        for name in ("bm25", "dense_bge", "hybrid_rrf")
    ]
    counts = {
        "already_successful": 6,
        "rescued_by_reranker": 1,
        "degraded_by_reranker": 1,
        "unchanged_failure": 1,
        "no_opportunity": 1,
    }
    transitions = [
        {"pipeline": name, "counts": counts}
        for name in ("bm25", "dense_bge", "hybrid_rrf")
    ]
    predictions = [
        {
            "raw_score_platt": probability,
            "calibrated": min(1.0, probability + 0.05),
            "final_success_10": target,
        }
        for probability, target in ((0.1, False), (0.4, False), (0.8, True), (1.0, True))
    ]
    curves = {
        model: [
            {"coverage": 0.5, "risk": 0.0},
            {"coverage": 1.0, "risk": 0.25},
        ]
        for model in ("raw_score", "calibrated")
    }
    paths = {
        "first": tmp_path / "first.png",
        "failure": tmp_path / "failure.png",
        "transitions": tmp_path / "transitions.png",
        "reliability": tmp_path / "reliability.png",
        "risk": tmp_path / "risk.png",
    }

    plot_first_stage_performance(first_stage, paths["first"], metadata)
    plot_failure_breakdown(decomposition, paths["failure"], metadata)
    plot_reranking_transitions(transitions, paths["transitions"], metadata)
    plot_reliability(predictions, paths["reliability"], metadata)
    plot_hybrid_risk_coverage(curves, paths["risk"], metadata)

    assert all(path.stat().st_size > 0 for path in paths.values())


def _write_phase5_fixture(root, split_path):
    pyarrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    query_ids = ["q1", "q2", "q3", "q4"]
    split_path.parent.mkdir(parents=True)
    split_path.write_text("\n".join(query_ids) + "\n")

    for offset, pipeline in enumerate(("bm25", "dense_bge", "hybrid_rrf")):
        run_dir = root / f"2026-01-01T00000{offset}_{pipeline}_calibration-dev"
        run_dir.mkdir(parents=True)
        manifest = {
            "pipeline": pipeline,
            "split": "calibration-dev",
            "git_commit": "source-commit",
            "git_dirty": False,
            "dataset": "beir/scifact",
            "model_revision": "dense-revision" if pipeline != "bm25" else None,
            "reranker_revision": "reranker-revision",
            "candidate_depth": 100,
            "rerank_depth": 50,
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest))
        rows = [
            {
                "query_id": query_id,
                "first_stage_recall@10": 0.5 + 0.1 * offset,
                "reranked_recall@10": 0.6 + 0.1 * offset,
                "first_stage_ndcg@10": 0.4 + 0.1 * offset,
                "reranked_ndcg@10": 0.5 + 0.1 * offset,
                "final_success_10": index % 2 == 0 or offset > 0,
            }
            for index, query_id in enumerate(query_ids)
        ]
        parquet.write_table(pyarrow.Table.from_pylist(rows), run_dir / "query_results.parquet")
        (run_dir / "rankings.jsonl").write_text("")
        (run_dir / "reranked_rankings.jsonl").write_text("")

    confidence_dir = root / "2026-01-01T000010_hybrid_rrf_calibration-train"
    confidence_dir.mkdir()
    (confidence_dir / "manifest.json").write_text(
        json.dumps(
            {
                "pipeline": "hybrid_rrf",
                "split": "calibration-train",
                "git_commit": "source-commit",
                "git_dirty": False,
                "dataset": "beir/scifact",
                "model_revision": "dense-revision",
                "reranker_revision": "reranker-revision",
                "candidate_depth": 100,
                "rerank_depth": 50,
            }
        )
    )
    predictions = [
        {
            "query_id": query_id,
            "raw_score": float(4 - index),
            "raw_score_platt": (0.8, 0.3, 0.7, 0.2)[index],
            "calibrated": (0.9, 0.1, 0.8, 0.05)[index],
            "final_success_10": index % 2 == 0,
            "transition_label": "already_successful" if index % 2 == 0 else "unchanged_failure",
        }
        for index, query_id in enumerate(query_ids)
    ]
    (confidence_dir / "confidence_predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions)
    )
    (confidence_dir / "risk_coverage.json").write_text(json.dumps({"models": {}}))


def test_bootstrap_artifact_has_exact_comparisons_finite_values_and_provenance(
    tmp_path, monkeypatch
):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    splits_dir = tmp_path / "splits"
    _write_phase5_fixture(runs_dir, splits_dir / "calibration_dev.txt")
    monkeypatch.setattr(analysis_module, "BOOTSTRAP_RESAMPLES", 20)

    payload = build_bootstrap_artifact(runs_dir, splits_dir)

    assert len(payload["rows"]) == 11
    assert payload["split"] == "calibration-dev"
    assert payload["bootstrap"]["paired"] is True
    assert payload["bootstrap"]["n_queries"] == 4
    assert payload["provenance"]["hybrid_rrf"]["git_commit"] == "source-commit"
    assert all(row["split"] == "calibration-dev" for row in payload["rows"])
    for row in payload["rows"]:
        for field in (
            "point_estimate_a",
            "point_estimate_b",
            "difference",
            "ci_lower",
            "ci_upper",
        ):
            assert np.isfinite(row[field])


def test_bootstrap_markdown_uses_the_exact_rounded_json_values():
    payload = {
        "split": "calibration-dev",
        "bootstrap": {
            "n_resamples": 1000,
            "confidence_level": 0.95,
            "seed": 42,
        },
        "provenance": {
            "fixture": {"run_dir": "saved-run", "git_commit": "abc"}
        },
        "rows": [
            {
                "comparison_id": "fixture",
                "metric": "recall@10",
                "side_a_label": "A",
                "side_b_label": "B",
                "point_estimate_a": 0.123456,
                "point_estimate_b": 0.234567,
                "difference": 0.111111,
                "ci_lower": -0.000001,
                "ci_upper": 0.222222,
                "n_queries": 4,
                "n_resamples": 1000,
                "seed": 42,
            }
        ],
    }

    markdown = render_bootstrap_markdown(payload)

    for field in ("point_estimate_a", "point_estimate_b", "difference", "ci_lower", "ci_upper"):
        assert f"{payload['rows'][0][field]:.6f}" in markdown


def test_failure_case_loader_uses_train_and_committed_dev_ids(tmp_path, monkeypatch):
    (tmp_path / "calibration_dev.txt").write_text("q-dev\n")
    seen = []
    train = {
        "corpus": {"d1": "Title\nAbstract"},
        "queries": {"q-dev": "dev", "q-train": "train"},
        "qrels": {"q-dev": {"d1": 1}, "q-train": {"d1": 1}},
    }

    def fake_load(split):
        seen.append(split)
        if split != "train":
            raise AssertionError("Final test must not be loaded")
        return train

    monkeypatch.setattr(analysis_module, "load_scifact_split", fake_load)

    result = load_failure_case_data(tmp_path)

    assert seen == ["train"]
    assert set(result["queries"]) == set(result["qrels"]) == {"q-dev"}
