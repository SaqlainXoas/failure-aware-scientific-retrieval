"""Run I/O: manifest contents, run-directory schema, and table generation."""

import json

import joblib
import pytest

import retrieval.runio as runio_module
from retrieval.run import per_query_metrics
from retrieval.runio import build_manifest, build_query_results, write_run_dir
from retrieval.tables import (
    MODEL_ROLES,
    build_confidence_table,
    build_rerank_delta_table,
    build_selective_table,
    load_confidence_artifacts,
    load_run_artifacts,
    write_run_manifests,
)

BM25_PARAMS = {
    "library": "bm25s",
    "library_version": "0.3.10",
    "k1": 1.5,
    "b": 0.75,
    "delta": 0.5,
    "method": "lucene",
    "idf_method": "lucene",
    "tokenizer": {"lower": True, "token_pattern": r"(?u)\b\w\w+\b", "stopwords": "english", "stemmer": None},
    "corpus_text_format": "<title>\\n<abstract>",
    "preprocessing_version": "v1",
}

# The manifest contract, checked field by field rather than "a manifest exists".
REQUIRED_MANIFEST_KEYS = {
    "git_commit",
    "git_dirty",
    "timestamp_utc",
    "dataset",
    "split",
    "pipeline",
    "n_queries",
    "candidate_depth",
    "package_versions",
    "os",
    "device",
    "seed",
    "config_path",
    "cache_hits",
    "reranker_model",
    "reranker_revision",
    "rerank_depth",
    "reranker_batch_size",
}


def _config(pipeline="hybrid_rrf"):
    return {
        "pipeline": pipeline,
        "dataset": "beir/scifact",
        "seed": 42,
        "candidate_depth": 100,
        "rrf_k": 60,
        "model_name": "BAAI/bge-small-en-v1.5",
        "model_revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        "batch_size": 32,
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "reranker_revision": "c5ee24cb16019beea0893ab7796b1df96625c6b8",
        "rerank_depth": 50,
        "reranker_batch_size": 32,
        "confidence_class_weight": "balanced",
    }


@pytest.mark.parametrize("pipeline", ["bm25", "dense_bge", "hybrid_rrf"])
def test_manifest_has_every_required_plan_field(pipeline):
    manifest = build_manifest(
        _config(pipeline),
        "configs/x.yaml",
        "calibration-dev",
        "cpu",
        n_queries=162,
        cache_hits={},
        retrieval_params={"bm25": BM25_PARAMS},
    )

    assert REQUIRED_MANIFEST_KEYS <= set(manifest)
    assert manifest["seed"] == 42
    assert manifest["rerank_depth"] == 50


def test_manifest_records_package_versions_needed_to_reproduce_the_calibrator():
    manifest = build_manifest(
        _config("bm25"), "c.yaml", "calibration-dev", "cpu", 1, {}, retrieval_params={"bm25": BM25_PARAMS}
    )
    versions = manifest["package_versions"]

    for package in ("python", "scikit-learn", "bm25s", "sentence-transformers", "torch"):
        assert versions.get(package), f"{package} version missing from manifest"


def test_manifest_records_concrete_bm25_params_not_a_placeholder():
    manifest = build_manifest(
        _config("bm25"), "c.yaml", "calibration-dev", "cpu", 1, {}, retrieval_params={"bm25": BM25_PARAMS}
    )

    # The manifest must record the values actually used, not the string "library default".
    assert manifest["bm25_params"]["k1"] == 1.5
    assert manifest["bm25_params"]["b"] == 0.75
    assert manifest["bm25_params"]["tokenizer"]["stopwords"] == "english"


def test_manifest_keeps_primary_and_exploratory_feature_sets_separate():
    hybrid = build_manifest(_config("hybrid_rrf"), "c.yaml", "calibration-dev", "cpu", 1, {})
    bm25 = build_manifest(_config("bm25"), "c.yaml", "calibration-dev", "cpu", 1, {})

    # The overlap feature is an addition to the common set, never a replacement.
    assert "hybrid_bm25_dense_top10_overlap" not in hybrid["confidence_primary_feature_names"]
    assert "hybrid_bm25_dense_top10_overlap" in hybrid["confidence_exploratory_feature_names"]
    assert bm25["confidence_exploratory_feature_names"] is None
    assert hybrid["confidence_primary_feature_names"] == bm25["confidence_primary_feature_names"]


def test_manifest_records_fit_and_eval_splits_only_when_calibrated():
    calibrated = build_manifest(_config(), "c.yaml", "calibration-train", "cpu", 1, {}, calibrated=True)
    plain = build_manifest(_config(), "c.yaml", "calibration-dev", "cpu", 1, {}, calibrated=False)

    assert calibrated["confidence_fit_split"] == "calibration-train"
    assert calibrated["confidence_eval_split"] == "calibration-dev"
    assert "confidence_fit_split" not in plain


def test_manifest_records_clean_source_state(monkeypatch):
    monkeypatch.setattr(runio_module, "_git_dirty", lambda: False)

    manifest = build_manifest(_config("bm25"), "c.yaml", "calibration-dev", "cpu", 1, {})

    assert manifest["git_dirty"] is False


def _rows(query_id, doc_scores):
    return [
        {"query_id": query_id, "doc_id": doc_id, "rank": rank, "score": score}
        for rank, (doc_id, score) in enumerate(doc_scores, start=1)
    ]


def test_write_run_dir_emits_the_full_plan_14_artifact_set(tmp_path):
    config = _config("bm25")
    manifest = build_manifest(config, "c.yaml", "calibration-dev", "cpu", 1, {})
    rows = _rows("q1", [("a", 3.0), ("b", 1.0)])
    reranked = _rows("q1", [("b", 0.9), ("a", 0.2)])
    qrels = {"q1": {"a": 1}}
    labels = [
        {"query_id": "q1", "candidate_success_50": True, "final_success_10": True, "transition_label": "already_successful"}
    ]
    features = {"q1": {"reranker_top1_score": 0.9}}
    query_results = build_query_results(
        labels, per_query_metrics(rows, qrels), per_query_metrics(reranked, qrels), features
    )

    run_dir = write_run_dir(
        config,
        manifest,
        rows,
        {"recall@10": 1.0},
        reranked,
        {"recall@10": 1.0},
        labels,
        {"final_success_rate": 1.0},
        [{"query_id": "q1", "reranker_top1_score": 0.9}],
        query_results,
        logs_text="captured run log\n",
        runs_dir=tmp_path,
    )

    for filename in (
        "config.yaml",
        "manifest.json",
        "rankings.jsonl",
        "metrics.json",
        "reranked_rankings.jsonl",
        "reranked_metrics.json",
        "failure_labels.jsonl",
        "decomposition_metrics.json",
        "confidence_features.jsonl",
        "query_results.parquet",
        "logs.txt",
    ):
        assert (run_dir / filename).exists(), f"{filename} missing from run directory"
    assert (run_dir / "logs.txt").read_text() == "captured run log\n"


def test_query_results_row_schema_is_joinable_per_query():
    rows = _rows("q1", [("a", 3.0), ("b", 1.0)])
    reranked = _rows("q1", [("b", 0.9), ("a", 0.2)])
    qrels = {"q1": {"a": 1}}
    labels = [
        {"query_id": "q1", "candidate_success_50": True, "final_success_10": False, "transition_label": "degraded_by_reranker"}
    ]

    result = build_query_results(
        labels, per_query_metrics(rows, qrels), per_query_metrics(reranked, qrels), {"q1": {"feat": 0.5}}
    )

    assert len(result) == 1
    row = result[0]
    assert row["query_id"] == "q1"
    assert row["transition_label"] == "degraded_by_reranker"
    assert row["candidate_success_50"] is True
    assert row["final_success_10"] is False
    assert "first_stage_ndcg@10" in row and "reranked_ndcg@10" in row
    assert row["feat"] == 0.5


def test_query_results_parquet_roundtrips(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    config = _config("bm25")
    manifest = build_manifest(config, "c.yaml", "calibration-dev", "cpu", 1, {})
    rows = _rows("q1", [("a", 3.0)])
    qrels = {"q1": {"a": 1}}
    labels = [
        {"query_id": "q1", "candidate_success_50": True, "final_success_10": True, "transition_label": "already_successful"}
    ]
    query_results = build_query_results(
        labels, per_query_metrics(rows, qrels), per_query_metrics(rows, qrels), {"q1": {"feat": 0.5}}
    )

    run_dir = write_run_dir(
        config, manifest, rows, {}, rows, {}, labels, {}, [], query_results, runs_dir=tmp_path
    )

    table = pq.read_table(run_dir / "query_results.parquet").to_pylist()
    assert table == query_results


def test_rerank_delta_table_is_computed_from_saved_metrics(tmp_path):
    run_dir = tmp_path / "2026-01-01T000000_bm25_calibration-dev"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"pipeline": "bm25", "split": "calibration-dev", "git_commit": "abc"})
    )
    (run_dir / "metrics.json").write_text(json.dumps({"recall@5": 0.5, "recall@10": 0.6, "mrr@10": 0.4, "ndcg@10": 0.45}))
    (run_dir / "reranked_metrics.json").write_text(
        json.dumps({"recall@5": 0.7, "recall@10": 0.65, "mrr@10": 0.5, "ndcg@10": 0.55})
    )
    (run_dir / "decomposition_metrics.json").write_text(json.dumps({}))

    artifacts = load_run_artifacts(tmp_path)
    table = build_rerank_delta_table(artifacts)

    assert len(table) == 1
    row = table[0]
    assert row["recall@10_first_stage"] == 0.6
    assert row["recall@10_reranked"] == 0.65
    assert row["recall@10_delta"] == pytest.approx(0.05)
    assert row["recall@5_delta"] == pytest.approx(0.2)
    assert artifacts["bm25"]["provenance"]["git_commit"] == "abc"


def _calibration_payload():
    model_result = {
        "confidence_metrics": {
            "auroc": 0.8,
            "auprc": 0.9,
            "brier": 0.1,
            "is_probability": True,
            "n_queries": 1,
        },
        "thresholds": {"1.0": 0.7, "0.8": 0.7, "0.6": 0.7},
        "selective_results": {
            level: {"n_kept": 1, "success_rate": 1.0}
            for level in ("1.0", "0.8", "0.6")
        },
        "risk_coverage": [{"coverage": 1.0, "risk": 0.0, "threshold": 0.7}],
    }
    raw_result = {
        **model_result,
        "confidence_metrics": {
            **model_result["confidence_metrics"],
            "brier": None,
            "is_probability": False,
        },
    }
    return {
        "estimators": {
            "raw_score_platt": {"kind": "platt"},
            "calibrated": {"kind": "common"},
        },
        "primary_model": "calibrated",
        "exploratory_models": [],
        "feature_names": {
            "raw_score_platt": ["reranker_top1_score"],
            "calibrated": ["reranker_top1_score"],
        },
        "class_weight": "balanced",
        "fit_split": "calibration-train",
        "eval_split": "calibration-dev",
        "results": {
            "raw_score": raw_result,
            "raw_score_platt": model_result,
            "calibrated": model_result,
        },
        "predictions": [
            {
                "query_id": "q1",
                "raw_score": 1.2,
                "raw_score_platt": 0.7,
                "calibrated": 0.8,
                "final_success_10": True,
                "transition_label": "already_successful",
            }
        ],
    }


def test_write_run_dir_persists_calibration_artifacts_and_estimators(tmp_path):
    config = _config("bm25")
    manifest = build_manifest(
        config, "c.yaml", "calibration-train", "cpu", 1, {}, calibrated=True
    )
    rows = _rows("q1", [("a", 1.0)])
    labels = [
        {
            "query_id": "q1",
            "candidate_success_50": True,
            "final_success_10": True,
            "transition_label": "already_successful",
        }
    ]

    run_dir = write_run_dir(
        config,
        manifest,
        rows,
        {},
        rows,
        {},
        labels,
        {},
        [],
        [{"query_id": "q1"}],
        calibration=_calibration_payload(),
        runs_dir=tmp_path,
    )

    for filename in (
        "calibrator_raw_score_platt.joblib",
        "calibrator_calibrated.joblib",
        "confidence_metrics.json",
        "thresholds.json",
        "selective_results.json",
        "risk_coverage.json",
        "confidence_predictions.jsonl",
        "risk_coverage.png",
    ):
        assert (run_dir / filename).exists()
    assert joblib.load(run_dir / "calibrator_calibrated.joblib") == {
        "kind": "common"
    }
    prediction = json.loads(
        (run_dir / "confidence_predictions.jsonl").read_text().strip()
    )
    assert prediction["transition_label"] == "already_successful"


def test_write_run_dir_never_reuses_a_timestamped_directory(tmp_path):
    config = _config("bm25")
    manifest = build_manifest(config, "c.yaml", "calibration-dev", "cpu", 1, {})
    rows = _rows("q1", [("a", 1.0)])

    first = write_run_dir(
        config, manifest, rows, {}, rows, {}, [], {}, [], [], runs_dir=tmp_path
    )
    second = write_run_dir(
        config, manifest, rows, {}, rows, {}, [], {}, [], [], runs_dir=tmp_path
    )

    assert first != second
    assert first.exists() and second.exists()


def _write_confidence_run(root, pipeline):
    run_dir = root / f"2026-01-01T000000_{pipeline}_calibration-train"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "pipeline": pipeline,
                "split": "calibration-train",
                "git_commit": "code-commit",
                "git_dirty": False,
                "confidence_class_weight": "balanced",
            }
        )
    )
    models = {
        "raw_score": {
            "auroc": 0.7,
            "auprc": 0.8,
            "brier": None,
            "n_queries": 162,
        },
        "raw_score_platt": {
            "auroc": 0.7,
            "auprc": 0.8,
            "brier": 0.12,
            "n_queries": 162,
        },
        "calibrated": {
            "auroc": 0.8,
            "auprc": 0.9,
            "brier": 0.1,
            "n_queries": 162,
        },
    }
    if pipeline == "hybrid_rrf":
        models["calibrated_hybrid_exploratory"] = {
            "auroc": 0.81,
            "auprc": 0.91,
            "brier": 0.09,
            "n_queries": 162,
        }
    common = {
        "fit_split": "calibration-train",
        "eval_split": "calibration-dev",
        "primary_model": "calibrated",
    }
    (run_dir / "confidence_metrics.json").write_text(
        json.dumps({**common, "models": models})
    )
    selective = {
        name: {
            level: {"n_kept": 162, "success_rate": 0.8}
            for level in ("1.0", "0.8", "0.6")
        }
        for name in models
    }
    (run_dir / "selective_results.json").write_text(
        json.dumps({**common, "models": selective})
    )
    return run_dir


def test_confidence_tables_label_roles_splits_primary_pipeline_and_provenance(
    tmp_path,
):
    for pipeline in ("bm25", "dense_bge", "hybrid_rrf"):
        _write_confidence_run(tmp_path, pipeline)

    artifacts = load_confidence_artifacts(tmp_path)
    confidence = build_confidence_table(artifacts)
    selective = build_selective_table(artifacts)

    assert {row["model_role"] for row in confidence} == set(MODEL_ROLES.values())
    assert all(
        row["is_primary_pipeline"] == (row["pipeline"] == "hybrid_rrf")
        for row in confidence + selective
    )
    assert artifacts["hybrid_rrf"]["provenance"]["git_dirty"] is False
    assert artifacts["bm25"]["provenance"]["fit_split"] == "calibration-train"
    assert artifacts["bm25"]["provenance"]["eval_split"] == "calibration-dev"


def test_write_run_manifests_copies_all_six_backing_manifests(tmp_path):
    for pipeline in ("bm25", "dense_bge", "hybrid_rrf"):
        _write_confidence_run(tmp_path, pipeline)
        dev_dir = tmp_path / f"2026-01-01T000001_{pipeline}_calibration-dev"
        dev_dir.mkdir()
        (dev_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "pipeline": pipeline,
                    "split": "calibration-dev",
                    "git_commit": "code-commit",
                    "git_dirty": False,
                }
            )
        )

    destination = tmp_path / "manifests"
    written = write_run_manifests(tmp_path, destination)

    assert len(written) == 6
    assert len(list(destination.glob("*.json"))) == 6
