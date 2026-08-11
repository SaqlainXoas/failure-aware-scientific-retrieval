"""Run-directory persistence: manifests, per-query tables, and saved artifacts.

Split from `run.py` so the pipeline orchestration is not interleaved with the file layout
every run has to write. Nothing here computes a result; it only records one.
"""

import importlib.metadata
import io
import json
import logging
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import yaml

from retrieval.confidence import confidence_feature_names, exploratory_feature_names
from retrieval.plots import plot_risk_coverage
from retrieval.retrieve import QUERY_INSTRUCTION

RUNS_DIR = "runs"
FIT_SPLIT = "calibration-train"
EVAL_SPLIT = "calibration-dev"
_PACKAGES = [
    "bm25s",
    "sentence-transformers",
    "torch",
    "numpy",
    "ir-datasets",
    "pyyaml",
    "scikit-learn",
    "joblib",
    "matplotlib",
    "pyarrow",
]


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _git_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
        return bool(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in _PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_manifest(
    config: dict[str, Any],
    config_path: str,
    split: str,
    device: str,
    n_queries: int,
    cache_hits: dict[str, bool],
    retrieval_params: dict[str, Any] | None = None,
    calibrated: bool = False,
) -> dict[str, Any]:
    pipeline = config["pipeline"]
    retrieval_params = retrieval_params or {}
    manifest: dict[str, Any] = {
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": config.get("dataset"),
        "split": split,
        "pipeline": pipeline,
        "n_queries": n_queries,
        "candidate_depth": config.get("candidate_depth"),
        "package_versions": _package_versions(),
        "os": platform.platform(),
        "device": device,
        "seed": config.get("seed"),
        "config_path": config_path,
        "cache_hits": cache_hits,
    }
    if pipeline in ("dense_bge", "hybrid_rrf"):
        manifest["model_name"] = config.get("model_name")
        manifest["model_revision"] = config.get("model_revision")
        manifest["batch_size"] = config.get("batch_size")
        manifest["query_instruction"] = QUERY_INSTRUCTION
        manifest["similarity"] = "cosine (normalized embeddings, exact matmul)"
    if pipeline in ("bm25", "hybrid_rrf"):
        # Concrete values read off the fitted index, not the string "library default", so the
        # manifest records what was actually used even if a bm25s release changes its defaults.
        manifest["bm25_params"] = retrieval_params.get("bm25")
    if pipeline == "hybrid_rrf":
        manifest["rrf_k"] = config.get("rrf_k", 60)
        manifest["fusion"] = "reciprocal_rank_fusion"
    manifest["reranker_model"] = config.get("reranker_model")
    manifest["reranker_revision"] = config.get("reranker_revision")
    manifest["rerank_depth"] = config.get("rerank_depth")
    manifest["reranker_batch_size"] = config.get("reranker_batch_size")
    manifest["confidence_primary_feature_names"] = confidence_feature_names(pipeline)
    manifest["confidence_exploratory_feature_names"] = exploratory_feature_names(pipeline)
    manifest["confidence_class_weight"] = config.get("confidence_class_weight")
    manifest["calibrated"] = calibrated
    if calibrated:
        manifest["confidence_fit_split"] = FIT_SPLIT
        manifest["confidence_eval_split"] = EVAL_SPLIT
    return manifest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_query_results(
    labels: list[dict],
    per_query_metrics: dict[str, dict[str, float]],
    reranked_per_query_metrics: dict[str, dict[str, float]],
    features_by_query: dict[str, dict[str, float]],
) -> list[dict]:
    """One row per query joining first-stage metrics, reranked metrics, failure/transition
    labels, and confidence features — the unit of resampling for the paired bootstrap."""
    rows = []
    for label in labels:
        query_id = label["query_id"]
        first = per_query_metrics.get(query_id, {})
        reranked = reranked_per_query_metrics.get(query_id, {})
        rows.append(
            {
                "query_id": query_id,
                **{f"first_stage_{k}": v for k, v in first.items()},
                **{f"reranked_{k}": v for k, v in reranked.items()},
                "candidate_success_50": label["candidate_success_50"],
                "final_success_10": label["final_success_10"],
                "transition_label": label["transition_label"],
                **features_by_query.get(query_id, {}),
            }
        )
    return rows


def _write_parquet(path: Path, rows: list[dict]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.Table.from_pylist(rows), path)


def write_run_dir(
    config: dict[str, Any],
    manifest: dict[str, Any],
    rows: list[dict],
    metrics: dict[str, Any],
    reranked_rows: list[dict],
    reranked_metrics: dict[str, Any],
    labels: list[dict],
    decomposition: dict[str, Any],
    confidence_features: list[dict],
    query_results: list[dict],
    calibration: dict[str, Any] | None = None,
    logs_text: str = "",
    runs_dir: str | Path = RUNS_DIR,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S.%fZ")
    split_slug = manifest["split"].replace(" ", "-")
    run_dir = Path(runs_dir) / f"{timestamp}_{config['pipeline']}_{split_slug}"
    run_dir.mkdir(parents=True, exist_ok=False)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_jsonl(run_dir / "rankings.jsonl", rows)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _write_jsonl(run_dir / "reranked_rankings.jsonl", reranked_rows)
    (run_dir / "reranked_metrics.json").write_text(json.dumps(reranked_metrics, indent=2) + "\n")
    _write_jsonl(run_dir / "failure_labels.jsonl", labels)
    (run_dir / "decomposition_metrics.json").write_text(json.dumps(decomposition, indent=2) + "\n")
    _write_jsonl(run_dir / "confidence_features.jsonl", confidence_features)
    _write_parquet(run_dir / "query_results.parquet", query_results)
    (run_dir / "logs.txt").write_text(logs_text)

    if calibration is not None:
        for name, estimator in calibration["estimators"].items():
            joblib.dump(estimator, run_dir / f"calibrator_{name}.joblib")
        splits = {"fit_split": calibration["fit_split"], "eval_split": calibration["eval_split"]}
        for artifact in ("confidence_metrics", "thresholds", "selective_results", "risk_coverage"):
            payload = {
                **splits,
                "primary_model": calibration["primary_model"],
                "models": {
                    name: result[artifact] for name, result in calibration["results"].items()
                },
            }
            (run_dir / f"{artifact}.json").write_text(json.dumps(payload, indent=2) + "\n")
        _write_jsonl(run_dir / "confidence_predictions.jsonl", calibration["predictions"])
        plot_risk_coverage(
            {name: result["risk_coverage"] for name, result in calibration["results"].items()},
            run_dir / "risk_coverage.png",
        )

        cross_validated = calibration.get("cross_validated")
        if cross_validated is not None:
            _write_cv_artifacts(run_dir, calibration, cross_validated)

    return run_dir


def _write_cv_artifacts(run_dir: Path, calibration: dict, cross_validated: dict) -> None:
    """Cross-validated confidence artifacts, kept in separate files from the predeclared
    train/dev ones so a reader can never mistake the pooled estimate for the protocol result."""
    (run_dir / "confidence_cv_metrics.json").write_text(
        json.dumps(
            {
                "protocol": cross_validated["protocol"],
                "n_splits": cross_validated["n_splits"],
                "seed": cross_validated["seed"],
                "n_queries": cross_validated["n_queries"],
                "n_failures": cross_validated["n_failures"],
                "primary_model": calibration["primary_model"],
                "models": {
                    name: result["confidence_metrics"]
                    for name, result in cross_validated["results"].items()
                },
                "selective_results": {
                    name: result["selective_results"]
                    for name, result in cross_validated["results"].items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    _write_jsonl(run_dir / "confidence_cv_predictions.jsonl", cross_validated["predictions"])


def _capture_logs() -> io.StringIO:
    """Tees root logging into a buffer so logs.txt can be written into the run directory,
    which does not exist yet when the run starts."""
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return buffer
