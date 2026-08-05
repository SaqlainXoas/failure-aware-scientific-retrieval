"""Thin entrypoint: retrieve -> rerank -> evaluate for a single pipeline config."""

import argparse
import importlib.metadata
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

from retrieval.confidence import (
    confidence_metrics,
    extract_features,
    fit_calibrator,
    predict_proba,
    raw_baseline_scores,
    risk_coverage_curve,
    select_thresholds,
    selective_results_at_coverage,
)
from retrieval.data import load_config, load_scifact_split, resolve_device, resolve_split, setup_logging
from retrieval.evaluate import decomposition_metrics, evaluate_query, label_transition
from retrieval.plots import plot_risk_coverage
from retrieval.rerank import rerank
from retrieval.retrieve import bm25_retrieve, dense_retrieve, hybrid_retrieve

logger = logging.getLogger(__name__)

RUNS_DIR = "runs"
CACHE_DIR = ".cache"
_PACKAGES = ["bm25s", "sentence-transformers", "torch", "numpy", "ir-datasets", "pyyaml"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a retrieval pipeline end to end (retrieve, rerank, evaluate)."
    )
    parser.add_argument("--config", required=True, help="Path to a pipeline config YAML.")
    parser.add_argument("--split", default="calibration-dev", help="Query split to run on.")
    parser.add_argument("--force", action="store_true", help="Bypass cache and recompute.")
    return parser.parse_args(argv)


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
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


def run_pipeline(
    config: dict[str, Any], data: dict[str, Any], device: str, force: bool
) -> tuple[list[dict], dict[str, bool], dict[str, list[dict]] | None]:
    """Dispatches on config['pipeline'] and returns (rankings rows, cache-hit flags,
    raw per-retriever rows). raw_rows is only populated for hybrid_rrf (the un-fused
    bm25/dense rows the confidence hybrid-overlap feature needs); None otherwise."""
    pipeline = config["pipeline"]
    corpus, queries = data["corpus"], data["queries"]
    candidate_depth = config.get("candidate_depth", 100)
    cache_hits: dict[str, bool] = {}

    def run_bm25() -> list[dict]:
        return bm25_retrieve(
            corpus,
            queries,
            top_k=candidate_depth,
            cache_dir=CACHE_DIR,
            force=force,
            cache_hits=cache_hits,
        )

    def run_dense() -> list[dict]:
        return dense_retrieve(
            corpus,
            queries,
            top_k=candidate_depth,
            model_name=config["model_name"],
            model_revision=config.get("model_revision"),
            device=device,
            batch_size=config.get("batch_size", 32),
            cache_dir=CACHE_DIR,
            force=force,
            cache_hits=cache_hits,
        )

    if pipeline == "bm25":
        return run_bm25(), cache_hits, None
    if pipeline == "dense_bge":
        return run_dense(), cache_hits, None
    if pipeline == "hybrid_rrf":
        bm25_rows = run_bm25()
        dense_rows = run_dense()
        rows = hybrid_retrieve(bm25_rows, dense_rows, k=config.get("rrf_k", 60))
        return rows, cache_hits, {"bm25": bm25_rows, "dense": dense_rows}
    raise ValueError(f"Unknown pipeline '{pipeline}'")


def compute_metrics(rows: list[dict], qrels: dict[str, dict[str, int]]) -> dict[str, Any]:
    ranked_by_query: dict[str, list[str]] = {}
    for row in rows:
        ranked_by_query.setdefault(row["query_id"], []).append(row)
    ranked_by_query = {
        query_id: [r["doc_id"] for r in sorted(query_rows, key=lambda r: r["rank"])]
        for query_id, query_rows in ranked_by_query.items()
    }

    per_query = [
        evaluate_query(ranked_by_query.get(query_id, []), qrels_for_query)
        for query_id, qrels_for_query in qrels.items()
    ]
    n_queries = len(per_query)
    metric_keys = per_query[0].keys() if per_query else []
    averaged = {
        key: sum(m[key] for m in per_query) / n_queries if n_queries else 0.0
        for key in metric_keys
    }
    n_with_relevant = sum(
        1 for grades in qrels.values() if any(grade > 0 for grade in grades.values())
    )
    return {
        **averaged,
        "n_queries": n_queries,
        "pct_queries_with_relevant": n_with_relevant / n_queries if n_queries else 0.0,
    }


def run_reranking(
    config: dict[str, Any], data: dict[str, Any], first_stage_rows: list[dict], device: str, force: bool
) -> tuple[list[dict], dict[str, bool]]:
    """Reranks each query's own top rerank_depth first-stage candidates via the frozen cross-encoder."""
    cache_hits: dict[str, bool] = {}
    rows = rerank(
        first_stage_rows,
        data["queries"],
        data["corpus"],
        top_k=config.get("rerank_depth", 50),
        model_name=config["reranker_model"],
        model_revision=config.get("reranker_revision"),
        device=device,
        batch_size=config.get("reranker_batch_size", 32),
        cache_dir=CACHE_DIR,
        force=force,
        cache_hits=cache_hits,
    )
    return rows, cache_hits


def compute_query_labels(
    first_stage_rows: list[dict],
    reranked_rows: list[dict],
    qrels: dict[str, dict[str, int]],
    rerank_depth: int,
) -> list[dict]:
    """Per-query candidate/final success and transition label — the row-level source the
    decomposition table and later manual failure analysis both read from."""
    first_stage_by_query: dict[str, list[str]] = {}
    for row in sorted(first_stage_rows, key=lambda r: r["rank"]):
        first_stage_by_query.setdefault(row["query_id"], []).append(row["doc_id"])
    reranked_by_query: dict[str, list[str]] = {}
    for row in sorted(reranked_rows, key=lambda r: r["rank"]):
        reranked_by_query.setdefault(row["query_id"], []).append(row["doc_id"])

    labels = []
    for query_id, qrels_for_query in qrels.items():
        gold_doc_ids = {doc_id for doc_id, grade in qrels_for_query.items() if grade > 0}
        first_stage_ranked = first_stage_by_query.get(query_id, [])
        reranked_ranked = reranked_by_query.get(query_id, [])
        transition = label_transition(gold_doc_ids, first_stage_ranked, reranked_ranked, rerank_depth)
        labels.append(
            {
                "query_id": query_id,
                "candidate_success_50": bool(gold_doc_ids & set(first_stage_ranked[:rerank_depth])),
                "final_success_10": bool(gold_doc_ids & set(reranked_ranked[:10])),
                "transition_label": transition,
            }
        )
    return labels


def confidence_feature_names(pipeline: str) -> list[str]:
    from retrieval.confidence import COMMON_FEATURES, HYBRID_FEATURES

    return COMMON_FEATURES + HYBRID_FEATURES if pipeline == "hybrid_rrf" else COMMON_FEATURES


def run_confidence_features(
    config: dict[str, Any],
    rows: list[dict],
    reranked_rows: list[dict],
    raw_rows: dict[str, list[dict]] | None,
    rerank_depth: int,
) -> dict[str, dict[str, float]]:
    return extract_features(rows, reranked_rows, rerank_depth, raw_rows=raw_rows)


def run_calibration(
    config: dict[str, Any],
    device: str,
    force: bool,
    train_features: dict[str, dict[str, float]],
    train_labels: dict[str, bool],
    dev_features: dict[str, dict[str, float]],
    dev_labels: dict[str, bool],
    dev_reranked_rows: list[dict],
) -> dict[str, Any]:
    """Fits on calibration-train only; every selection/metric below is computed on
    calibration-dev, never on train. Baseline and calibrated results are kept separate."""
    feature_names = confidence_feature_names(config["pipeline"])
    class_weight = config.get("confidence_class_weight")
    coverage_levels = tuple(config.get("confidence_coverage_levels", [1.0, 0.8, 0.6]))

    calibrator = fit_calibrator(train_features, train_labels, feature_names, class_weight=class_weight)
    dev_calibrated_probs = predict_proba(calibrator, dev_features, feature_names)
    dev_baseline_scores = raw_baseline_scores(dev_reranked_rows)

    return {
        "calibrator": calibrator,
        "feature_names": feature_names,
        "class_weight": class_weight,
        "confidence_metrics": {
            "baseline": confidence_metrics(dev_baseline_scores, dev_labels),
            "calibrated": confidence_metrics(dev_calibrated_probs, dev_labels),
        },
        "thresholds": {
            "baseline": select_thresholds(dev_baseline_scores, dev_labels, coverage_levels),
            "calibrated": select_thresholds(dev_calibrated_probs, dev_labels, coverage_levels),
        },
        "selective_results": {
            "baseline": selective_results_at_coverage(dev_baseline_scores, dev_labels, coverage_levels),
            "calibrated": selective_results_at_coverage(dev_calibrated_probs, dev_labels, coverage_levels),
        },
        "risk_coverage": {
            "baseline": risk_coverage_curve(dev_baseline_scores, dev_labels),
            "calibrated": risk_coverage_curve(dev_calibrated_probs, dev_labels),
        },
    }


def build_manifest(
    config: dict[str, Any],
    config_path: str,
    split: str,
    device: str,
    n_queries: int,
    cache_hits: dict[str, bool],
    calibrated: bool = False,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": config.get("dataset"),
        "split": split,
        "pipeline": config["pipeline"],
        "n_queries": n_queries,
        "candidate_depth": config.get("candidate_depth"),
        "package_versions": _package_versions(),
        "os": platform.platform(),
        "device": device,
        "seed": config.get("seed"),
        "config_path": config_path,
        "cache_hits": cache_hits,
    }
    if config["pipeline"] in ("dense_bge", "hybrid_rrf"):
        manifest["model_name"] = config.get("model_name")
        manifest["model_revision"] = config.get("model_revision")
        manifest["batch_size"] = config.get("batch_size")
    if config["pipeline"] in ("bm25", "hybrid_rrf"):
        manifest["bm25_params"] = {"k1": "library_default", "b": "library_default"}
    if config["pipeline"] == "hybrid_rrf":
        manifest["rrf_k"] = config.get("rrf_k", 60)
    manifest["reranker_model"] = config.get("reranker_model")
    manifest["reranker_revision"] = config.get("reranker_revision")
    manifest["rerank_depth"] = config.get("rerank_depth")
    manifest["reranker_batch_size"] = config.get("reranker_batch_size")
    manifest["confidence_feature_names"] = confidence_feature_names(config["pipeline"])
    manifest["confidence_class_weight"] = config.get("confidence_class_weight")
    manifest["calibrated"] = calibrated
    return manifest


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


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
    calibration: dict[str, Any] | None = None,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    split_slug = manifest["split"].replace(" ", "-")
    run_dir = Path(RUNS_DIR) / f"{timestamp}_{config['pipeline']}_{split_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    _write_jsonl(run_dir / "rankings.jsonl", rows)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    _write_jsonl(run_dir / "reranked_rankings.jsonl", reranked_rows)
    (run_dir / "reranked_metrics.json").write_text(json.dumps(reranked_metrics, indent=2) + "\n")
    _write_jsonl(run_dir / "failure_labels.jsonl", labels)
    (run_dir / "decomposition_metrics.json").write_text(json.dumps(decomposition, indent=2) + "\n")
    _write_jsonl(run_dir / "confidence_features.jsonl", confidence_features)

    if calibration is not None:
        joblib.dump(calibration["calibrator"], run_dir / "calibrator.joblib")
        (run_dir / "confidence_metrics.json").write_text(
            json.dumps(calibration["confidence_metrics"], indent=2) + "\n"
        )
        (run_dir / "thresholds.json").write_text(json.dumps(calibration["thresholds"], indent=2) + "\n")
        (run_dir / "selective_results.json").write_text(
            json.dumps(calibration["selective_results"], indent=2) + "\n"
        )
        (run_dir / "risk_coverage.json").write_text(json.dumps(calibration["risk_coverage"], indent=2) + "\n")
        plot_risk_coverage(
            calibration["risk_coverage"]["baseline"],
            calibration["risk_coverage"]["calibrated"],
            run_dir / "risk_coverage.png",
        )

    return run_dir


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    logger.info("Loaded config from %s for split %s: %s", args.config, args.split, config)

    train_data = load_scifact_split("train")
    data = resolve_split(args.split, train_data, splits_dir=config.get("splits_dir", "splits"))
    device = resolve_device(config.get("device", "auto"))

    rows, cache_hits, raw_rows = run_pipeline(config, data, device, args.force)
    metrics = compute_metrics(rows, data["qrels"])

    rerank_depth = config.get("rerank_depth", 50)
    reranked_rows, rerank_cache_hits = run_reranking(config, data, rows, device, args.force)
    reranked_metrics = compute_metrics(reranked_rows, data["qrels"])
    labels = compute_query_labels(rows, reranked_rows, data["qrels"], rerank_depth)
    decomposition = decomposition_metrics([label["transition_label"] for label in labels])
    features_by_query = run_confidence_features(config, rows, reranked_rows, raw_rows, rerank_depth)
    confidence_features = [{"query_id": qid, **feats} for qid, feats in sorted(features_by_query.items())]

    calibration = None
    if args.split == "calibration-train":
        dev_data = resolve_split("calibration-dev", train_data, splits_dir=config.get("splits_dir", "splits"))
        dev_rows, dev_cache_hits, dev_raw_rows = run_pipeline(config, dev_data, device, args.force)
        dev_reranked_rows, dev_rerank_cache_hits = run_reranking(config, dev_data, dev_rows, device, args.force)
        dev_labels_list = compute_query_labels(dev_rows, dev_reranked_rows, dev_data["qrels"], rerank_depth)
        dev_features_by_query = run_confidence_features(
            config, dev_rows, dev_reranked_rows, dev_raw_rows, rerank_depth
        )
        train_labels = {label["query_id"]: label["final_success_10"] for label in labels}
        dev_labels = {label["query_id"]: label["final_success_10"] for label in dev_labels_list}
        calibration = run_calibration(
            config,
            device,
            args.force,
            features_by_query,
            train_labels,
            dev_features_by_query,
            dev_labels,
            dev_reranked_rows,
        )
        cache_hits = {**cache_hits, **rerank_cache_hits, **dev_cache_hits, **dev_rerank_cache_hits}
    else:
        cache_hits = {**cache_hits, **rerank_cache_hits}

    manifest = build_manifest(
        config,
        args.config,
        args.split,
        device,
        metrics["n_queries"],
        cache_hits,
        calibrated=calibration is not None,
    )
    run_dir = write_run_dir(
        config,
        manifest,
        rows,
        metrics,
        reranked_rows,
        reranked_metrics,
        labels,
        decomposition,
        confidence_features,
        calibration=calibration,
    )

    logger.info("Wrote run directory %s", run_dir)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))
    logger.info("Reranked metrics: %s", json.dumps(reranked_metrics, indent=2))
    logger.info("Decomposition: %s", json.dumps(decomposition, indent=2))
    if calibration is not None:
        logger.info("Confidence metrics: %s", json.dumps(calibration["confidence_metrics"], indent=2))


if __name__ == "__main__":
    main()
