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

import yaml

from retrieval.data import load_config, load_scifact_split, resolve_device, resolve_split, setup_logging
from retrieval.evaluate import evaluate_query
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
) -> tuple[list[dict], dict[str, bool]]:
    """Dispatches on config['pipeline'] and returns (rankings rows, cache-hit flags)."""
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
        return run_bm25(), cache_hits
    if pipeline == "dense_bge":
        return run_dense(), cache_hits
    if pipeline == "hybrid_rrf":
        bm25_rows = run_bm25()
        dense_rows = run_dense()
        rows = hybrid_retrieve(bm25_rows, dense_rows, k=config.get("rrf_k", 60))
        return rows, cache_hits
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


def build_manifest(
    config: dict[str, Any],
    config_path: str,
    split: str,
    device: str,
    n_queries: int,
    cache_hits: dict[str, bool],
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
    return manifest


def write_run_dir(
    config: dict[str, Any], manifest: dict[str, Any], rows: list[dict], metrics: dict[str, Any]
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    split_slug = manifest["split"].replace(" ", "-")
    run_dir = Path(RUNS_DIR) / f"{timestamp}_{config['pipeline']}_{split_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with open(run_dir / "rankings.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    return run_dir


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    logger.info("Loaded config from %s for split %s: %s", args.config, args.split, config)

    train_data = load_scifact_split("train")
    data = resolve_split(args.split, train_data, splits_dir=config.get("splits_dir", "splits"))
    device = resolve_device(config.get("device", "auto"))

    rows, cache_hits = run_pipeline(config, data, device, args.force)
    metrics = compute_metrics(rows, data["qrels"])
    manifest = build_manifest(
        config, args.config, args.split, device, metrics["n_queries"], cache_hits
    )
    run_dir = write_run_dir(config, manifest, rows, metrics)

    logger.info("Wrote run directory %s", run_dir)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
