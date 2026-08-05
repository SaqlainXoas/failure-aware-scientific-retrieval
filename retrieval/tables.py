"""Builds the result tables from saved run outputs on calibration-dev.

Every number here is read out of a run directory's saved JSON — nothing is recomputed or
typed by hand (plan §17 Phase 5). Each table carries a provenance block naming the run
directory, git commit, and model revisions it came from, because `runs/` is not tracked.
"""

import json
from pathlib import Path
from typing import Any

RUNS_DIR = "runs"
SPLIT = "calibration-dev"
CALIBRATION_SPLIT = "calibration-train"
PIPELINES = ["bm25", "dense_bge", "hybrid_rrf"]
PRIMARY_PIPELINE = "hybrid_rrf"
TABLES_DIR = "results/tables"
MANIFESTS_DIR = "results/manifests"
COVERAGE_LEVELS = ["1.0", "0.8", "0.6"]
MODEL_ROLES = {
    "raw_score": "raw_ranking_baseline",
    "raw_score_platt": "train_fitted_probability_baseline",
    "calibrated": "common_feature_model",
    "calibrated_hybrid_exploratory": "exploratory_overlap_ablation",
}

METRIC_COLUMNS = ["recall@5", "recall@10", "recall@50", "mrr@10", "ndcg@10", "n_queries"]
DECOMPOSITION_COLUMNS = [
    "candidate_set_failure_rate",
    "reranking_failure_rate",
    "final_success_rate",
    "opportunity_rate",
    "rescue_rate",
    "degradation_rate",
    "conditional_conversion_rate",
    "share_of_failures_from_candidate_set",
]
DELTA_METRICS = ["recall@5", "recall@10", "mrr@10", "ndcg@10"]
DELTA_COLUMNS = [
    column
    for metric in DELTA_METRICS
    for column in (f"{metric}_first_stage", f"{metric}_reranked", f"{metric}_delta")
]


def _latest_run_dirs(runs_dir: str | Path = RUNS_DIR, split: str = SPLIT) -> dict[str, Path]:
    """Finds the most recent run dir per pipeline (by timestamp-prefixed name) under runs_dir."""
    runs_by_pipeline: dict[str, list[Path]] = {pipeline: [] for pipeline in PIPELINES}
    runs_path = Path(runs_dir)
    if not runs_path.exists():
        return {}
    for run_dir in runs_path.iterdir():
        if not run_dir.is_dir():
            continue
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        pipeline = manifest.get("pipeline")
        if pipeline in runs_by_pipeline and manifest.get("split") == split:
            runs_by_pipeline[pipeline].append(run_dir)

    return {
        pipeline: sorted(run_dirs, key=lambda p: p.name)[-1]
        for pipeline, run_dirs in runs_by_pipeline.items()
        if run_dirs
    }


def _load(run_dir: Path, filename: str) -> dict:
    return json.loads((run_dir / filename).read_text())


def load_run_artifacts(runs_dir: str | Path = RUNS_DIR, split: str = SPLIT) -> dict[str, dict[str, Any]]:
    """Loads each pipeline's latest `split` run: first-stage metrics, reranked metrics,
    decomposition metrics, and the manifest fields needed for provenance."""
    artifacts = {}
    for pipeline, run_dir in _latest_run_dirs(runs_dir, split).items():
        manifest = _load(run_dir, "manifest.json")
        artifacts[pipeline] = {
            "metrics": _load(run_dir, "metrics.json"),
            "reranked_metrics": _load(run_dir, "reranked_metrics.json"),
            "decomposition": _load(run_dir, "decomposition_metrics.json"),
            "provenance": {
                "run_dir": run_dir.name,
                "git_commit": manifest.get("git_commit"),
                "git_dirty": manifest.get("git_dirty"),
                "timestamp_utc": manifest.get("timestamp_utc"),
                "model_revision": manifest.get("model_revision"),
                "reranker_revision": manifest.get("reranker_revision"),
                "candidate_depth": manifest.get("candidate_depth"),
                "rerank_depth": manifest.get("rerank_depth"),
            },
        }
    return artifacts


def build_comparison_table(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per pipeline: first-stage recall@5/10/50, mrr@10, ndcg@10, n_queries."""
    return [
        {"pipeline": pipeline, **{col: artifacts[pipeline]["metrics"][col] for col in METRIC_COLUMNS}}
        for pipeline in PIPELINES
        if pipeline in artifacts
    ]


def build_decomposition_table(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per pipeline, columns = the plan §10.2 decomposition rates.

    All rates are denominated over every query in the split except the two conditional ones:
    `conditional_conversion_rate` (denominator: queries with a top-50 opportunity) and
    `share_of_failures_from_candidate_set` (denominator: final failures).
    """
    return [
        {
            "pipeline": pipeline,
            **{col: artifacts[pipeline]["decomposition"][col] for col in DECOMPOSITION_COLUMNS},
        }
        for pipeline in PIPELINES
        if pipeline in artifacts
    ]


def build_rerank_delta_table(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Before/after reranking metrics with the delta (plan §17 Phase 3: "reranking deltas are
    generated automatically"; plan §21 core table 3). Recall@50 is omitted because the reranked
    list is exactly the top-50 candidate set, so it cannot change by construction."""
    rows = []
    for pipeline in PIPELINES:
        if pipeline not in artifacts:
            continue
        first_stage = artifacts[pipeline]["metrics"]
        reranked = artifacts[pipeline]["reranked_metrics"]
        row: dict[str, Any] = {"pipeline": pipeline}
        for metric in DELTA_METRICS:
            row[f"{metric}_first_stage"] = first_stage[metric]
            row[f"{metric}_reranked"] = reranked[metric]
            row[f"{metric}_delta"] = reranked[metric] - first_stage[metric]
        rows.append(row)
    return rows


def load_confidence_artifacts(
    runs_dir: str | Path = RUNS_DIR, split: str = CALIBRATION_SPLIT
) -> dict[str, dict[str, Any]]:
    """Loads each pipeline's latest calibration run. These runs fit on calibration-train and
    report on calibration-dev; the split labels travel with the artifacts so the table cannot
    misattribute dev results to train."""
    artifacts = {}
    for pipeline, run_dir in _latest_run_dirs(runs_dir, split).items():
        confidence = _load(run_dir, "confidence_metrics.json")
        selective = _load(run_dir, "selective_results.json")
        manifest = _load(run_dir, "manifest.json")
        artifacts[pipeline] = {
            "confidence": confidence,
            "selective": selective,
            "provenance": {
                "run_dir": run_dir.name,
                "git_commit": manifest.get("git_commit"),
                "git_dirty": manifest.get("git_dirty"),
                "timestamp_utc": manifest.get("timestamp_utc"),
                "fit_split": confidence.get("fit_split"),
                "eval_split": confidence.get("eval_split"),
                "class_weight": manifest.get("confidence_class_weight"),
            },
        }
    return artifacts


def build_confidence_table(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Plan §21 table 5: raw score vs. calibrator, one row per (pipeline, confidence model).

    `brier` is null for the raw score by design — plan §9 forbids treating it as a probability,
    so the Platt-scaled row is the like-for-like Brier comparison."""
    rows = []
    for pipeline in PIPELINES:
        if pipeline not in artifacts:
            continue
        models = artifacts[pipeline]["confidence"]["models"]
        for model, metrics in models.items():
            rows.append(
                {
                    "pipeline": pipeline,
                    "is_primary_pipeline": pipeline == PRIMARY_PIPELINE,
                    "model": model,
                    "model_role": MODEL_ROLES[model],
                    "auroc": metrics["auroc"],
                    "auprc": metrics["auprc"],
                    "brier": metrics["brier"],
                    "n_queries": metrics["n_queries"],
                }
            )
    return rows


def build_selective_table(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Plan §21 table 6: selective success rate at the three fixed coverage levels §10.3 names
    up front (100%, 80%, 60%) — reported for every model so no favourable point can be cherry-picked."""
    rows = []
    for pipeline in PIPELINES:
        if pipeline not in artifacts:
            continue
        models = artifacts[pipeline]["selective"]["models"]
        for model, coverages in models.items():
            row: dict[str, Any] = {
                "pipeline": pipeline,
                "is_primary_pipeline": pipeline == PRIMARY_PIPELINE,
                "model": model,
                "model_role": MODEL_ROLES[model],
            }
            for level in COVERAGE_LEVELS:
                row[f"success@{level}"] = coverages[level]["success_rate"]
                row[f"n_kept@{level}"] = coverages[level]["n_kept"]
            rows.append(row)
    return rows


def render_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for col in columns:
            value = row[col]
            if value is None:
                cells.append("n/a")
            elif isinstance(value, float):
                cells.append(f"{value:+.4f}" if col.endswith("_delta") else f"{value:.4f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_table(
    rows: list[dict[str, Any]],
    columns: list[str],
    name: str,
    artifacts: dict[str, dict[str, Any]],
    split: str,
    tables_dir: str | Path = TABLES_DIR,
) -> None:
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "split": split,
        "provenance": {pipeline: artifacts[pipeline]["provenance"] for pipeline in artifacts},
        "rows": rows,
    }
    (tables_dir / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (tables_dir / f"{name}.md").write_text(render_markdown_table(rows, columns))


def write_run_manifests(
    runs_dir: str | Path = RUNS_DIR, manifests_dir: str | Path = MANIFESTS_DIR
) -> list[Path]:
    """Copies the manifest of every run backing a committed table into `results/manifests/`.

    `runs/` is regenerable and untracked (plan §14), so without this the committed numbers would
    have no version-controlled link to a git commit, model revision, or package set."""
    manifests_dir = Path(manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for split in (SPLIT, CALIBRATION_SPLIT):
        for pipeline, run_dir in _latest_run_dirs(runs_dir, split).items():
            destination = manifests_dir / f"{pipeline}_{split}.json"
            destination.write_text((run_dir / "manifest.json").read_text())
            written.append(destination)
    return written


def main(runs_dir: str | Path = RUNS_DIR, tables_dir: str | Path = TABLES_DIR) -> None:
    artifacts = load_run_artifacts(runs_dir)
    missing = [pipeline for pipeline in PIPELINES if pipeline not in artifacts]
    if missing:
        raise FileNotFoundError(
            f"No saved '{SPLIT}' run found for pipeline(s) {missing} under {runs_dir}/ "
            "— run retrieval.run for each pipeline first."
        )

    _write_table(
        build_comparison_table(artifacts),
        ["pipeline", *METRIC_COLUMNS],
        "first_stage_comparison",
        artifacts,
        SPLIT,
        tables_dir,
    )
    _write_table(
        build_decomposition_table(artifacts),
        ["pipeline", *DECOMPOSITION_COLUMNS],
        "failure_decomposition",
        artifacts,
        SPLIT,
        tables_dir,
    )
    _write_table(
        build_rerank_delta_table(artifacts),
        ["pipeline", *DELTA_COLUMNS],
        "rerank_deltas",
        artifacts,
        SPLIT,
        tables_dir,
    )

    confidence = load_confidence_artifacts(runs_dir)
    if confidence:
        _write_table(
            build_confidence_table(confidence),
            [
                "pipeline",
                "is_primary_pipeline",
                "model",
                "model_role",
                "auroc",
                "auprc",
                "brier",
                "n_queries",
            ],
            "confidence_comparison",
            confidence,
            CALIBRATION_SPLIT,
            tables_dir,
        )
        selective_columns = ["pipeline", "is_primary_pipeline", "model", "model_role"] + [
            column
            for level in COVERAGE_LEVELS
            for column in (f"success@{level}", f"n_kept@{level}")
        ]
        _write_table(
            build_selective_table(confidence),
            selective_columns,
            "selective_coverage",
            confidence,
            CALIBRATION_SPLIT,
            tables_dir,
        )

    write_run_manifests(runs_dir, Path(tables_dir).parent / "manifests")


if __name__ == "__main__":
    main()
