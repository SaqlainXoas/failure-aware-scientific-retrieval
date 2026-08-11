"""Builds the result tables from saved run outputs on calibration-dev.

Every number is read out of a run directory's saved JSON — nothing is recomputed or typed by
hand. Each table carries a provenance block naming its run directory, git commit, and model
revisions, because `runs/` is not tracked.
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

# Markdown-only presentation. The JSON keys stay machine-readable and unabbreviated; these
# labels and formats exist so the .md files are legible without a decoder ring.
COLUMN_LABELS = {
    "pipeline": "Pipeline",
    "is_primary_pipeline": "Primary",
    "model": "Confidence model",
    "model_role": "Role",
    "metric": "Metric",
    "comparison_id": "Comparison",
    "n_queries": "Queries",
    "n_failures": "Failures",
    "n_resamples": "Resamples",
    "seed": "Seed",
    "mrr@10": "MRR@10",
    "ndcg@10": "nDCG@10",
    "auroc": "AUROC",
    "auprc": "AUPRC",
    "brier": "Brier",
    "candidate_set_failure_rate": "Candidate-set failure",
    "reranking_failure_rate": "Reranking failure",
    "final_success_rate": "Final success",
    "opportunity_rate": "Had opportunity",
    "rescue_rate": "Rescued",
    "degradation_rate": "Degraded",
    "conditional_conversion_rate": "Success given opportunity",
    "share_of_failures_from_candidate_set": "Failures from candidate set",
    "side_a_label": "Side A",
    "side_b_label": "Side B",
    "point_estimate_a": "A",
    "point_estimate_b": "B",
    "difference": "Difference (B − A)",
    "excludes_zero": "Excludes 0",
    **{f"recall@{k}": f"Recall@{k}" for k in (5, 10, 50)},
    **{
        label: text
        for metric, name in zip(DELTA_METRICS, ("Recall@5", "Recall@10", "MRR@10", "nDCG@10"))
        for label, text in (
            (f"{metric}_first_stage", f"{name} before"),
            (f"{metric}_reranked", f"{name} after"),
            (f"{metric}_delta", f"{name} Δ"),
        )
    },
    **{
        label: text
        for level in COVERAGE_LEVELS
        for label, text in (
            (f"success@{level}", f"Success @{float(level):.0%}"),
            (f"n_kept@{level}", f"Kept @{float(level):.0%}"),
        )
    },
}

# Rendered as percentages because they are shares of a query population, where "11.7%" is
# immediately meaningful and "0.1173" is not. Ranking metrics keep the decimal form the IR
# literature uses, so they stay comparable with published numbers.
PERCENT_COLUMNS = frozenset(DECOMPOSITION_COLUMNS)


# Metric identifiers appear as cell values in the bootstrap tables, where they read as
# acronyms rather than as JSON keys.
METRIC_LABELS = {"auroc": "AUROC", "auprc": "AUPRC", "brier": "Brier", "ndcg@10": "nDCG@10",
                 "mrr@10": "MRR@10", "recall@10": "Recall@10", "recall@5": "Recall@5",
                 "recall@50": "Recall@50", "final_success_10": "Final success@10"}
MODEL_LABELS = {
    "raw_score": "raw reranker score",
    "raw_score_platt": "raw score, Platt-scaled",
    "calibrated": "common-feature calibrator",
    "calibrated_hybrid_exploratory": "calibrator + BM25/dense overlap",
}
MODEL_ROLE_LABELS = {
    "raw_ranking_baseline": "ranking baseline",
    "train_fitted_probability_baseline": "probability baseline",
    "common_feature_model": "primary",
    "exploratory_overlap_ablation": "exploratory ablation",
}


def column_label(column: str) -> str:
    return COLUMN_LABELS.get(column, column)


def format_cell(column: str, value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if column == "metric":
        return METRIC_LABELS.get(value, str(value))
    if column == "model":
        return MODEL_LABELS.get(value, str(value))
    if column == "model_role":
        return MODEL_ROLE_LABELS.get(value, str(value))
    if isinstance(value, float):
        if column in PERCENT_COLUMNS:
            return f"{value:.1%}"
        if column.endswith("_delta") or column == "difference":
            return f"{value:+.3f}"
        return f"{value:.3f}"
    return str(value)


def markdown_header(columns: list[str], numeric: list[bool] | None = None) -> list[str]:
    """Header and alignment rows; numeric columns are right-aligned so digits line up."""
    numeric = numeric or [False] * len(columns)
    return [
        "| " + " | ".join(column_label(column) for column in columns) + " |",
        "| " + " | ".join("---:" if is_numeric else "---" for is_numeric in numeric) + " |",
    ]


def _numeric_columns(rows: list[dict[str, Any]], columns: list[str]) -> list[bool]:
    return [
        any(
            isinstance(row.get(column), (int, float)) and not isinstance(row.get(column), bool)
            for row in rows
        )
        for column in columns
    ]


def latest_run_dirs(runs_dir: str | Path = RUNS_DIR, split: str = SPLIT) -> dict[str, Path]:
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


def load_json(run_dir: Path, filename: str) -> dict:
    return json.loads((run_dir / filename).read_text())


def load_run_artifacts(runs_dir: str | Path = RUNS_DIR, split: str = SPLIT) -> dict[str, dict[str, Any]]:
    """Loads each pipeline's latest `split` run: first-stage metrics, reranked metrics,
    decomposition metrics, and the manifest fields needed for provenance."""
    artifacts = {}
    for pipeline, run_dir in latest_run_dirs(runs_dir, split).items():
        manifest = load_json(run_dir, "manifest.json")
        artifacts[pipeline] = {
            "metrics": load_json(run_dir, "metrics.json"),
            "reranked_metrics": load_json(run_dir, "reranked_metrics.json"),
            "decomposition": load_json(run_dir, "decomposition_metrics.json"),
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
    """One row per pipeline, columns = the failure-decomposition rates."""
    return [
        {
            "pipeline": pipeline,
            **{col: artifacts[pipeline]["decomposition"][col] for col in DECOMPOSITION_COLUMNS},
        }
        for pipeline in PIPELINES
        if pipeline in artifacts
    ]


def build_rerank_delta_table(artifacts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Before/after reranking metrics with the delta. Recall@50 is omitted because the reranked
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
    for pipeline, run_dir in latest_run_dirs(runs_dir, split).items():
        confidence = load_json(run_dir, "confidence_metrics.json")
        selective = load_json(run_dir, "selective_results.json")
        manifest = load_json(run_dir, "manifest.json")
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
    """Raw score vs. calibrator, one row per (pipeline, confidence model).

    `brier` is null for the raw score by design — it is not a probability, so the Platt-scaled
    row is the like-for-like Brier comparison."""
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
    """Selective success rate at the three coverage levels fixed up front (100%, 80%, 60%) —
    reported for every model so no favourable point can be cherry-picked."""
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
    lines = markdown_header(columns, _numeric_columns(rows, columns))
    for row in rows:
        lines.append("| " + " | ".join(format_cell(col, row[col]) for col in columns) + " |")
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

    `runs/` is regenerable and untracked, so without this the committed numbers would
    have no version-controlled link to a git commit, model revision, or package set."""
    manifests_dir = Path(manifests_dir)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for split in (SPLIT, CALIBRATION_SPLIT):
        for pipeline, run_dir in latest_run_dirs(runs_dir, split).items():
            destination = manifests_dir / f"{pipeline}_{split}.json"
            destination.write_text((run_dir / "manifest.json").read_text())
            written.append(destination)
    return written


def main(runs_dir: str | Path = RUNS_DIR, tables_dir: str | Path = TABLES_DIR) -> None:
    from retrieval.analysis import write_phase5_artifacts

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
    write_phase5_artifacts(
        runs_dir=runs_dir,
        tables_dir=tables_dir,
        figures_dir=Path(tables_dir).parent / "figures",
    )


if __name__ == "__main__":
    main()
