"""Builds the first-stage ranking-comparison table from saved run outputs on calibration-dev."""

import json
from pathlib import Path
from typing import Any

RUNS_DIR = "runs"
SPLIT = "calibration-dev"
PIPELINES = ["bm25", "dense_bge", "hybrid_rrf"]
OUTPUT_JSON = "results/tables/first_stage_comparison.json"
OUTPUT_MD = "results/tables/first_stage_comparison.md"
METRIC_COLUMNS = ["recall@5", "recall@10", "recall@50", "mrr@10", "ndcg@10", "n_queries"]
DECOMPOSITION_OUTPUT_JSON = "results/tables/failure_decomposition.json"
DECOMPOSITION_OUTPUT_MD = "results/tables/failure_decomposition.md"
DECOMPOSITION_COLUMNS = [
    "candidate_set_failure_rate",
    "reranking_failure_rate",
    "final_success_rate",
    "opportunity_rate",
    "rescue_rate",
    "degradation_rate",
    "conditional_conversion_rate",
]


def _latest_run_dirs(runs_dir: str | Path = RUNS_DIR, split: str = SPLIT) -> dict[str, Path]:
    """Finds the most recent run dir per pipeline (by timestamp-prefixed name) under runs_dir."""
    runs_by_pipeline: dict[str, list[Path]] = {pipeline: [] for pipeline in PIPELINES}
    for run_dir in Path(runs_dir).iterdir():
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


def load_run_metrics(runs_dir: str | Path = RUNS_DIR, split: str = SPLIT) -> dict[str, dict]:
    """Loads each pipeline's latest run's metrics.json (first-stage) for `split`."""
    return {
        pipeline: json.loads((run_dir / "metrics.json").read_text())
        for pipeline, run_dir in _latest_run_dirs(runs_dir, split).items()
    }


def load_decomposition_metrics(runs_dir: str | Path = RUNS_DIR, split: str = SPLIT) -> dict[str, dict]:
    """Loads each pipeline's latest run's decomposition_metrics.json for `split`."""
    return {
        pipeline: json.loads((run_dir / "decomposition_metrics.json").read_text())
        for pipeline, run_dir in _latest_run_dirs(runs_dir, split).items()
    }


def build_comparison_table(pipeline_metrics: dict[str, dict]) -> list[dict[str, Any]]:
    """One row per pipeline: recall@5/10/50, mrr@10, ndcg@10, n_queries — sourced only from saved metrics.json."""
    return [
        {"pipeline": pipeline, **{col: pipeline_metrics[pipeline][col] for col in METRIC_COLUMNS}}
        for pipeline in PIPELINES
        if pipeline in pipeline_metrics
    ]


def build_decomposition_table(pipeline_decomposition: dict[str, dict]) -> list[dict[str, Any]]:
    """One row per pipeline, columns = the plan.md §10.2 decomposition rates — sourced only from
    saved decomposition_metrics.json, never recomputed by hand."""
    return [
        {"pipeline": pipeline, **{col: pipeline_decomposition[pipeline][col] for col in DECOMPOSITION_COLUMNS}}
        for pipeline in PIPELINES
        if pipeline in pipeline_decomposition
    ]


def render_markdown_table(rows: list[dict[str, Any]], columns: list[str] = METRIC_COLUMNS) -> str:
    header = ["pipeline", *columns]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        cells = [row["pipeline"]] + [
            f"{row[col]:.4f}" if isinstance(row[col], float) else str(row[col]) for col in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_table(rows: list[dict[str, Any]], columns: list[str], output_json: str, output_md: str) -> None:
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    Path(output_md).write_text(render_markdown_table(rows, columns))


def main() -> None:
    pipeline_metrics = load_run_metrics()
    pipeline_decomposition = load_decomposition_metrics()
    missing = [p for p in PIPELINES if p not in pipeline_metrics or p not in pipeline_decomposition]
    if missing:
        raise FileNotFoundError(
            f"No saved '{SPLIT}' run found for pipeline(s) {missing} under {RUNS_DIR}/ "
            "— run retrieval.run for each pipeline first."
        )

    _write_table(build_comparison_table(pipeline_metrics), METRIC_COLUMNS, OUTPUT_JSON, OUTPUT_MD)
    _write_table(
        build_decomposition_table(pipeline_decomposition),
        DECOMPOSITION_COLUMNS,
        DECOMPOSITION_OUTPUT_JSON,
        DECOMPOSITION_OUTPUT_MD,
    )


if __name__ == "__main__":
    main()
