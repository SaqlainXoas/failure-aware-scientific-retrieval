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


def load_run_metrics(runs_dir: str | Path = RUNS_DIR, split: str = SPLIT) -> dict[str, dict]:
    """Finds the most recent run dir per pipeline (by timestamp-prefixed name) under runs_dir and loads its metrics.json."""
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

    metrics: dict[str, dict] = {}
    for pipeline, run_dirs in runs_by_pipeline.items():
        if not run_dirs:
            continue
        latest = sorted(run_dirs, key=lambda p: p.name)[-1]
        metrics[pipeline] = json.loads((latest / "metrics.json").read_text())
    return metrics


def build_comparison_table(pipeline_metrics: dict[str, dict]) -> list[dict[str, Any]]:
    """One row per pipeline: recall@5/10/50, mrr@10, ndcg@10, n_queries — sourced only from saved metrics.json."""
    return [
        {"pipeline": pipeline, **{col: pipeline_metrics[pipeline][col] for col in METRIC_COLUMNS}}
        for pipeline in PIPELINES
        if pipeline in pipeline_metrics
    ]


def render_markdown_table(rows: list[dict[str, Any]]) -> str:
    header = ["pipeline", *METRIC_COLUMNS]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows:
        cells = [row["pipeline"]] + [
            f"{row[col]:.4f}" if isinstance(row[col], float) else str(row[col])
            for col in METRIC_COLUMNS
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    pipeline_metrics = load_run_metrics()
    missing = [p for p in PIPELINES if p not in pipeline_metrics]
    if missing:
        raise FileNotFoundError(
            f"No saved '{SPLIT}' run found for pipeline(s) {missing} under {RUNS_DIR}/ "
            "— run retrieval.run for each pipeline first."
        )

    rows = build_comparison_table(pipeline_metrics)

    output_json = Path(OUTPUT_JSON)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(rows, indent=2) + "\n")

    Path(OUTPUT_MD).write_text(render_markdown_table(rows))


if __name__ == "__main__":
    main()
