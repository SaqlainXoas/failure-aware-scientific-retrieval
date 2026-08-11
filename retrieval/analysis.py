"""Phase 5 uncertainty, figures, and deterministic failure-case artifacts."""

import json
import math
from pathlib import Path
from typing import Any

from retrieval.confidence import bootstrap_mean_comparison, bootstrap_score_comparison
from retrieval.data import load_scifact_split, read_split_file, resolve_split
from retrieval.plots import (
    plot_failure_breakdown,
    plot_first_stage_performance,
    plot_hybrid_risk_coverage,
    plot_reliability,
    plot_reranking_transitions,
)
from retrieval.tables import (
    CALIBRATION_SPLIT,
    PIPELINES,
    PRIMARY_PIPELINE,
    RUNS_DIR,
    SPLIT,
    TABLES_DIR,
    _latest_run_dirs,
    _load,
    build_comparison_table,
    build_decomposition_table,
    load_run_artifacts,
)

FIGURES_DIR = "results/figures"
ANALYSIS_DIR = "analysis"
SPLITS_DIR = "splits"
BOOTSTRAP_RESAMPLES = 1_000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
CASE_QUOTAS = {
    "no_opportunity": 3,
    "unchanged_failure": 3,
    "rescued_by_reranker": 2,
    "degraded_by_reranker": 2,
}
CASE_GROUPS = {
    "no_opportunity": "candidate_set_failure",
    "unchanged_failure": "reranking_failure",
    "rescued_by_reranker": "reranker_rescue",
    "degraded_by_reranker": "reranker_degradation",
}
INTERPRETATION_LABELS = {
    "terminology mismatch",
    "lexical distraction",
    "partial topical relevance",
    "title bias",
    "entity/numerical mismatch",
    "cross-encoder overconfidence",
    "evidence outside the cutoff",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _load_parquet(path: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _source_provenance(run_dir: Path) -> dict[str, Any]:
    manifest = _load(run_dir, "manifest.json")
    return {
        "run_dir": run_dir.name,
        "run_split": manifest.get("split"),
        "pipeline": manifest.get("pipeline"),
        "git_commit": manifest.get("git_commit"),
        "git_dirty": manifest.get("git_dirty"),
        "dataset": manifest.get("dataset"),
        "model_name": manifest.get("model_name"),
        "model_revision": manifest.get("model_revision"),
        "reranker_model": manifest.get("reranker_model"),
        "reranker_revision": manifest.get("reranker_revision"),
        "candidate_depth": manifest.get("candidate_depth"),
        "rerank_depth": manifest.get("rerank_depth"),
    }


def _phase5_source_runs(runs_dir: str | Path) -> tuple[dict[str, Path], Path]:
    dev_runs = _latest_run_dirs(runs_dir, SPLIT)
    missing = [pipeline for pipeline in PIPELINES if pipeline not in dev_runs]
    if missing:
        raise FileNotFoundError(f"Missing calibration-dev source runs for {missing}")
    calibration_runs = _latest_run_dirs(runs_dir, CALIBRATION_SPLIT)
    if PRIMARY_PIPELINE not in calibration_runs:
        raise FileNotFoundError("Missing hybrid calibration-train confidence source run")

    required_dev = ("query_results.parquet", "rankings.jsonl", "reranked_rankings.jsonl")
    for pipeline, run_dir in dev_runs.items():
        manifest = _load(run_dir, "manifest.json")
        if manifest.get("git_dirty") is not False:
            raise ValueError(f"Phase 5 source run must be clean: {run_dir.name}")
        for filename in required_dev:
            if not (run_dir / filename).exists():
                raise FileNotFoundError(f"{run_dir.name}/{filename} is missing")

    confidence_run = calibration_runs[PRIMARY_PIPELINE]
    confidence_manifest = _load(confidence_run, "manifest.json")
    if confidence_manifest.get("git_dirty") is not False:
        raise ValueError(f"Phase 5 confidence source run must be clean: {confidence_run.name}")
    for filename in ("confidence_predictions.jsonl", "risk_coverage.json"):
        if not (confidence_run / filename).exists():
            raise FileNotFoundError(f"{confidence_run.name}/{filename} is missing")
    return dev_runs, confidence_run


def _rows_by_query_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        query_id = str(row["query_id"])
        if query_id in result:
            raise ValueError(f"Duplicate query row for {query_id}")
        result[query_id] = row
    return result


def _validate_dev_query_ids(
    rows_by_source: dict[str, dict[str, dict[str, Any]]],
    splits_dir: str | Path,
) -> list[str]:
    expected = set(read_split_file(Path(splits_dir) / "calibration_dev.txt"))
    if not expected:
        raise ValueError("Committed calibration-dev split is empty")
    for source, rows in rows_by_source.items():
        if set(rows) != expected:
            missing = sorted(expected - set(rows))
            extra = sorted(set(rows) - expected)
            raise ValueError(
                f"{source} query IDs do not exactly match calibration-dev "
                f"(missing={missing[:3]}, extra={extra[:3]})"
            )
    return sorted(expected)


def _rounded_bootstrap(result: dict[str, float | int]) -> dict[str, float | int]:
    numeric_fields = (
        "point_estimate_a",
        "point_estimate_b",
        "difference",
        "ci_lower",
        "ci_upper",
    )
    return {
        **result,
        **{field: round(float(result[field]), 6) for field in numeric_fields},
    }


def build_bootstrap_artifact(
    runs_dir: str | Path = RUNS_DIR,
    splits_dir: str | Path = SPLITS_DIR,
) -> dict[str, Any]:
    """Builds the eleven predeclared paired comparisons from saved calibration-dev rows."""
    dev_runs, confidence_run = _phase5_source_runs(runs_dir)
    query_rows = {
        pipeline: _rows_by_query_id(_load_parquet(run_dir / "query_results.parquet"))
        for pipeline, run_dir in dev_runs.items()
    }
    prediction_rows = _rows_by_query_id(
        _load_jsonl(confidence_run / "confidence_predictions.jsonl")
    )
    query_ids = _validate_dev_query_ids(
        {**{f"{pipeline} query results": rows for pipeline, rows in query_rows.items()},
         "hybrid confidence predictions": prediction_rows},
        splits_dir,
    )

    comparisons: list[dict[str, Any]] = []

    def add_mean(
        comparison_id: str,
        metric: str,
        label_a: str,
        label_b: str,
        role_a: str,
        role_b: str,
        values_a: dict[str, float],
        values_b: dict[str, float],
        source_pipelines: list[str],
    ) -> None:
        result = _rounded_bootstrap(
            bootstrap_mean_comparison(
                values_a,
                values_b,
                n_resamples=BOOTSTRAP_RESAMPLES,
                confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
                seed=BOOTSTRAP_SEED,
            )
        )
        comparisons.append(
            {
                "comparison_id": comparison_id,
                "metric": metric,
                "side_a_label": label_a,
                "side_b_label": label_b,
                "side_a_role": role_a,
                "side_b_role": role_b,
                **result,
                "split": SPLIT,
                "fit_split": None,
                "source_runs": [dev_runs[pipeline].name for pipeline in source_pipelines],
            }
        )

    for pipeline in PIPELINES:
        for metric in ("recall@10", "ndcg@10"):
            add_mean(
                f"{pipeline}_{metric.replace('@', '_at_')}_before_after",
                metric,
                f"{pipeline}:first_stage",
                f"{pipeline}:reranked",
                "first_stage_metric",
                "reranked_metric",
                {
                    qid: float(query_rows[pipeline][qid][f"first_stage_{metric}"])
                    for qid in query_ids
                },
                {
                    qid: float(query_rows[pipeline][qid][f"reranked_{metric}"])
                    for qid in query_ids
                },
                [pipeline],
            )

    for pipeline_a, pipeline_b in (
        ("bm25", "dense_bge"),
        ("bm25", "hybrid_rrf"),
        ("dense_bge", "hybrid_rrf"),
    ):
        add_mean(
            f"final_success_10_{pipeline_a}_vs_{pipeline_b}",
            "final_success_10",
            f"{pipeline_a}:reranked_final_success_10",
            f"{pipeline_b}:reranked_final_success_10",
            "final_top10_success",
            "final_top10_success",
            {
                qid: float(query_rows[pipeline_a][qid]["final_success_10"])
                for qid in query_ids
            },
            {
                qid: float(query_rows[pipeline_b][qid]["final_success_10"])
                for qid in query_ids
            },
            [pipeline_a, pipeline_b],
        )

    labels = {qid: bool(prediction_rows[qid]["final_success_10"]) for qid in query_ids}
    confidence_specs = (
        (
            "hybrid_auprc_raw_vs_common_calibrated",
            "auprc",
            "hybrid_rrf:raw_reranker_score",
            "hybrid_rrf:common_feature_calibrated",
            "raw_ranking_baseline",
            "common_feature_model",
            "raw_score",
            "calibrated",
            False,
            True,
        ),
        (
            "hybrid_brier_platt_vs_common_calibrated",
            "brier",
            "hybrid_rrf:train_fitted_platt",
            "hybrid_rrf:common_feature_calibrated",
            "train_fitted_probability_baseline",
            "common_feature_model",
            "raw_score_platt",
            "calibrated",
            True,
            True,
        ),
    )
    for (
        comparison_id,
        metric,
        label_a,
        label_b,
        role_a,
        role_b,
        field_a,
        field_b,
        probability_a,
        probability_b,
    ) in confidence_specs:
        result = _rounded_bootstrap(
            bootstrap_score_comparison(
                labels,
                {qid: float(prediction_rows[qid][field_a]) for qid in query_ids},
                {qid: float(prediction_rows[qid][field_b]) for qid in query_ids},
                metric=metric,
                side_a_is_probability=probability_a,
                side_b_is_probability=probability_b,
                n_resamples=BOOTSTRAP_RESAMPLES,
                confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
                seed=BOOTSTRAP_SEED,
            )
        )
        comparisons.append(
            {
                "comparison_id": comparison_id,
                "metric": metric,
                "side_a_label": label_a,
                "side_b_label": label_b,
                "side_a_role": role_a,
                "side_b_role": role_b,
                **result,
                "split": SPLIT,
                "fit_split": CALIBRATION_SPLIT,
                "source_runs": [confidence_run.name],
            }
        )

    if len(comparisons) != 11:
        raise AssertionError(f"Expected exactly 11 bootstrap comparisons, got {len(comparisons)}")
    return {
        "split": SPLIT,
        "fit_split_for_confidence": CALIBRATION_SPLIT,
        "query_ids_file": str(Path(splits_dir) / "calibration_dev.txt"),
        "bootstrap": {
            "unit": "query_id",
            "paired": True,
            "difference_direction": "B - A",
            "n_resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "interval": "percentile",
            "seed": BOOTSTRAP_SEED,
            "n_queries": len(query_ids),
        },
        "provenance": {
            **{pipeline: _source_provenance(run_dir) for pipeline, run_dir in dev_runs.items()},
            "hybrid_confidence": _source_provenance(confidence_run),
        },
        "rows": comparisons,
    }


CV_COMPARISONS = (
    ("cv_auroc_raw_vs_common_calibrated", "auroc", "raw_reranker_score", "common_feature_calibrated", "raw_score", "calibrated", False, True),
    ("cv_auprc_raw_vs_common_calibrated", "auprc", "raw_reranker_score", "common_feature_calibrated", "raw_score", "calibrated", False, True),
    ("cv_brier_platt_vs_common_calibrated", "brier", "train_fitted_platt", "common_feature_calibrated", "raw_score_platt", "calibrated", True, True),
)


def build_cv_bootstrap_artifact(runs_dir: str | Path = RUNS_DIR) -> dict[str, Any]:
    """Bootstraps the raw-vs-calibrated comparison on pooled out-of-fold predictions.

    Same paired query-level procedure as the predeclared analysis, but over all 809 calibration
    queries instead of the 162-query dev split — roughly six times the failures, which is what
    makes this comparison decisive where the dev-only version was not. Reported as a separate
    artifact so it never overwrites the predeclared train/dev result."""
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for pipeline in PIPELINES:
        run_dir = _latest_run_dirs(runs_dir, CALIBRATION_SPLIT).get(pipeline)
        if run_dir is None or not (run_dir / "confidence_cv_predictions.jsonl").exists():
            continue
        provenance[pipeline] = _source_provenance(run_dir)
        predictions = _load_jsonl(run_dir / "confidence_cv_predictions.jsonl")
        query_ids = [row["query_id"] for row in predictions]
        labels = {row["query_id"]: bool(row["final_success_10"]) for row in predictions}
        by_query = {row["query_id"]: row for row in predictions}

        for comparison_id, metric, label_a, label_b, field_a, field_b, prob_a, prob_b in CV_COMPARISONS:
            result = _rounded_bootstrap(
                bootstrap_score_comparison(
                    labels,
                    {qid: float(by_query[qid][field_a]) for qid in query_ids},
                    {qid: float(by_query[qid][field_b]) for qid in query_ids},
                    metric=metric,
                    side_a_is_probability=prob_a,
                    side_b_is_probability=prob_b,
                    n_resamples=BOOTSTRAP_RESAMPLES,
                    confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
                    seed=BOOTSTRAP_SEED,
                )
            )
            excludes_zero = result["ci_lower"] > 0.0 or result["ci_upper"] < 0.0
            rows.append(
                {
                    "comparison_id": f"{pipeline}_{comparison_id}",
                    "pipeline": pipeline,
                    "metric": metric,
                    "side_a_label": f"{pipeline}:{label_a}",
                    "side_b_label": f"{pipeline}:{label_b}",
                    **result,
                    "excludes_zero": excludes_zero,
                    "n_failures": sum(1 for qid in query_ids if not labels[qid]),
                }
            )
    return {
        "protocol": "stratified 5-fold cross-validation over calibration-train + calibration-dev",
        "why": (
            "The predeclared train/dev protocol evaluates on 162 queries holding ~20 failures, "
            "too few to separate the confidence models. Pooling out-of-fold predictions over all "
            "calibration queries raises the failure count without touching the test split."
        ),
        "bootstrap": {
            "unit": "query_id",
            "paired": True,
            "difference_direction": "B - A",
            "n_resamples": BOOTSTRAP_RESAMPLES,
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "interval": "percentile",
            "seed": BOOTSTRAP_SEED,
        },
        "provenance": provenance,
        "rows": rows,
    }


def render_cv_bootstrap_markdown(payload: dict[str, Any]) -> str:
    columns = [
        "comparison_id",
        "metric",
        "point_estimate_a",
        "point_estimate_b",
        "difference",
        "ci_lower",
        "ci_upper",
        "excludes_zero",
        "n_queries",
        "n_failures",
    ]
    lines = [
        "# Cross-validated bootstrap intervals (higher-powered secondary analysis)",
        "",
        payload["why"],
        "",
        (
            f"Protocol: {payload['protocol']}. Difference: `B - A`. "
            f"{payload['bootstrap']['n_resamples']} paired resamples, "
            f"{payload['bootstrap']['confidence_level']:.0%} percentile intervals, "
            f"seed {payload['bootstrap']['seed']}."
        ),
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    float_columns = {"point_estimate_a", "point_estimate_b", "difference", "ci_lower", "ci_upper"}
    for row in payload["rows"]:
        lines.append(
            "| "
            + " | ".join(
                f"{row[column]:.6f}" if column in float_columns else str(row[column])
                for column in columns
            )
            + " |"
        )
    provenance = ", ".join(
        f"`{name}` → `{source['run_dir']}` @ `{source['git_commit']}`"
        for name, source in payload["provenance"].items()
    )
    lines.extend(["", f"Sources: {provenance}.", ""])
    return "\n".join(lines)


def render_bootstrap_markdown(payload: dict[str, Any]) -> str:
    columns = [
        "comparison_id",
        "metric",
        "side_a_label",
        "side_b_label",
        "point_estimate_a",
        "point_estimate_b",
        "difference",
        "ci_lower",
        "ci_upper",
        "n_queries",
        "n_resamples",
        "seed",
    ]
    lines = [
        "# Paired query-level bootstrap intervals",
        "",
        (
            f"Split: `{payload['split']}`. Difference: `B - A`. "
            f"{payload['bootstrap']['n_resamples']} paired resamples, "
            f"{payload['bootstrap']['confidence_level']:.0%} percentile intervals, "
            f"seed {payload['bootstrap']['seed']}."
        ),
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    float_columns = {
        "point_estimate_a",
        "point_estimate_b",
        "difference",
        "ci_lower",
        "ci_upper",
    }
    for row in payload["rows"]:
        cells = [
            f"{row[column]:.6f}" if column in float_columns else str(row[column])
            for column in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    provenance = ", ".join(
        f"`{name}` → `{source['run_dir']}` @ `{source['git_commit']}`"
        for name, source in payload["provenance"].items()
    )
    lines.extend(["", f"Sources: {provenance}.", ""])
    return "\n".join(lines)


def write_bootstrap_artifact(
    payload: dict[str, Any], tables_dir: str | Path = TABLES_DIR
) -> tuple[Path, Path]:
    destination = Path(tables_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "bootstrap_intervals.json"
    markdown_path = destination / "bootstrap_intervals.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    markdown_path.write_text(render_bootstrap_markdown(payload))
    return json_path, markdown_path


def select_failure_cases(
    predictions: list[dict[str, Any]],
    quotas: dict[str, int] = CASE_QUOTAS,
) -> list[dict[str, Any]]:
    """Selects unique hybrid cases before content inspection using fixed confidence ranks.

    The two highest-confidence errors are selected first. Remaining transition cases use
    evenly spread confidence quantiles with query-ID string tie-breaking.
    """
    rows = sorted(predictions, key=lambda row: str(row["query_id"]))
    if len({str(row["query_id"]) for row in rows}) != len(rows):
        raise ValueError("Failure-case predictions contain duplicate query IDs")
    for row in rows:
        confidence = float(row["calibrated"])
        if not math.isfinite(confidence):
            raise ValueError("Failure-case confidence must be finite")

    incorrect = sorted(
        (row for row in rows if not row["final_success_10"]),
        key=lambda row: (-float(row["calibrated"]), str(row["query_id"])),
    )
    if len(incorrect) < 2:
        raise ValueError("Need at least two incorrect queries for high-confidence selection")
    selected = []
    used = set()
    for rank, row in enumerate(incorrect[:2], start=1):
        query_id = str(row["query_id"])
        selected.append(
            {
                "query_id": query_id,
                "selection_group": "high_confidence_incorrect",
                "transition_label": row["transition_label"],
                "calibrated_confidence": round(float(row["calibrated"]), 6),
                "selection_reason": f"incorrect confidence rank {rank} of {len(incorrect)}",
                "selection_rank": rank,
                "quantile_index": None,
            }
        )
        used.add(query_id)

    for transition, quota in quotas.items():
        candidates = sorted(
            (
                row
                for row in rows
                if row["transition_label"] == transition and str(row["query_id"]) not in used
            ),
            key=lambda row: (float(row["calibrated"]), str(row["query_id"])),
        )
        if len(candidates) < quota:
            raise ValueError(
                f"Transition {transition} has {len(candidates)} remaining cases; needs {quota}"
            )
        indices = [min(len(candidates) - 1, math.floor((j + 0.5) * len(candidates) / quota))
                   for j in range(quota)]
        if len(set(indices)) != quota:
            raise AssertionError(f"Quantile selection for {transition} produced duplicate indices")
        for index in indices:
            row = candidates[index]
            query_id = str(row["query_id"])
            selected.append(
                {
                    "query_id": query_id,
                    "selection_group": CASE_GROUPS[transition],
                    "transition_label": transition,
                    "calibrated_confidence": round(float(row["calibrated"]), 6),
                    "selection_reason": (
                        f"confidence-spread index {index} of {len(candidates)} eligible "
                        f"{transition} cases"
                    ),
                    "selection_rank": None,
                    "quantile_index": index,
                }
            )
            used.add(query_id)

    if len(selected) != 12 or len(used) != 12:
        raise AssertionError("Failure-case selection must contain exactly 12 unique queries")
    return selected


def build_failure_case_selection_artifact(
    confidence_run: Path,
    splits_dir: str | Path = SPLITS_DIR,
    ranking_run: Path | None = None,
) -> dict[str, Any]:
    predictions = _load_jsonl(confidence_run / "confidence_predictions.jsonl")
    prediction_rows = _rows_by_query_id(predictions)
    _validate_dev_query_ids({"hybrid confidence predictions": prediction_rows}, splits_dir)
    source_runs = [confidence_run.name]
    provenance = {"confidence_predictions": _source_provenance(confidence_run)}
    if ranking_run is not None:
        source_runs.append(ranking_run.name)
        provenance["rankings"] = _source_provenance(ranking_run)
    cases = [
        {
            **case,
            "split": SPLIT,
            "pipeline": PRIMARY_PIPELINE,
            "confidence_model": "calibrated",
            "source_runs": source_runs,
        }
        for case in select_failure_cases(predictions)
    ]
    return {
        "split": SPLIT,
        "pipeline": PRIMARY_PIPELINE,
        "confidence_model": "calibrated",
        "fit_split": CALIBRATION_SPLIT,
        "selection_order": [
            "two highest common-feature-confidence incorrect cases",
            "transition cases from ascending-confidence spread quantiles after exclusions",
        ],
        "quantile_index_formula": "floor((j + 0.5) * n / quota)",
        "query_id_tie_break": "ascending opaque string",
        "quotas": {
            "candidate_set_failure": 3,
            "reranking_failure": 3,
            "reranker_rescue": 2,
            "reranker_degradation": 2,
            "high_confidence_incorrect": 2,
        },
        "provenance": provenance,
        "cases": cases,
    }


def _ranking_rows_by_query(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["query_id"]), []).append(row)
    return {
        query_id: sorted(query_rows, key=lambda row: int(row["rank"]))
        for query_id, query_rows in grouped.items()
    }


def _document_parts(document: str) -> tuple[str, str]:
    title, separator, abstract = document.partition("\n")
    return title.strip() or "(untitled)", abstract.strip() if separator else ""


def _markdown_text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _representative_gold(
    qrels: dict[str, int], first_stage_rows: list[dict[str, Any]]
) -> str:
    first_rank = {str(row["doc_id"]): int(row["rank"]) for row in first_stage_rows}
    positive = [(str(doc_id), int(grade)) for doc_id, grade in qrels.items() if grade > 0]
    if not positive:
        raise ValueError("Selected query has no positive qrel")
    return min(
        positive,
        key=lambda item: (-item[1], first_rank.get(item[0], math.inf), item[0]),
    )[0]


def _format_rank(rank_by_doc: dict[str, int], doc_id: str, missing: str) -> str:
    return str(rank_by_doc[doc_id]) if doc_id in rank_by_doc else missing


def _ranking_markdown(
    rows: list[dict[str, Any]],
    corpus: dict[str, str],
    gold_doc_ids: set[str],
) -> list[str]:
    lines = ["| Rank | Document | Score | Gold |", "| ---: | --- | ---: | :---: |"]
    for row in rows[:5]:
        doc_id = str(row["doc_id"])
        title, _ = _document_parts(corpus[doc_id])
        lines.append(
            f"| {row['rank']} | {_markdown_text(title)} (`{doc_id}`) | "
            f"{float(row['score']):.6f} | {'yes' if doc_id in gold_doc_ids else ''} |"
        )
    return lines


def render_failure_cases(
    selection: dict[str, Any],
    annotations: dict[str, Any],
    data: dict[str, Any],
    first_stage_rows: list[dict[str, Any]],
    reranked_rows: list[dict[str, Any]],
) -> str:
    """Renders saved ranks/scores automatically while keeping human interpretations separate."""
    cases = selection["cases"]
    annotations_by_query = annotations.get("cases", {})
    selected_ids = {case["query_id"] for case in cases}
    if set(annotations_by_query) != selected_ids:
        raise ValueError("Failure-case annotations must exactly match selected query IDs")

    first_by_query = _ranking_rows_by_query(first_stage_rows)
    reranked_by_query = _ranking_rows_by_query(reranked_rows)
    lines = [
        "# Structured Calibration-Dev Failure Analysis",
        "",
        (
            "These 12 hybrid-RRF cases were selected by the predefined confidence-ranking "
            "rules in `results/tables/failure_case_selection.json`, before reading case content. "
            "Scores and ranks below are rendered from saved run artifacts; interpretations are "
            "retrieval-error descriptions, not support/refute judgments."
        ),
        "",
        f"- Split: `{selection['split']}`",
        f"- Pipeline: `{selection['pipeline']}`",
        f"- Confidence model: `{selection['confidence_model']}` fitted on `{selection['fit_split']}`",
        (
            f"- Confidence source: "
            f"`{selection['provenance']['confidence_predictions']['run_dir']}`"
        ),
        (
            f"- Ranking source: "
            f"`{selection['provenance']['rankings']['run_dir']}`"
        ),
        "",
    ]

    for number, case in enumerate(cases, start=1):
        query_id = case["query_id"]
        annotation = annotations_by_query[query_id]
        first_rows = first_by_query[query_id]
        reranked = reranked_by_query[query_id]
        qrels = data["qrels"][query_id]
        gold_doc_ids = {str(doc_id) for doc_id, grade in qrels.items() if grade > 0}
        representative = _representative_gold(qrels, first_rows)
        if annotation.get("gold_doc_id") != representative:
            raise ValueError(
                f"Annotation gold_doc_id for {query_id} must be representative {representative}"
            )
        excerpt = str(annotation.get("excerpt", "")).strip()
        if not excerpt or len(excerpt.split()) > 25:
            raise ValueError(f"Excerpt for {query_id} must contain 1-25 words")
        if excerpt not in data["corpus"][representative]:
            raise ValueError(f"Excerpt for {query_id} is not verbatim in representative document")
        labels = annotation.get("interpretation_labels", [])
        if not labels or not set(labels) <= INTERPRETATION_LABELS:
            raise ValueError(f"Unknown or missing interpretation label for {query_id}")
        interpretation = str(annotation.get("interpretation", "")).strip()
        if not interpretation:
            raise ValueError(f"Interpretation for {query_id} is empty")

        title, _ = _document_parts(data["corpus"][representative])
        first_rank = {str(row["doc_id"]): int(row["rank"]) for row in first_rows}
        reranked_rank = {str(row["doc_id"]): int(row["rank"]) for row in reranked}
        lines.extend(
            [
                f"## {number}. Query `{query_id}` — {case['selection_group']}",
                "",
                f"**Claim:** {_markdown_text(data['queries'][query_id])}",
                "",
                f"- Transition: `{case['transition_label']}`",
                f"- Selection reason: {case['selection_reason']}",
                f"- Common-feature confidence: {case['calibrated_confidence']:.6f}",
                "",
                "### Representative gold evidence",
                "",
                f"**{_markdown_text(title)}** (`{representative}`)",
                "",
                f"> {_markdown_text(excerpt)}",
                "",
                (
                    f"Saved position: first stage "
                    f"{_format_rank(first_rank, representative, 'not in top-100')}; reranked "
                    f"{_format_rank(reranked_rank, representative, 'not in top-50')}."
                ),
                "",
                "### First-stage top five",
                "",
                *_ranking_markdown(first_rows, data["corpus"], gold_doc_ids),
                "",
                "### Reranked top five",
                "",
                *_ranking_markdown(reranked, data["corpus"], gold_doc_ids),
                "",
                "### Interpretation",
                "",
                f"Labels: {', '.join(f'`{label}`' for label in labels)}.",
                "",
                interpretation,
                "",
            ]
        )
    return "\n".join(lines)


def load_failure_case_data(splits_dir: str | Path = SPLITS_DIR) -> dict[str, Any]:
    """Loads only SciFact train, then filters it to the committed calibration-dev IDs."""
    train_data = load_scifact_split("train")
    return resolve_split(SPLIT, train_data, splits_dir=splits_dir)


def write_phase5_artifacts(
    runs_dir: str | Path = RUNS_DIR,
    tables_dir: str | Path = TABLES_DIR,
    figures_dir: str | Path = FIGURES_DIR,
    analysis_dir: str | Path = ANALYSIS_DIR,
    splits_dir: str | Path = SPLITS_DIR,
) -> None:
    dev_runs, confidence_run = _phase5_source_runs(runs_dir)
    bootstrap = build_bootstrap_artifact(runs_dir, splits_dir)
    write_bootstrap_artifact(bootstrap, tables_dir)

    cv_bootstrap = build_cv_bootstrap_artifact(runs_dir)
    if cv_bootstrap["rows"]:
        destination = Path(tables_dir)
        (destination / "cv_bootstrap_intervals.json").write_text(
            json.dumps(cv_bootstrap, indent=2) + "\n"
        )
        (destination / "cv_bootstrap_intervals.md").write_text(
            render_cv_bootstrap_markdown(cv_bootstrap)
        )

    selection = build_failure_case_selection_artifact(
        confidence_run,
        splits_dir,
        ranking_run=dev_runs[PRIMARY_PIPELINE],
    )
    selection_path = Path(tables_dir) / "failure_case_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")

    artifacts = load_run_artifacts(runs_dir)
    first_stage = build_comparison_table(artifacts)
    decomposition = build_decomposition_table(artifacts)
    transitions = [
        {
            "pipeline": pipeline,
            "counts": artifacts[pipeline]["decomposition"]["counts"],
        }
        for pipeline in PIPELINES
    ]
    predictions = _load_jsonl(confidence_run / "confidence_predictions.jsonl")
    risk_coverage = _load(confidence_run, "risk_coverage.json")["models"]

    figures_dir = Path(figures_dir)
    figure_sources = {
        **{pipeline: _source_provenance(run_dir) for pipeline, run_dir in dev_runs.items()},
        "hybrid_confidence": _source_provenance(confidence_run),
    }
    common_metadata = {
        "split": SPLIT,
        "generated_by": "python -m retrieval.tables",
        "source_runs": figure_sources,
    }
    figure_specs = {
        "first_stage_performance.png": ["bm25", "dense_bge", "hybrid_rrf"],
        "failure_breakdown.png": ["bm25", "dense_bge", "hybrid_rrf"],
        "reranking_transitions.png": ["bm25", "dense_bge", "hybrid_rrf"],
        "hybrid_reliability.png": ["hybrid_confidence"],
        "hybrid_risk_coverage.png": ["hybrid_confidence"],
    }
    plot_first_stage_performance(
        first_stage, figures_dir / "first_stage_performance.png", common_metadata
    )
    plot_failure_breakdown(
        decomposition, figures_dir / "failure_breakdown.png", common_metadata
    )
    plot_reranking_transitions(
        transitions, figures_dir / "reranking_transitions.png", common_metadata
    )
    plot_reliability(
        predictions, figures_dir / "hybrid_reliability.png", common_metadata
    )
    plot_hybrid_risk_coverage(
        risk_coverage, figures_dir / "hybrid_risk_coverage.png", common_metadata
    )
    figure_provenance = {
        "split": SPLIT,
        "generated_by": "python -m retrieval.tables",
        "empty_reliability_bins": "omitted; retained points are annotated with sample counts",
        "sources": figure_sources,
        "figures": {
            filename: {"source_keys": source_keys, "split": SPLIT}
            for filename, source_keys in figure_specs.items()
        },
    }
    (figures_dir / "figure_provenance.json").write_text(
        json.dumps(figure_provenance, indent=2) + "\n"
    )

    dev_data = load_failure_case_data(splits_dir)
    selected_ids = {case["query_id"] for case in selection["cases"]}
    if not selected_ids <= set(dev_data["queries"]):
        raise ValueError("Selected failure cases are not all in calibration-dev")
    annotations_path = Path(analysis_dir) / "failure_case_annotations.json"
    if not annotations_path.exists():
        raise FileNotFoundError(
            f"{annotations_path} is required to render the manual failure analysis"
        )
    annotations = json.loads(annotations_path.read_text())
    hybrid_run = dev_runs[PRIMARY_PIPELINE]
    failure_markdown = render_failure_cases(
        selection,
        annotations,
        dev_data,
        _load_jsonl(hybrid_run / "rankings.jsonl"),
        _load_jsonl(hybrid_run / "reranked_rankings.jsonl"),
    )
    (Path(analysis_dir) / "failure_cases.md").write_text(failure_markdown)
