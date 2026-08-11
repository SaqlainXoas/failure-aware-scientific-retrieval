"""Deterministic selection and rendering of the hand-annotated failure cases.

Cases are picked by fixed confidence-rank rules before any case text is read, so the
qualitative section cannot be assembled from whichever examples happen to make a point.
"""

import math
from pathlib import Path
from typing import Any

from retrieval.analysis import (
    SPLITS_DIR,
    load_jsonl,
    rows_by_query_id,
    source_provenance,
    validate_dev_query_ids,
)
from retrieval.data import load_scifact_split, resolve_split
from retrieval.tables import CALIBRATION_SPLIT, PRIMARY_PIPELINE, SPLIT


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
    predictions = load_jsonl(confidence_run / "confidence_predictions.jsonl")
    prediction_rows = rows_by_query_id(predictions)
    validate_dev_query_ids({"hybrid confidence predictions": prediction_rows}, splits_dir)
    source_runs = [confidence_run.name]
    provenance = {"confidence_predictions": source_provenance(confidence_run)}
    if ranking_run is not None:
        source_runs.append(ranking_run.name)
        provenance["rankings"] = source_provenance(ranking_run)
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
