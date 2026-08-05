"""Restrained Phase 4/5 result plotting from saved artifacts."""

import json
import os
from pathlib import Path
from typing import Any

LABELS = {
    "raw_score": "raw reranker score (baseline)",
    "raw_score_platt": "raw score, Platt-scaled",
    "calibrated": "calibrated (common features)",
    "calibrated_hybrid_exploratory": "calibrated + BM25/dense overlap (exploratory)",
}


def _pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_figure(fig, save_path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    png_metadata = {
        "Title": save_path.stem.replace("_", " "),
        "Software": "failure-aware-scientific-retrieval",
    }
    if metadata:
        png_metadata["Description"] = json.dumps(metadata, sort_keys=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, metadata=png_metadata)
    _pyplot().close(fig)


def plot_risk_coverage(curves: dict[str, list[dict[str, float]]], save_path: str | Path) -> None:
    """One risk-coverage line per confidence model, plotted together so the raw baseline, its
    Platt-scaled form, and the calibrator are compared but never conflated (plan §9)."""
    plt = _pyplot()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, curve in curves.items():
        if curve:
            ax.plot(
                [p["coverage"] for p in curve],
                [p["risk"] for p in curve],
                label=LABELS.get(name, name),
            )
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk (1 - selective success rate)")
    ax.set_title("Risk-coverage: raw score vs. calibrated confidence")
    ax.legend(fontsize="small")
    _save_figure(fig, save_path)


def plot_first_stage_performance(
    rows: list[dict[str, Any]], save_path: str | Path, metadata: dict[str, Any]
) -> None:
    """Grouped comparison of the three predeclared first-stage retrieval metrics."""
    import numpy as np

    plt = _pyplot()
    metrics = ["recall@10", "recall@50", "ndcg@10"]
    labels = ["Recall@10", "Recall@50", "nDCG@10"]
    pipelines = [row["pipeline"] for row in rows]
    positions = np.arange(len(pipelines))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for offset, (metric, label) in enumerate(zip(metrics, labels)):
        ax.bar(
            positions + (offset - 1) * width,
            [row[metric] for row in rows],
            width,
            label=label,
        )
    ax.set_xticks(positions, pipelines)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("First-stage retrieval performance (calibration-dev)")
    ax.legend(fontsize="small")
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, save_path, metadata)


def plot_failure_breakdown(
    rows: list[dict[str, Any]], save_path: str | Path, metadata: dict[str, Any]
) -> None:
    """Stacked all-query partition: candidate failure, reranking failure, final success."""
    import numpy as np

    plt = _pyplot()
    pipelines = [row["pipeline"] for row in rows]
    candidate = np.array([row["candidate_set_failure_rate"] for row in rows])
    reranking = np.array([row["reranking_failure_rate"] for row in rows])
    success = np.array([row["final_success_rate"] for row in rows])
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.bar(pipelines, candidate, label="candidate-set failure")
    ax.bar(pipelines, reranking, bottom=candidate, label="reranking failure")
    ax.bar(pipelines, success, bottom=candidate + reranking, label="final top-10 success")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("share of queries")
    ax.set_title("Candidate-set vs. reranking outcomes (calibration-dev)")
    ax.legend(fontsize="small", loc="lower right")
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, save_path, metadata)


def plot_reranking_transitions(
    rows: list[dict[str, Any]], save_path: str | Path, metadata: dict[str, Any]
) -> None:
    """Grouped counts for the five exact plan §8 reranking transition labels."""
    import numpy as np

    plt = _pyplot()
    categories = [
        "already_successful",
        "rescued_by_reranker",
        "degraded_by_reranker",
        "unchanged_failure",
        "no_opportunity",
    ]
    labels = ["already successful", "rescued", "degraded", "unchanged failure", "no opportunity"]
    pipelines = [row["pipeline"] for row in rows]
    positions = np.arange(len(categories))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for offset, row in enumerate(rows):
        ax.bar(
            positions + (offset - 1) * width,
            [row["counts"][category] for category in categories],
            width,
            label=row["pipeline"],
        )
    ax.set_xticks(positions, labels, rotation=18, ha="right")
    ax.set_ylabel("query count")
    ax.set_title("Reranking transitions (calibration-dev)")
    ax.legend(fontsize="small")
    ax.grid(axis="y", alpha=0.2)
    _save_figure(fig, save_path, metadata)


def reliability_bins(
    predictions: list[dict[str, Any]], model: str, n_bins: int = 10
) -> list[dict[str, float | int]]:
    """Fixed equal-width reliability bins; [lower, upper), with 1.0 in the final bin.

    Empty bins are deliberately omitted from the returned points. Their handling is stated in
    the figure subtitle, while every retained point carries its sample count.
    """
    import math

    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for row in predictions:
        probability = float(row[model])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"{model} contains a non-probability value")
        index = min(int(probability * n_bins), n_bins - 1)
        buckets[index].append((probability, bool(row["final_success_10"])))

    points = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        points.append(
            {
                "bin": index,
                "lower": index / n_bins,
                "upper": (index + 1) / n_bins,
                "mean_confidence": sum(probability for probability, _ in bucket) / len(bucket),
                "observed_success": sum(target for _, target in bucket) / len(bucket),
                "count": len(bucket),
            }
        )
    return points


def plot_reliability(
    predictions: list[dict[str, Any]], save_path: str | Path, metadata: dict[str, Any]
) -> None:
    """Reliability diagram for the train-fitted Platt and primary common-feature models."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.55", label="perfect reliability")
    for model, label, marker in (
        ("raw_score_platt", "train-fitted Platt baseline", "o"),
        ("calibrated", "common-feature calibrator", "s"),
    ):
        points = reliability_bins(predictions, model, n_bins=10)
        ax.plot(
            [point["mean_confidence"] for point in points],
            [point["observed_success"] for point in points],
            marker=marker,
            label=label,
        )
        for point in points:
            near_ceiling = point["observed_success"] >= 0.95
            offset = (
                (0, -12 if model == "raw_score_platt" else -24)
                if near_ceiling
                else (3, 4)
            )
            ax.annotate(
                f"n={point['count']}",
                (point["mean_confidence"], point["observed_success"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
                ha="center" if near_ceiling else "left",
            )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed final top-10 success rate")
    ax.set_title("Hybrid reliability (10 fixed bins; empty bins omitted)")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.2)
    _save_figure(fig, save_path, metadata)


def plot_hybrid_risk_coverage(
    curves: dict[str, list[dict[str, float]]],
    save_path: str | Path,
    metadata: dict[str, Any],
) -> None:
    """Hybrid raw-score ordering versus the primary common-feature confidence model."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for model, label in (
        ("raw_score", "raw reranker score"),
        ("calibrated", "common-feature calibrator"),
    ):
        curve = curves[model]
        ax.plot(
            [point["coverage"] for point in curve],
            [point["risk"] for point in curve],
            label=label,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk (1 - selective success rate)")
    ax.set_title("Hybrid risk-coverage (calibration-dev)")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.2)
    _save_figure(fig, save_path, metadata)
