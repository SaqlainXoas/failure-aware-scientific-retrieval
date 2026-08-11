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


def _save_figure(
    fig, save_path: str | Path, metadata: dict[str, Any] | None = None, already_tight: bool = False
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    png_metadata = {
        "Title": save_path.stem.replace("_", " "),
        "Software": "failure-aware-scientific-retrieval",
    }
    if metadata:
        png_metadata["Description"] = json.dumps(metadata, sort_keys=True)
    if not already_tight:
        fig.tight_layout()
    fig.savefig(save_path, dpi=150, metadata=png_metadata)
    _pyplot().close(fig)


def plot_risk_coverage(curves: dict[str, list[dict[str, float]]], save_path: str | Path) -> None:
    """One risk-coverage line per confidence model, plotted together so the raw baseline, its
    Platt-scaled form, and the calibrator are compared but never conflated."""
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
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.bar(pipelines, candidate, label="candidate-set failure")
    ax.bar(pipelines, reranking, bottom=candidate, label="reranking failure")
    ax.bar(pipelines, success, bottom=candidate + reranking, label="final top-10 success")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("share of queries")
    # Every bar spans the full 0-1 range, so there is no empty region inside the axes for a
    # legend to sit in without covering a data segment. The title lives at figure level
    # (suptitle) and the legend just above the axes, with top margin reserved for both so
    # they never overlap each other or the data.
    fig.suptitle("Candidate-set vs. reranking outcomes (calibration-dev)", y=0.98)
    ax.legend(fontsize="small", loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    _save_figure(fig, save_path, metadata, already_tight=True)


def plot_reranking_transitions(
    rows: list[dict[str, Any]], save_path: str | Path, metadata: dict[str, Any]
) -> None:
    """Grouped counts for the five reranking transition labels."""
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


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial rate — well-behaved at small n and at 0/1,
    unlike a normal-approximation interval. Used so a 2-query bin visibly carries a wide
    interval instead of implying the same precision as a 76-query bin."""
    if n == 0:
        return 0.0, 0.0
    phat = successes / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half_width = (z * ((phat * (1 - phat) + z**2 / (4 * n)) / n) ** 0.5) / denom
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def plot_reliability(
    predictions: list[dict[str, Any]], save_path: str | Path, metadata: dict[str, Any]
) -> None:
    """Reliability diagram for the train-fitted Platt and primary common-feature models.

    Bins are equal-width (`reliability_bins`, unchanged), but at this sample size (162
    calibration-dev queries) several low-probability bins hold only 2-5 queries — connecting
    those with a plain line makes ordinary small-sample noise look like a real miscalibration
    pattern. Each point instead gets a 95% Wilson interval and a marker sized by its query
    count, so a 2-query swing reads as uncertain and a 76-query point reads as a real signal."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.55", label="perfect reliability", zorder=1)

    for model, label, color, marker in (
        ("raw_score_platt", "train-fitted Platt baseline", "tab:blue", "o"),
        ("calibrated", "common-feature calibrator", "tab:orange", "s"),
    ):
        points = reliability_bins(predictions, model, n_bins=10)
        xs = [point["mean_confidence"] for point in points]
        ys = [point["observed_success"] for point in points]
        counts = [point["count"] for point in points]
        intervals = [_wilson_interval(round(point["observed_success"] * point["count"]), point["count"]) for point in points]
        lower_err = [y - lo for y, (lo, _hi) in zip(ys, intervals)]
        upper_err = [hi - y for y, (_lo, hi) in zip(ys, intervals)]

        ax.errorbar(
            xs,
            ys,
            yerr=[lower_err, upper_err],
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            alpha=0.5,
            zorder=2,
        )
        ax.plot(xs, ys, linestyle=":", color=color, alpha=0.4, linewidth=1.0, zorder=2)
        ax.scatter(
            xs,
            ys,
            s=[18 + 6 * count for count in counts],
            facecolor=color,
            edgecolor="white",
            linewidth=0.6,
            marker=marker,
            label=f"{label} (marker size ∝ bin n)",
            zorder=3,
        )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed final top-10 success rate")
    ax.set_title(
        "Hybrid reliability (10 fixed bins; error bars = 95% Wilson interval)",
        fontsize=11,
    )
    ax.legend(fontsize="small", loc="lower right")
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
