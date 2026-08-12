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

# Okabe-Ito, which stays distinguishable under the common forms of colour blindness. One fixed
# colour per pipeline across every figure, so a reader learns the mapping once.
PIPELINE_COLORS = {"bm25": "#0072B2", "dense_bge": "#D55E00", "hybrid_rrf": "#009E73"}
CANDIDATE_FAILURE_COLOR = "#4C566A"
RERANKING_FAILURE_COLOR = "#D55E00"
MODEL_COLORS = {"raw_score": "#0072B2", "raw_score_platt": "#0072B2", "calibrated": "#D55E00"}

_FIGURE_WIDTH = 6.5  # inches; at dpi 150 this is ~975px, which is README column width

_STYLE = {
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 150,
}


def _pyplot():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(_STYLE)
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
    metric_labels = ["Recall@10", "Recall@50", "nDCG@10"]
    positions = np.arange(len(metrics))
    width = 0.26
    fig, ax = plt.subplots(figsize=(_FIGURE_WIDTH, 3.8))
    # Grouped by metric rather than by pipeline: the comparison a reader wants is
    # "which retriever wins this metric", which is a within-group read.
    for offset, row in enumerate(rows):
        values = [row[metric] for metric in metrics]
        bars = ax.bar(
            positions + (offset - 1) * width,
            values,
            width,
            label=row["pipeline"],
            color=PIPELINE_COLORS.get(row["pipeline"]),
        )
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7.5, color="0.25")
    ax.set_xticks(positions, metric_labels)
    ax.set_ylim(0.0, 1.08)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("score")
    ax.set_title(
        f"First-stage retrieval, before reranking ({rows[0]['n_queries']} calibration-dev queries)",
        pad=26,
    )
    # Every bar starts at zero, so there is no gap inside the axes wide enough for a legend
    # that does not cover data; it goes above the axes instead.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=3)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    _save_figure(fig, save_path, metadata)


def plot_failure_breakdown(
    rows: list[dict[str, Any]], save_path: str | Path, metadata: dict[str, Any]
) -> None:
    """Where each pipeline's failures come from: a missing candidate, or a mis-ordered one.

    Only the failing share is drawn — the full 0-1 partition puts ~85% of every bar in the
    success band and squashes the composition difference that is the point.
    """
    import numpy as np

    plt = _pyplot()
    pipelines = [row["pipeline"] for row in rows]
    candidate = np.array([row["candidate_set_failure_rate"] for row in rows])
    reranking = np.array([row["reranking_failure_rate"] for row in rows])
    total = candidate + reranking

    fig, ax = plt.subplots(figsize=(_FIGURE_WIDTH, 4.0))
    ax.bar(pipelines, candidate, width=0.55, label="candidate-set failure — gold not in top 50",
           color=CANDIDATE_FAILURE_COLOR)
    ax.bar(pipelines, reranking, bottom=candidate, width=0.55,
           label="reranking failure — gold retrieved, not in final top 10",
           color=RERANKING_FAILURE_COLOR)

    for index, row in enumerate(rows):
        if candidate[index] > 0.012:
            ax.text(index, candidate[index] / 2, f"{candidate[index]:.1%}",
                    ha="center", va="center", color="white", fontsize=8.5)
        if reranking[index] > 0.012:
            ax.text(index, candidate[index] + reranking[index] / 2, f"{reranking[index]:.1%}",
                    ha="center", va="center", color="white", fontsize=8.5)
        # Derived from the two plotted quantities rather than read from the artifact, so the
        # annotation can never disagree with the bar it sits on.
        share = candidate[index] / total[index] if total[index] else 0.0
        ax.text(index, total[index] + 0.006,
                f"{share:.0%} of failures\nfrom the candidate set",
                ha="center", va="bottom", fontsize=8, color="0.3")

    ax.set_ylim(0.0, float(total.max()) * 1.6)
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    ax.set_ylabel("share of all queries")
    success = [row["final_success_rate"] for row in rows]
    ax.set_title(
        f"Failing queries only — the other {min(success):.0%}–{max(success):.0%} succeed "
        "and are not drawn",
        pad=10,
    )
    ax.legend(loc="upper right")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    _save_figure(fig, save_path, metadata)


def plot_reranking_transitions(
    rows: list[dict[str, Any]], save_path: str | Path, metadata: dict[str, Any]
) -> None:
    """Counts for the four transition labels that describe an outcome change.

    `already_successful` is excluded from the bars and reported in the subtitle instead: at
    ~135 of 162 queries it is an order of magnitude larger than every other category and
    flattens them into a single readable row of stubs.
    """
    import numpy as np

    plt = _pyplot()
    categories = [
        "rescued_by_reranker",
        "degraded_by_reranker",
        "unchanged_failure",
        "no_opportunity",
    ]
    labels = [
        "rescued\ninto top 10",
        "degraded\nout of top 10",
        "unchanged failure\nnever ranked",
        "no opportunity\nnot in top 50",
    ]
    positions = np.arange(len(categories))
    width = 0.26
    fig, ax = plt.subplots(figsize=(_FIGURE_WIDTH, 4.0))
    for offset, row in enumerate(rows):
        bars = ax.bar(
            positions + (offset - 1) * width,
            [row["counts"][category] for category in categories],
            width,
            label=row["pipeline"],
            color=PIPELINE_COLORS.get(row["pipeline"]),
        )
        ax.bar_label(bars, padding=2, fontsize=8, color="0.25")

    already = " / ".join(str(row["counts"]["already_successful"]) for row in rows)
    total = sum(rows[0]["counts"].values())
    ax.set_xticks(positions, labels, fontsize=8.5)
    ax.set_ylabel("query count")
    ax.set_ylim(0, max(row["counts"][c] for row in rows for c in categories) * 1.3)
    ax.set_title(
        f"What reranking changed, of {total} queries\n"
        f"already successful before and after, not shown: {already}",
        pad=10,
    )
    ax.legend(loc="upper left", ncol=3)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
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
    """Reliability diagram, one panel per confidence model.

    Points are deliberately not connected: a line through a 2-query bin turns sampling noise
    into an apparent miscalibration trend. Marker area is proportional to bin count, so a small
    marker with a tall interval reads as "too little data here" rather than as a finding.
    """
    plt = _pyplot()
    panels = (
        ("raw_score_platt", "raw score, Platt-scaled (baseline)", "o"),
        ("calibrated", "common-feature calibrator", "s"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(_FIGURE_WIDTH, 3.4), sharey=True)

    for ax, (model, label, marker) in zip(axes, panels):
        color = MODEL_COLORS[model]
        points = reliability_bins(predictions, model, n_bins=10)
        xs = [point["mean_confidence"] for point in points]
        ys = [point["observed_success"] for point in points]
        counts = [point["count"] for point in points]
        intervals = [
            _wilson_interval(round(point["observed_success"] * point["count"]), point["count"])
            for point in points
        ]

        ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=1.0, zorder=1)
        ax.errorbar(
            xs,
            ys,
            yerr=[[y - lo for y, (lo, _hi) in zip(ys, intervals)],
                  [hi - y for y, (_lo, hi) in zip(ys, intervals)]],
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=2.5,
            alpha=0.45,
            zorder=2,
        )
        # Area stays proportional to bin count, but the constant is small enough that the
        # largest bin (n=76) does not overflow the axes and collide with its neighbour.
        ax.scatter(xs, ys, s=[12 + 1.3 * count for count in counts], facecolor=color,
                   edgecolor="white", linewidth=0.6, marker=marker, zorder=3)
        ax.set_xlim(-0.02, 1.04)
        ax.set_ylim(-0.03, 1.06)
        ax.set_xlabel("mean predicted probability")
        ax.set_title(label, fontsize=9.5)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("observed top-10 success rate")
    fig.suptitle("Hybrid reliability: predicted vs. observed top-10 success", y=0.98)
    fig.text(
        0.5,
        0.015,
        "10 fixed bins · marker area ∝ queries in bin · bars = 95% Wilson interval · "
        "dashed = perfect calibration",
        ha="center",
        fontsize=7.5,
        color="0.4",
    )
    fig.tight_layout(rect=(0.0, 0.05, 1.0, 0.94))
    _save_figure(fig, save_path, metadata, already_tight=True)


def plot_hybrid_risk_coverage(
    curves: dict[str, list[dict[str, float]]],
    save_path: str | Path,
    metadata: dict[str, Any],
) -> None:
    """Hybrid raw-score ordering versus the primary common-feature confidence model.

    A lower curve means the confidence score is better at pushing the queries that will fail to
    the end of the ranking, so abstaining on them costs less.
    """
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(_FIGURE_WIDTH, 3.8))

    # Below ~30% coverage each point is an average over fewer than 50 queries and a single
    # failure moves the curve by two percentage points; shaded so the staircase there is not
    # read as a difference between the models.
    low_sample = 0.3
    ax.axvspan(0.0, low_sample, color="0.93", zorder=0)

    for model, label in (
        ("raw_score", "raw reranker score"),
        ("calibrated", "common-feature calibrator"),
    ):
        curve = curves[model]
        coverages = [point["coverage"] for point in curve]
        risks = [point["risk"] for point in curve]
        ax.plot(coverages, risks, label=label, color=MODEL_COLORS[model], linewidth=1.4, zorder=2)
        # The two reported operating points, so the figure and the selective-coverage table
        # can be read against each other.
        for level in (0.8, 0.6):
            index = min(range(len(coverages)), key=lambda i: abs(coverages[i] - level))
            ax.plot(coverages[index], risks[index], marker="o", markersize=4.5,
                    color=MODEL_COLORS[model], zorder=3)

    top = ax.get_ylim()[1]
    for level in (0.6, 0.8):
        ax.axvline(level, color="0.75", linewidth=0.8, linestyle=":", zorder=1)
        ax.text(level, top * 0.99, f"{level:.0%} coverage" if level == 0.8 else f"{level:.0%}",
                ha="center", va="top", fontsize=7.5, color="0.45")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{value:.0%}")
    ax.set_xlabel("coverage — share of queries answered rather than abstained on")
    ax.set_ylabel("risk — failures among\nqueries answered")
    ax.set_title(
        "Hybrid risk-coverage: what abstention buys (calibration-dev)\n"
        "shaded: below 30% coverage too few queries remain for the rate to mean much",
        fontsize=10,
        pad=12,
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    _save_figure(fig, save_path, metadata)
