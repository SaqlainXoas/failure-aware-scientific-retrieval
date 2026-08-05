"""Risk-coverage curve plotting."""

from pathlib import Path

LABELS = {
    "raw_score": "raw reranker score (baseline)",
    "raw_score_platt": "raw score, Platt-scaled",
    "calibrated": "calibrated (common features)",
    "calibrated_hybrid_exploratory": "calibrated + BM25/dense overlap (exploratory)",
}


def plot_risk_coverage(curves: dict[str, list[dict[str, float]]], save_path: str | Path) -> None:
    """One risk-coverage line per confidence model, plotted together so the raw baseline, its
    Platt-scaled form, and the calibrator are compared but never conflated (plan §9)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
