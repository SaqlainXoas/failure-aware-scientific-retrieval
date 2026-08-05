"""Risk-coverage curve plotting."""

from pathlib import Path


def plot_risk_coverage(
    curve_baseline: list[dict[str, float]], curve_calibrated: list[dict[str, float]], save_path: str | Path
) -> None:
    """Risk-coverage lines for the raw reranker-score baseline vs. the calibrated
    probability, plotted together so the two are compared but never conflated (plan §9)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5))
    if curve_baseline:
        ax.plot(
            [p["coverage"] for p in curve_baseline],
            [p["risk"] for p in curve_baseline],
            label="raw reranker score",
        )
    if curve_calibrated:
        ax.plot(
            [p["coverage"] for p in curve_calibrated],
            [p["risk"] for p in curve_calibrated],
            label="calibrated probability",
        )
    ax.set_xlabel("coverage")
    ax.set_ylabel("risk (1 - selective success rate)")
    ax.set_title("Risk-coverage: raw score vs. calibrated confidence")
    ax.legend()
    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
