"""Generate experiment figures with error bars from multi-seed runs."""

from pathlib import Path

import numpy as np


def plot_learning_curves(
    results: list,  # list[RunResult], avoiding circular import
    title: str = "STEP Learning Curve",
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    """Plot rolling IoU over tokens with mean +/- std across seeds."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Align results to common token indices
    # All runs should have the same log_interval, so rolling_ious align
    all_ts: list[list[int]] = []
    all_vals: list[list[float]] = []

    for result in results:
        ts = [t for t, _ in result.rolling_ious]
        vals = [v for _, v in result.rolling_ious]
        all_ts.append(ts)
        all_vals.append(vals)

    # Use the shortest run's length to align
    min_len = min(len(v) for v in all_vals)
    ts_arr = np.array(all_ts[0][:min_len])
    vals_arr = np.array([v[:min_len] for v in all_vals])

    mean = vals_arr.mean(axis=0)
    std = vals_arr.std(axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ts_arr, mean, color="#2563eb", linewidth=2, label="Mean IoU")
    ax.fill_between(
        ts_arr,
        mean - std,
        mean + std,
        alpha=0.2,
        color="#2563eb",
        label=f"\u00b11 std (n={len(results)} seeds)",
    )

    ax.set_xlabel("Tokens", fontsize=12)
    ax.set_ylabel("Rolling IoU", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)


def plot_iou_histogram(
    results: list,
    output_path: Path | None = None,
    show: bool = False,
) -> None:
    """Plot distribution of per-step IoU values across all seeds."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_ious = []
    for result in results:
        all_ious.extend(result.ious)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_ious, bins=50, color="#2563eb", alpha=0.7, edgecolor="white")
    ax.set_xlabel("IoU", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Per-Step IoU", fontsize=14)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    if show:
        plt.show()

    plt.close(fig)
