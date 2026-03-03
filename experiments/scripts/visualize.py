#!/usr/bin/env python3
"""Read JSON experiment results and produce figures."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(run_dir: Path) -> list[dict]:
    """Load all seed JSON files from a run directory."""
    results = []
    for path in sorted(run_dir.glob("*.json")):
        with open(path) as f:
            results.append(json.load(f))
    return results


def plot_learning_curves(
    results: list[dict],
    title: str,
    output_path: Path,
) -> None:
    """Plot rolling IoU over tokens with mean +/- std across seeds."""
    all_vals: list[list[float]] = []
    all_ts: list[list[int]] = []

    for r in results:
        ts = [t for t, _ in r["rolling_ious"]]
        vals = [v for _, v in r["rolling_ious"]]
        all_ts.append(ts)
        all_vals.append(vals)

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

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_iou_histogram(
    results: list[dict],
    output_path: Path,
) -> None:
    """Plot distribution of per-step IoU values across all seeds."""
    all_ious = []
    for r in results:
        all_ious.extend(r["ious"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_ious, bins=50, color="#2563eb", alpha=0.7, edgecolor="white")
    ax.set_xlabel("IoU", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Per-Step IoU", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main(run_dir_path: str) -> None:
    run_dir = Path(run_dir_path)
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory")
        sys.exit(1)

    results = load_results(run_dir)
    if not results:
        print(f"No JSON result files found in {run_dir}")
        sys.exit(1)

    name = run_dir.name
    figures_dir = run_dir.parent.parent / "figures"

    print(f"Loaded {len(results)} seed results from {run_dir}")

    plot_learning_curves(
        results,
        title=f"STEP Learning Curve ({name})",
        output_path=figures_dir / f"{name}_learning_curve.png",
    )

    plot_iou_histogram(
        results,
        output_path=figures_dir / f"{name}_iou_dist.png",
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <run_dir>")
        print("Example: python visualize.py experiments/runs/exp0_tinystories/")
        sys.exit(1)
    main(sys.argv[1])
