#!/usr/bin/env python3
"""Visualize STEP diagnostic data from exp2.

Reads JSON files from {run_dir}/diagnostics/ and produces 5 plots.
Usage: uv run --extra viz experiments/scripts/visualize_diagnostics.py <run_dir>
"""

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_iou_distribution(run_dir: Path, figures_dir: Path) -> None:
    """Plot IoU/native metric distribution from step_memory run.json."""
    run_json = run_dir / "step_memory" / "run.json"
    if not run_json.exists():
        print(f"  Skipping IoU distribution: {run_json} not found")
        return

    data = json.loads(run_json.read_text())
    metrics = data.get("native_metrics", [])
    if not metrics:
        print("  Skipping IoU distribution: no native_metrics")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(metrics, bins=50, color="#2563eb", alpha=0.7, edgecolor="white")
    ax.set_xlabel(data.get("native_metric_name", "IoU"), fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Distribution of Native Metric (IoU)", fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = figures_dir / "iou_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_weight_stats(diag_dir: Path, figures_dir: Path) -> None:
    """Plot weight statistics over training steps."""
    path = diag_dir / "weight_stats.json"
    if not path.exists():
        print(f"  Skipping weight stats: {path} not found")
        return

    stats = json.loads(path.read_text())
    if not stats:
        print("  Skipping weight stats: empty")
        return

    steps = [s["step"] for s in stats]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Mean & std
    ax = axes[0, 0]
    ax.plot(steps, [s["mean"] for s in stats], label="mean", color="#2563eb")
    ax.plot(steps, [s["std"] for s in stats], label="std", color="#dc2626")
    ax.set_title("Weight Mean & Std")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Max & min
    ax = axes[0, 1]
    ax.plot(steps, [s["max"] for s in stats], label="max", color="#16a34a")
    ax.plot(steps, [s["min"] for s in stats], label="min", color="#f59e0b")
    ax.set_title("Weight Max & Min")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Near-zero & nonzero fraction
    ax = axes[1, 0]
    ax.plot(
        steps, [s["near_zero_frac"] for s in stats],
        label="|w| < 0.01", color="#8b5cf6",
    )
    ax.plot(
        steps, [s["nonzero_frac"] for s in stats],
        label="nonzero", color="#06b6d4",
    )
    ax.set_title("Weight Sparsity")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Fraction > 1.0
    ax = axes[1, 1]
    ax.plot(
        steps, [s["gt_one_frac"] for s in stats],
        label="|w| > 1.0", color="#ef4444",
    )
    ax.set_title("Fraction of Weights > 1.0")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel("Training Step")

    fig.suptitle("Weight Statistics Over Training", fontsize=14, y=1.02)
    fig.tight_layout()

    out = figures_dir / "weight_stats.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_bigram_overlap(
    diag_dir: Path, figures_dir: Path, n: int = 2048, k: int = 40,
) -> None:
    """Plot bigram frequency vs SDR overlap, with random expectation line."""
    path = diag_dir / "bigram_overlap.json"
    if not path.exists():
        print(f"  Skipping bigram overlap: {path} not found")
        return

    bigrams = json.loads(path.read_text())
    if not bigrams:
        print("  Skipping bigram overlap: empty")
        return

    counts = [b["count"] for b in bigrams]
    overlaps = [b["overlap"] for b in bigrams]

    # Random expectation: E[overlap] = k^2 / n
    random_expected = k * k / n

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(counts, overlaps, alpha=0.6, color="#2563eb", s=40)
    ax.axhline(
        y=random_expected, color="#dc2626", linestyle="--", linewidth=1.5,
        label=f"Random expectation (k²/n = {random_expected:.2f})",
    )
    ax.set_xlabel("Bigram Frequency", fontsize=12)
    ax.set_ylabel("SDR Bit Overlap", fontsize=12)
    ax.set_title("Bigram Frequency vs SDR Overlap", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = figures_dir / "bigram_overlap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_position_accuracy(diag_dir: Path, figures_dir: Path) -> None:
    """Plot mean accuracy by position within story."""
    path = diag_dir / "position_accuracy.json"
    if not path.exists():
        print(f"  Skipping position accuracy: {path} not found")
        return

    data = json.loads(path.read_text())
    if not data:
        print("  Skipping position accuracy: empty")
        return

    positions = sorted(int(k) for k in data)
    accs = [data[str(p)] for p in positions]

    # Bin into groups of 10 for smoother plot
    bin_size = 10
    binned_pos = []
    binned_acc = []
    for i in range(0, len(positions), bin_size):
        chunk_pos = positions[i : i + bin_size]
        chunk_acc = accs[i : i + bin_size]
        binned_pos.append(np.mean(chunk_pos))
        binned_acc.append(np.mean(chunk_acc))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(binned_pos, binned_acc, color="#2563eb", linewidth=1.5)
    ax.set_xlabel("Position Within Story (tokens)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy by Position Within Story", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = figures_dir / "position_accuracy.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_top_confusions(diag_dir: Path, figures_dir: Path, top_n: int = 20) -> None:
    """Plot top-N most confused token pairs."""
    path = diag_dir / "predictions.json"
    if not path.exists():
        print(f"  Skipping top confusions: {path} not found")
        return

    predictions = json.loads(path.read_text())
    if not predictions:
        print("  Skipping top confusions: empty")
        return

    # Count confusion pairs (predicted != actual)
    confusion_counts: Counter[tuple[int, int]] = Counter()
    for _t, predicted, actual, _metric in predictions:
        if predicted != actual and predicted >= 0:
            confusion_counts[(predicted, actual)] += 1

    if not confusion_counts:
        print("  Skipping top confusions: no confused pairs")
        return

    # Try to decode tokens
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        def decode(tid: int) -> str:
            text = tokenizer.decode([tid]).strip()
            if not text:
                return f"[{tid}]"
            return repr(text) if len(text) < 15 else repr(text[:12] + "...")
    except Exception:
        def decode(tid: int) -> str:
            return str(tid)

    top = confusion_counts.most_common(top_n)
    labels = [f"{decode(p)} -> {decode(a)}" for (p, a), _ in top]
    values = [c for _, c in top]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color="#2563eb", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Count", fontsize=12)
    title = f"Top-{top_n} Most Confused Token Pairs (pred -> actual)"
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()

    out = figures_dir / "top_confusions.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def main(run_dir_path: str) -> None:
    run_dir = Path(run_dir_path)
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory")
        sys.exit(1)

    diag_dir = run_dir / "diagnostics"
    figures_dir = run_dir.parent.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    name = run_dir.name
    print(f"Generating diagnostic plots for: {name}")
    print(f"  Diagnostics dir: {diag_dir}")
    print(f"  Figures dir: {figures_dir}")
    print()

    plot_iou_distribution(run_dir, figures_dir)
    plot_weight_stats(diag_dir, figures_dir)
    plot_bigram_overlap(diag_dir, figures_dir)
    plot_position_accuracy(diag_dir, figures_dir)
    plot_top_confusions(diag_dir, figures_dir)

    print()
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_diagnostics.py <run_dir>")
        print("Example: python visualize_diagnostics.py "
              "experiments/runs/exp2_diagnostics/")
        sys.exit(1)
    main(sys.argv[1])
