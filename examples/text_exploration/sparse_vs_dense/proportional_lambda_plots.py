"""Render the retention bar chart + per-bit bonus distribution histogram
for the proportional-lambda-consolidation sweep.

Bar chart: mean retention on the shared SimLex partition by variant,
with seed-level error bars.

Histogram: per-bit BCM bonus magnitude distribution, comparing absolute
vs proportional. Built from one seed's phase-1 accumulator (the cache
written by the sweep).

Usage:
    uv run --extra embeddings --extra viz python -m \
        examples.text_exploration.sparse_vs_dense.proportional_lambda_plots \
        --sweep-root runs/arb139/proportional_lambda_2026-05-02
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VARIANT_LABEL = {
    "w2v_baseline": "w2v",
    "ssh_baseline": "SSH (no bonus)",
    "ssh_absolute": "SSH+BCM-absolute",
    "ssh_proportional": "SSH+BCM-proportional",
}
VARIANT_ORDER = ["w2v_baseline", "ssh_baseline", "ssh_absolute", "ssh_proportional"]
VARIANT_COLOR = {
    "w2v_baseline": "#1f77b4",
    "ssh_baseline": "#7f7f7f",
    "ssh_absolute": "#ff7f0e",
    "ssh_proportional": "#2ca02c",
}


def plot_retention_bar(
    summary_csv: Path, out_path: Path, bonus: float, n_per_phase: int
) -> None:
    df = pd.read_csv(summary_csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    means, stds, ns, xs = [], [], [], []
    for i, variant in enumerate(VARIANT_ORDER):
        sub = df[df["variant"] == variant]["retention_shared"].dropna()
        if len(sub) == 0:
            continue
        means.append(sub.mean())
        stds.append(sub.std() if len(sub) > 1 else 0.0)
        ns.append(len(sub))
        xs.append(i)
        ax.bar(
            i,
            sub.mean(),
            yerr=(sub.std() if len(sub) > 1 else 0.0),
            capsize=4,
            color=VARIANT_COLOR[variant],
            label=f"{VARIANT_LABEL[variant]} (n={len(sub)})",
        )
        # Per-seed dots over the bar.
        ax.scatter(
            [i] * len(sub),
            sub.values,
            color="black",
            s=14,
            zorder=3,
            alpha=0.6,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([VARIANT_LABEL[VARIANT_ORDER[i]] for i in xs], rotation=15)
    ax.axhline(0.85, color="green", linestyle="--", alpha=0.4,
               label="spec target (0.85)")
    ax.axhline(0.70, color="red", linestyle="--", alpha=0.4,
               label="falsification (0.70)")
    ax.set_ylabel("retention on shared (sim_phase2 / sim_phase1)")
    ax.set_title(
        f"Proportional-λ vs absolute-λ BCM consolidation\n"
        f"cross-domain (text8→Gutenberg), {n_per_phase:,} tokens/phase, "
        f"matched 8-epoch phase 1, λ={bonus}"
    )
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Wrote {out_path}")


def plot_bonus_histogram(
    cache_dir: Path, out_path: Path, k_active: int, bonus: float, seed: int = 0
) -> None:
    """Per-bit bonus distribution: absolute (flat) vs proportional (scaled by A_phase1)."""
    cache_path = cache_dir / f"seed_{seed}.pkl"
    if not cache_path.exists():
        print(f"  [warn] no phase-1 cache at {cache_path}; skipping histogram")
        return
    with cache_path.open("rb") as f:
        cache = pickle.load(f)
    A_phase1 = cache["A_center"]
    V, D = A_phase1.shape

    # Top-k mask.
    mask = np.zeros((V, D), dtype=bool)
    for w in range(V):
        idx = np.argpartition(-A_phase1[w], k_active)[:k_active]
        mask[w, idx] = True

    bonus_absolute = (np.float32(bonus) * mask.astype(np.float32))[mask]
    bonus_proportional = (
        np.float32(bonus) * mask.astype(np.float32) * A_phase1.astype(np.float32)
    )[mask]

    # Diagnostics.
    print(
        f"  absolute bonus on top-k bits: "
        f"mean={bonus_absolute.mean():.4f} (flat by construction)"
    )
    print(
        f"  proportional bonus on top-k bits: "
        f"mean={bonus_proportional.mean():.4f} median={np.median(bonus_proportional):.4f} "
        f"min={bonus_proportional.min():.4f} max={bonus_proportional.max():.4f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    axes[0].hist(
        bonus_absolute,
        bins=40,
        color=VARIANT_COLOR["ssh_absolute"],
        alpha=0.85,
    )
    axes[0].set_xlabel("per-bit bonus magnitude")
    axes[0].set_ylabel("# top-k bits across vocab")
    axes[0].set_title(f"absolute λ={bonus} (flat)")
    axes[0].grid(alpha=0.3)

    axes[1].hist(
        bonus_proportional,
        bins=40,
        color=VARIANT_COLOR["ssh_proportional"],
        alpha=0.85,
    )
    axes[1].axvline(
        bonus_proportional.mean(),
        color="black",
        linestyle="--",
        alpha=0.6,
        label=f"mean={bonus_proportional.mean():.3f}",
    )
    axes[1].set_xlabel("per-bit bonus magnitude")
    axes[1].set_ylabel("# top-k bits across vocab")
    axes[1].set_title(f"proportional λ·A_phase1 (right-skewed)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"BCM bonus distribution on top-{k_active} phase-1 bits (seed={seed})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-root", type=str, required=True)
    p.add_argument("--n-per-phase", type=int, default=1_000_000)
    p.add_argument("--bonus", type=float, default=0.5)
    p.add_argument("--k-active", type=int, default=40)
    p.add_argument("--seed-for-histogram", type=int, default=0)
    args = p.parse_args()

    root = Path(args.sweep_root)
    fig_dir = root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_retention_bar(
        root / "summary_retention.csv",
        fig_dir / "retention_bar.png",
        bonus=args.bonus,
        n_per_phase=args.n_per_phase,
    )

    plot_bonus_histogram(
        root / "ssh_phase1_caches",
        fig_dir / "bonus_distribution.png",
        k_active=args.k_active,
        bonus=args.bonus,
        seed=args.seed_for_histogram,
    )


if __name__ == "__main__":
    main()
