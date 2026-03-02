#!/usr/bin/env python3
"""Run a parameter sweep and produce comparison figures."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from step.config import EncoderConfig, ModelConfig, TrainingConfig
from step.experiment import ExperimentConfig, run_multi_seed

ROOT = Path(__file__).parent


def sweep_sdr_sizes() -> None:
    """Compare different SDR sizes (n, k) on the same task."""
    configs = [
        ("n=256, k=10", 256, 10, 25),
        ("n=512, k=20", 512, 20, 50),
        ("n=1024, k=30", 1024, 30, 75),
    ]
    seeds = [0, 1, 2]
    max_tokens = 3000
    log_interval = 25

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2563eb", "#dc2626", "#16a34a"]

    for (label, n, k, window), color in zip(configs, colors):
        base = ExperimentConfig(
            encoder=EncoderConfig(n=n, k=k),
            model=ModelConfig(
                n=n, k=k, eligibility_window=window,
                max_lr=0.5, weight_decay=0.999, penalty_factor=0.5,
            ),
            training=TrainingConfig(
                max_tokens=max_tokens, log_interval=log_interval,
                rolling_window=50,
            ),
            name=f"sweep_n{n}_k{k}",
        )

        print(f"Running {label}...")
        results = run_multi_seed(
            base, seeds, ROOT / "runs" / f"sweep_n{n}_k{k}"
        )

        # Extract aligned rolling IoU curves
        all_vals = []
        for r in results:
            vals = [v for _, v in r.rolling_ious]
            all_vals.append(vals)

        min_len = min(len(v) for v in all_vals)
        ts = np.array([t for t, _ in results[0].rolling_ious[:min_len]])
        vals_arr = np.array([v[:min_len] for v in all_vals])
        mean = vals_arr.mean(axis=0)
        std = vals_arr.std(axis=0)

        ax.plot(ts, mean, color=color, linewidth=2, label=label)
        ax.fill_between(ts, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Tokens", fontsize=12)
    ax.set_ylabel("Rolling IoU", fontsize=12)
    ax.set_title("STEP: Effect of SDR Size on Learning", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = ROOT / "figures" / "sweep_sdr_sizes.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")


def sweep_learning_rates() -> None:
    """Compare different learning rates."""
    lrs = [0.1, 0.3, 0.5, 0.8]
    seeds = [0, 1, 2]
    n, k, window = 512, 20, 50
    max_tokens = 3000
    log_interval = 25

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#6366f1", "#2563eb", "#dc2626", "#f59e0b"]

    for lr, color in zip(lrs, colors):
        base = ExperimentConfig(
            encoder=EncoderConfig(n=n, k=k),
            model=ModelConfig(
                n=n, k=k, eligibility_window=window,
                max_lr=lr, weight_decay=0.999, penalty_factor=0.5,
            ),
            training=TrainingConfig(
                max_tokens=max_tokens, log_interval=log_interval,
                rolling_window=50,
            ),
            name=f"sweep_lr{lr}",
        )

        print(f"Running lr={lr}...")
        results = run_multi_seed(
            base, seeds, ROOT / "runs" / f"sweep_lr{lr}"
        )

        all_vals = []
        for r in results:
            vals = [v for _, v in r.rolling_ious]
            all_vals.append(vals)

        min_len = min(len(v) for v in all_vals)
        ts = np.array([t for t, _ in results[0].rolling_ious[:min_len]])
        vals_arr = np.array([v[:min_len] for v in all_vals])
        mean = vals_arr.mean(axis=0)
        std = vals_arr.std(axis=0)

        ax.plot(ts, mean, color=color, linewidth=2, label=f"lr={lr}")
        ax.fill_between(ts, mean - std, mean + std, alpha=0.15, color=color)

    ax.set_xlabel("Tokens", fontsize=12)
    ax.set_ylabel("Rolling IoU", fontsize=12)
    ax.set_title("STEP: Effect of Learning Rate", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    out = ROOT / "figures" / "sweep_learning_rates.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    sweep_sdr_sizes()
    sweep_learning_rates()
