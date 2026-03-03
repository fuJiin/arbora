#!/usr/bin/env python3
"""Run parameter sweeps and save raw JSON results."""

from pathlib import Path

import numpy as np

from step.config import EncoderConfig, ModelConfig, TrainingConfig
from step.experiment import ExperimentConfig, run_multi_seed

ROOT = Path(__file__).resolve().parent.parent


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

    for label, n, k, window in configs:
        base = ExperimentConfig(
            encoder=EncoderConfig(n=n, k=k),
            model=ModelConfig(
                n=n,
                k=k,
                eligibility_window=window,
                max_lr=0.5,
                weight_decay=0.999,
                penalty_factor=0.5,
            ),
            training=TrainingConfig(
                max_tokens=max_tokens,
                log_interval=log_interval,
                rolling_window=50,
            ),
            name=f"sweep_n{n}_k{k}",
        )

        print(f"Running {label}...")
        results = run_multi_seed(base, seeds, ROOT / "runs" / f"sweep_n{n}_k{k}")

        all_vals = []
        for r in results:
            vals = [v for _, v in r.rolling_ious]
            all_vals.append(vals)

        min_len = min(len(v) for v in all_vals)
        vals_arr = np.array([v[:min_len] for v in all_vals])
        mean_final = vals_arr[:, -1].mean()
        print(f"  Final mean IoU: {mean_final:.4f}")


def sweep_learning_rates() -> None:
    """Compare different learning rates."""
    lrs = [0.1, 0.3, 0.5, 0.8]
    seeds = [0, 1, 2]
    n, k, window = 512, 20, 50
    max_tokens = 3000
    log_interval = 25

    for lr in lrs:
        base = ExperimentConfig(
            encoder=EncoderConfig(n=n, k=k),
            model=ModelConfig(
                n=n,
                k=k,
                eligibility_window=window,
                max_lr=lr,
                weight_decay=0.999,
                penalty_factor=0.5,
            ),
            training=TrainingConfig(
                max_tokens=max_tokens,
                log_interval=log_interval,
                rolling_window=50,
            ),
            name=f"sweep_lr{lr}",
        )

        print(f"Running lr={lr}...")
        results = run_multi_seed(base, seeds, ROOT / "runs" / f"sweep_lr{lr}")

        all_vals = []
        for r in results:
            vals = [v for _, v in r.rolling_ious]
            all_vals.append(vals)

        min_len = min(len(v) for v in all_vals)
        vals_arr = np.array([v[:min_len] for v in all_vals])
        mean_final = vals_arr[:, -1].mean()
        print(f"  Final mean IoU: {mean_final:.4f}")


if __name__ == "__main__":
    sweep_sdr_sizes()
    sweep_learning_rates()
