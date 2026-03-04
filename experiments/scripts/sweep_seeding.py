#!/usr/bin/env python3
"""Compare seeding modes: hash-based, active-bits, predicted-SDR.

Usage: uv run --extra comparison experiments/scripts/sweep_seeding.py
"""

import time

from step.config import EncoderConfig, ModelConfig, TrainingConfig
from step.data import prepare_token_cache
from step.experiment import (
    ExperimentConfig,
    pretrain_step_model,
    run_experiment,
)
from step.wrappers import StepMemoryModel

PRETRAIN_TOKENS = 50_000
EVAL_TOKENS = 5_000


def run_one(adaptive, seeding, fraction, window, train_cache, eval_cache):
    enc_cfg = EncoderConfig(
        model_name="gpt2", n=2048, k=40, vocab_size=10000,
        adaptive=adaptive, context_fraction=fraction, seeding=seeding,
    )
    model_cfg = ModelConfig(
        n=2048, k=40, max_lr=0.5, weight_decay=0.999,
        penalty_factor=0.5, eligibility_window=window,
    )
    train_tc = TrainingConfig(
        dataset_name="roneneldan/TinyStories", dataset_split="train",
        max_tokens=PRETRAIN_TOKENS, log_interval=50_000,
    )
    eval_tc = TrainingConfig(
        dataset_name="roneneldan/TinyStories", dataset_split="validation",
        max_tokens=EVAL_TOKENS, log_interval=1_000,
    )
    pretrain_cfg = ExperimentConfig(
        encoder=enc_cfg, model=model_cfg, training=train_tc, name="sweep",
    )
    eval_cfg = ExperimentConfig(
        encoder=enc_cfg, model=model_cfg, training=eval_tc, name="sweep",
    )

    model = StepMemoryModel(model_cfg, enc_cfg)
    pretrain_step_model(model, pretrain_cfg, train_cache)

    def factory(config, _m=model):
        return _m

    result = run_experiment(eval_cfg, factory, "sweep", "iou", eval_cache)
    acc = result.rolling_accuracies[-1][1] if result.rolling_accuracies else 0.0
    iou = result.rolling_native[-1][1] if result.rolling_native else 0.0
    return acc, iou


def main():
    enc_cfg = EncoderConfig(model_name="gpt2", n=2048, k=40, vocab_size=10000)
    train_tc = TrainingConfig(
        dataset_name="roneneldan/TinyStories", dataset_split="train",
        max_tokens=PRETRAIN_TOKENS,
    )
    eval_tc = TrainingConfig(
        dataset_name="roneneldan/TinyStories", dataset_split="validation",
        max_tokens=EVAL_TOKENS,
    )

    print("Caching data...")
    train_cache = prepare_token_cache(train_tc, enc_cfg)
    eval_cache = prepare_token_cache(eval_tc, enc_cfg)
    print(f"  Train: {len(train_cache):,}, Eval: {len(eval_cache):,}\n")

    # Configs: (label, adaptive, seeding, fraction, window)
    configs = [
        # Baselines: hash-based
        ("hash         ", False, "active", 0.0, 3),
        ("hash         ", False, "active", 0.0, 5),
        ("hash         ", False, "active", 0.0, 10),
        # Active seeding (frac=0.3 — best from fraction sweep)
        ("active f=0.3 ", True, "active", 0.3, 3),
        ("active f=0.3 ", True, "active", 0.3, 5),
        ("active f=0.3 ", True, "active", 0.3, 10),
        # Predicted seeding (frac=0.3)
        ("predict f=0.3", True, "predicted", 0.3, 3),
        ("predict f=0.3", True, "predicted", 0.3, 5),
        ("predict f=0.3", True, "predicted", 0.3, 10),
        # Predicted seeding (frac=0.5)
        ("predict f=0.5", True, "predicted", 0.5, 3),
        ("predict f=0.5", True, "predicted", 0.5, 5),
        ("predict f=0.5", True, "predicted", 0.5, 10),
    ]

    results = []
    for label, adaptive, seeding, fraction, window in configs:
        start = time.monotonic()
        acc, iou = run_one(adaptive, seeding, fraction, window, train_cache, eval_cache)
        elapsed = time.monotonic() - start
        results.append((label, window, acc, iou, elapsed))
        print(f"  {label} w={window:2d}: acc={acc:.1%} iou={iou:.4f} ({elapsed:.0f}s)")

    print(f"\n{'Seeding':15s} {'Window':>6s} {'Accuracy':>10s} {'IoU':>8s}")
    for label, w, acc, iou, _ in results:
        print(f"{label} {w:6d} {acc:10.1%} {iou:8.4f}")


if __name__ == "__main__":
    main()
