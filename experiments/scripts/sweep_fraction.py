#!/usr/bin/env python3
"""Sweep context_fraction values for adaptive encoding.

Usage: uv run --extra comparison experiments/scripts/sweep_fraction.py
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


def run_one(
    fraction: float,
    window: int,
    train_cache,
    eval_cache,
) -> tuple[float, float]:
    enc_cfg = EncoderConfig(
        model_name="gpt2",
        n=2048,
        k=40,
        vocab_size=10000,
        adaptive=True,
        context_fraction=fraction,
    )
    model_cfg = ModelConfig(
        n=2048,
        k=40,
        max_lr=0.5,
        weight_decay=0.999,
        penalty_factor=0.5,
        eligibility_window=window,
    )
    train_tc = TrainingConfig(
        dataset_name="roneneldan/TinyStories",
        dataset_split="train",
        max_tokens=PRETRAIN_TOKENS,
        log_interval=50_000,
    )
    eval_tc = TrainingConfig(
        dataset_name="roneneldan/TinyStories",
        dataset_split="validation",
        max_tokens=EVAL_TOKENS,
        log_interval=1_000,
    )

    pretrain_cfg = ExperimentConfig(
        encoder=enc_cfg,
        model=model_cfg,
        training=train_tc,
        name="sweep",
    )
    eval_cfg = ExperimentConfig(
        encoder=enc_cfg,
        model=model_cfg,
        training=eval_tc,
        name="sweep",
    )

    model = StepMemoryModel(model_cfg, enc_cfg)
    pretrain_step_model(model, pretrain_cfg, train_cache)

    def factory(config, _m=model):
        return _m

    result = run_experiment(eval_cfg, factory, "sweep", "iou", eval_cache)
    final_acc = result.rolling_accuracies[-1][1] if result.rolling_accuracies else 0.0
    final_iou = result.rolling_native[-1][1] if result.rolling_native else 0.0
    return final_acc, final_iou


def main():
    enc_cfg = EncoderConfig(model_name="gpt2", n=2048, k=40, vocab_size=10000)
    train_tc = TrainingConfig(
        dataset_name="roneneldan/TinyStories",
        dataset_split="train",
        max_tokens=PRETRAIN_TOKENS,
    )
    eval_tc = TrainingConfig(
        dataset_name="roneneldan/TinyStories",
        dataset_split="validation",
        max_tokens=EVAL_TOKENS,
    )

    print("Caching data...")
    train_cache = prepare_token_cache(train_tc, enc_cfg)
    eval_cache = prepare_token_cache(eval_tc, enc_cfg)
    print(f"  Train: {len(train_cache):,}, Eval: {len(eval_cache):,}\n")

    fractions = [0.0, 0.1, 0.2, 0.3, 0.5]
    windows = [3, 5]

    results = []
    for frac in fractions:
        for w in windows:
            start = time.monotonic()
            acc, iou = run_one(frac, w, train_cache, eval_cache)
            elapsed = time.monotonic() - start
            results.append((frac, w, acc, iou, elapsed))
            print(
                f"  frac={frac:.1f} w={w:2d}: "
                f"acc={acc:.1%} iou={iou:.4f} ({elapsed:.0f}s)"
            )

    print(f"\n{'Fraction':>8s} {'Window':>6s} {'Accuracy':>10s} {'IoU':>8s}")
    for frac, w, acc, iou, _ in results:
        print(f"{frac:8.1f} {w:6d} {acc:10.1%} {iou:8.4f}")


if __name__ == "__main__":
    main()
