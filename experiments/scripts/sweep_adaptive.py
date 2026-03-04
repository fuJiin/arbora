#!/usr/bin/env python3
"""Quick sweep: adaptive vs hash-based encoding across window sizes.

Usage: uv run --extra comparison experiments/scripts/sweep_adaptive.py
"""

import json
import time
from pathlib import Path

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


def run_one(adaptive: bool, window: int, train_cache, eval_cache) -> float:
    enc_cfg = EncoderConfig(
        model_name="gpt2", n=2048, k=40, vocab_size=10000,
        adaptive=adaptive, context_fraction=0.5,
    )
    model_cfg = ModelConfig(
        n=2048, k=40, max_lr=0.5, weight_decay=0.999,
        penalty_factor=0.5, eligibility_window=window,
    )
    train_tc = TrainingConfig(
        dataset_name="roneneldan/TinyStories",
        dataset_split="train",
        max_tokens=PRETRAIN_TOKENS,
        log_interval=1_000,
    )
    eval_tc = TrainingConfig(
        dataset_name="roneneldan/TinyStories",
        dataset_split="validation",
        max_tokens=EVAL_TOKENS,
        log_interval=1_000,
    )

    pretrain_config = ExperimentConfig(
        encoder=enc_cfg, model=model_cfg, training=train_tc, name="sweep",
    )
    eval_config = ExperimentConfig(
        encoder=enc_cfg, model=model_cfg, training=eval_tc, name="sweep",
    )

    model = StepMemoryModel(model_cfg, enc_cfg)
    pretrain_step_model(model, pretrain_config, train_cache)

    def factory(config, _m=model):
        return _m

    result = run_experiment(eval_config, factory, "sweep", "iou", eval_cache)
    final_acc = result.rolling_accuracies[-1][1] if result.rolling_accuracies else 0.0
    return final_acc


def main():
    # Cache data once (hash-based SDRs — adaptive models will override)
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
    print(f"  Train: {len(train_cache):,}, Eval: {len(eval_cache):,}")
    print()

    configs = [
        ("hash", 3), ("hash", 5), ("hash", 10),
        ("adaptive", 3), ("adaptive", 5), ("adaptive", 10),
    ]

    results = []
    for encoding, window in configs:
        adaptive = encoding == "adaptive"
        start = time.monotonic()
        acc = run_one(adaptive, window, train_cache, eval_cache)
        elapsed = time.monotonic() - start
        results.append((encoding, window, acc, elapsed))
        print(f"  {encoding:8s} w={window:2d}: {acc:.1%} ({elapsed:.1f}s)")

    print("\n--- Summary ---")
    print(f"{'Encoding':10s} {'Window':>6s} {'Accuracy':>10s}")
    for encoding, window, acc, _ in results:
        print(f"{encoding:10s} {window:6d} {acc:10.1%}")


if __name__ == "__main__":
    main()
