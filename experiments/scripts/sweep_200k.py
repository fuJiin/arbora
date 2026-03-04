#!/usr/bin/env python3
"""Scale active f=0.3 to 200K pretrain with IoU lift.

Usage: uv run --extra comparison experiments/scripts/sweep_200k.py
"""

import time

import numpy as np

from step.config import EncoderConfig, ModelConfig, TrainingConfig
from step.data import prepare_token_cache
from step.experiment import (
    ExperimentConfig,
    pretrain_step_model,
    run_experiment,
)
from step.sdr import encode_token
from step.wrappers import StepMemoryModel

PRETRAIN_TOKENS = 200_000
EVAL_TOKENS = 10_000
BASELINE_SAMPLES = 2000


def compute_baseline_iou(enc_cfg, train_cache):
    token_sdrs = {}
    for token_id, _sdr in train_cache:
        if token_id not in token_sdrs:
            token_sdrs[token_id] = encode_token(token_id, enc_cfg)
    sdr_list = list(token_sdrs.values())
    rng = np.random.default_rng(42)
    k = enc_cfg.k
    total = 0.0
    n = min(BASELINE_SAMPLES, len(sdr_list) * (len(sdr_list) - 1) // 2)
    for _ in range(n):
        i, j = rng.choice(len(sdr_list), 2, replace=False)
        total += len(sdr_list[i] & sdr_list[j]) / k
    return total / n


def run_one(enc_cfg, window, train_cache, eval_cache):
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
        max_tokens=EVAL_TOKENS, log_interval=2_000,
    )
    pretrain_cfg = ExperimentConfig(
        encoder=enc_cfg, model=model_cfg, training=train_tc, name="sweep200k",
    )
    eval_cfg = ExperimentConfig(
        encoder=enc_cfg, model=model_cfg, training=eval_tc, name="sweep200k",
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
    base_enc = EncoderConfig(model_name="gpt2", n=2048, k=40, vocab_size=10000)
    train_tc = TrainingConfig(
        dataset_name="roneneldan/TinyStories", dataset_split="train",
        max_tokens=PRETRAIN_TOKENS,
    )
    eval_tc = TrainingConfig(
        dataset_name="roneneldan/TinyStories", dataset_split="validation",
        max_tokens=EVAL_TOKENS,
    )

    print("Caching data...")
    train_cache = prepare_token_cache(train_tc, base_enc)
    eval_cache = prepare_token_cache(eval_tc, base_enc)
    print(f"  Train: {len(train_cache):,}, Eval: {len(eval_cache):,}\n")

    schemes = {
        "hash": EncoderConfig(
            model_name="gpt2", n=2048, k=40, vocab_size=10000),
        "active f=0.3": EncoderConfig(
            model_name="gpt2", n=2048, k=40, vocab_size=10000,
            adaptive=True, context_fraction=0.3, seeding="active"),
    }

    # Baselines
    print("Computing baseline IoU...")
    baselines = {}
    for name, enc_cfg in schemes.items():
        bl = compute_baseline_iou(enc_cfg, train_cache)
        baselines[name] = bl
        print(f"  {name}: {bl:.4f}")
    print()

    windows = [3, 5, 10]
    results = []
    for name, enc_cfg in schemes.items():
        for w in windows:
            start = time.monotonic()
            acc, iou = run_one(enc_cfg, w, train_cache, eval_cache)
            elapsed = time.monotonic() - start
            bl = baselines[name]
            lift = (iou - bl) / (1 - bl)
            results.append((name, w, acc, iou, bl, lift, elapsed))
            print(f"  {name:15s} w={w:2d}: acc={acc:.1%} iou={iou:.4f} "
                  f"lift={lift:.4f} ({elapsed:.0f}s)")

    print(f"\n{'Scheme':15s} {'w':>3s} {'Acc':>7s} {'IoU':>7s} "
          f"{'Base':>7s} {'Lift':>7s}")
    for name, w, acc, iou, bl, lift, _ in results:
        print(f"{name:15s} {w:3d} {acc:7.1%} {iou:7.4f} "
              f"{bl:7.4f} {lift:7.4f}")


if __name__ == "__main__":
    main()
