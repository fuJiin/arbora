#!/usr/bin/env python3
"""Final sweep: weight-aware decode + IoU lift metric.

Computes baseline IoU per encoding scheme (random pair overlap) and reports
IoU lift = (model_IoU - baseline_IoU) / (1 - baseline_IoU).

Usage: uv run --extra comparison experiments/scripts/sweep_final.py
"""

import itertools
import time

import numpy as np

from step.config import EncoderConfig, ModelConfig, TrainingConfig
from step.data import prepare_token_cache
from step.experiment import (
    ExperimentConfig,
    pretrain_step_model,
    run_experiment,
)
from step.sdr import AdaptiveEncoder, encode_token
from step.wrappers import StepMemoryModel

PRETRAIN_TOKENS = 50_000
EVAL_TOKENS = 5_000
BASELINE_SAMPLES = 2000  # random pairs to estimate baseline IoU


def compute_baseline_iou(enc_cfg: EncoderConfig, train_cache) -> float:
    """Compute expected IoU between random pairs of token SDRs."""
    # Collect unique token SDRs using the same encoding the model would use
    token_sdrs: dict[int, frozenset[int]] = {}

    if enc_cfg.adaptive:
        # Build adaptive SDRs by replaying the training sequence
        encoder = AdaptiveEncoder(enc_cfg)
        # Simulate encoding with active bits from a sliding window
        history: list[frozenset[int]] = []
        window = 3  # small window for context estimation
        for token_id, _hash_sdr in train_cache:
            active = []
            for sdr in history[-window:]:
                active.extend(sdr)
            sdr = encoder.encode(token_id, active or None)
            if token_id not in token_sdrs:
                token_sdrs[token_id] = sdr
            history.append(sdr)
            if len(history) > window:
                history.pop(0)
    else:
        # Hash-based: deterministic per token
        for token_id, _sdr in train_cache:
            if token_id not in token_sdrs:
                token_sdrs[token_id] = encode_token(token_id, enc_cfg)

    # Sample random pairs and compute IoU
    sdr_list = list(token_sdrs.values())
    if len(sdr_list) < 2:
        return 0.0

    rng = np.random.default_rng(42)
    k = enc_cfg.k
    total_iou = 0.0
    n_pairs = min(BASELINE_SAMPLES, len(sdr_list) * (len(sdr_list) - 1) // 2)
    for _ in range(n_pairs):
        i, j = rng.choice(len(sdr_list), 2, replace=False)
        overlap = len(sdr_list[i] & sdr_list[j])
        total_iou += overlap / k

    return total_iou / n_pairs


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
    # Cache data once
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

    # Define encoding schemes
    schemes = {
        "hash": EncoderConfig(
            model_name="gpt2", n=2048, k=40, vocab_size=10000,
        ),
        "active f=0.3": EncoderConfig(
            model_name="gpt2", n=2048, k=40, vocab_size=10000,
            adaptive=True, context_fraction=0.3, seeding="active",
        ),
        "predict f=0.3": EncoderConfig(
            model_name="gpt2", n=2048, k=40, vocab_size=10000,
            adaptive=True, context_fraction=0.3, seeding="predicted",
        ),
        "predict f=0.5": EncoderConfig(
            model_name="gpt2", n=2048, k=40, vocab_size=10000,
            adaptive=True, context_fraction=0.5, seeding="predicted",
        ),
    }

    # Compute baseline IoU per scheme
    print("Computing baseline IoU per encoding scheme...")
    baselines: dict[str, float] = {}
    for name, enc_cfg in schemes.items():
        bl = compute_baseline_iou(enc_cfg, train_cache)
        baselines[name] = bl
        print(f"  {name:15s}: baseline IoU = {bl:.4f}")
    print()

    # Run sweep
    windows = [3, 5, 10]
    results = []
    for name, enc_cfg in schemes.items():
        for w in windows:
            start = time.monotonic()
            acc, iou = run_one(enc_cfg, w, train_cache, eval_cache)
            elapsed = time.monotonic() - start
            bl = baselines[name]
            lift = (iou - bl) / (1 - bl) if bl < 1.0 else 0.0
            results.append((name, w, acc, iou, bl, lift, elapsed))
            print(
                f"  {name:15s} w={w:2d}: "
                f"acc={acc:.1%} iou={iou:.4f} "
                f"baseline={bl:.4f} lift={lift:.4f} "
                f"({elapsed:.0f}s)"
            )

    print(f"\n{'Scheme':15s} {'w':>3s} {'Acc':>7s} {'IoU':>7s} "
          f"{'Base':>7s} {'Lift':>7s}")
    for name, w, acc, iou, bl, lift, _ in results:
        print(f"{name:15s} {w:3d} {acc:7.1%} {iou:7.4f} "
              f"{bl:7.4f} {lift:7.4f}")


if __name__ == "__main__":
    main()
