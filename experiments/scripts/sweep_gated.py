#!/usr/bin/env python3
"""Test three-factor gated learning rule vs original.

Minimal experiment: hash SDRs, w=3,5,10, original vs gated.
If gated w=5 beats original w=5 (and approaches w=3), it works.

Usage: uv run --extra comparison experiments/scripts/sweep_gated.py
"""

import time
from collections import Counter

import numpy as np

from step.config import EncoderConfig, ModelConfig, TrainingConfig
from step.data import (
    STORY_BOUNDARY,
    cached_token_stream,
    prepare_token_cache,
)
from step.experiment import ExperimentConfig, pretrain_step_model
from step.wrappers import StepMemoryModel

PRETRAIN_TOKENS = 50_000
EVAL_TOKENS = 5_000


def bigram_baseline(train_cache, eval_cache):
    bigrams: Counter = Counter()
    prev = None
    for token_id, _sdr in train_cache:
        if token_id == STORY_BOUNDARY:
            prev = None
            continue
        if prev is not None:
            bigrams[(prev, token_id)] += 1
        prev = token_id
    best_next: dict[int, int] = {}
    for (a, b), count in bigrams.items():
        if a not in best_next or count > bigrams.get(
            (a, best_next[a]), 0
        ):
            best_next[a] = b
    correct = 0
    total = 0
    prev = None
    for token_id, _sdr in eval_cache:
        if token_id == STORY_BOUNDARY:
            prev = None
            continue
        if prev is not None:
            if best_next.get(prev) == token_id:
                correct += 1
            total += 1
        prev = token_id
    return correct / total if total > 0 else 0.0


def run_one(enc_cfg, model_cfg, train_cache, eval_cache):
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
        name="gated",
    )

    model = StepMemoryModel(model_cfg, enc_cfg)
    pretrain_step_model(model, pretrain_cfg, train_cache)

    # Eval with story boundaries
    correct = 0
    total = 0
    ious = []
    after_boundary = False
    for t, token_id, sdr in cached_token_stream(
        eval_cache, EVAL_TOKENS
    ):
        if token_id == STORY_BOUNDARY:
            model.observe(t, token_id, sdr)
            after_boundary = True
            continue
        if t > 0 and not after_boundary:
            pred = model.predict_token(t)
            pred_sdr = model.predict_sdr(t)
            iou = len(sdr & pred_sdr) / enc_cfg.k if sdr else 0.0
            ious.append(iou)
            if pred == token_id:
                correct += 1
            total += 1
            model.learn(t, sdr, pred_sdr)
        after_boundary = False
        model.observe(t, token_id, sdr)

    acc = correct / total if total > 0 else 0.0
    mean_iou = np.mean(ious) if ious else 0.0
    return acc, mean_iou


def main():
    enc_cfg = EncoderConfig(
        model_name="gpt2", n=2048, k=40, vocab_size=10000
    )
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
    print(f"  Train: {len(train_cache):,}, Eval: {len(eval_cache):,}")

    bigram_acc = bigram_baseline(train_cache, eval_cache)
    print(f"  Bigram baseline: {bigram_acc:.1%}\n")

    windows = [3, 5, 10]
    configs = [
        ("original", 0.0, 0.0),
        ("init=0.01", 0.0, 0.01),
        ("gate+init", 0.025, 0.01),
        ("gate+init2", 0.05, 0.01),
        ("gate+init3", 0.1, 0.01),
    ]
    results = []

    for label, gate, init in configs:
        for w in windows:
            model_cfg = ModelConfig(
                n=2048,
                k=40,
                max_lr=0.5,
                weight_decay=0.999,
                penalty_factor=0.5,
                eligibility_window=w,
                relevance_gate=gate,
                weight_init=init,
            )
            start = time.monotonic()
            acc, iou = run_one(
                enc_cfg, model_cfg, train_cache, eval_cache
            )
            elapsed = time.monotonic() - start
            results.append((label, w, acc, iou, elapsed))
            print(
                f"  {label:12s} w={w:2d}: "
                f"acc={acc:.1%} iou={iou:.4f} ({elapsed:.0f}s)"
            )

    print(
        f"\n{'Rule':12s} {'w':>3s} {'Acc':>7s} {'IoU':>7s}"
    )
    print(
        f"{'bigram':12s} {'–':>3s} {bigram_acc:7.1%} {'–':>7s}"
    )
    for label, w, acc, iou, _ in results:
        print(f"{label:12s} {w:3d} {acc:7.1%} {iou:7.4f}")


if __name__ == "__main__":
    main()
