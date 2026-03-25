#!/usr/bin/env python3
"""Quick sweep: columns x LTD rate for PositionalCharEncoder."""

import time
from collections import Counter

import numpy as np

import step.env  # noqa: F401
from step.config import CortexConfig
from step.cortex.sensory import SensoryRegion
from step.data import STORY_BOUNDARY, prepare_tokens_charlevel
from step.decoders.dendritic import DendriticDecoder
from step.encoders.positional import PositionalCharEncoder

tokens = prepare_tokens_charlevel(20000)
alphabet = sorted({ch for _, ch in tokens if _ != STORY_BOUNDARY})

total_non_boundary = sum(1 for tid, _ in tokens if tid != STORY_BOUNDARY)
majority_frac = (
    Counter(tid for tid, _ in tokens if tid != STORY_BOUNDARY).most_common(1)[0][1]
    / total_non_boundary
)

print(f"Majority baseline: {majority_frac:.4f}")
print(
    f"Alphabet: {len(alphabet)} chars, Positional dim: "
    f"{PositionalCharEncoder(''.join(alphabet)).input_dim}"
)
print()

configs = [
    (32, 4, 0.05),
    (32, 4, 0.1),
    (32, 4, 0.2),
    (64, 4, 0.05),
    (64, 4, 0.1),
    (64, 4, 0.2),
    (64, 8, 0.05),
    (64, 8, 0.1),
]

header = (
    f"{'cols':>4s} {'k':>2s} {'ltd':>5s} | "
    f"{'top1':>6s} {'burst':>6s} {'dead':>4s} "
    f"{'Q1':>6s} {'Q4':>6s} | {'time':>5s}"
)
print(header)
print("-" * 62)

for n_cols, k_cols, ltd_rate in configs:
    cfg = CortexConfig(
        n_columns=n_cols,
        k_columns=k_cols,
        ltd_rate=ltd_rate,
    )
    enc = PositionalCharEncoder("".join(alphabet), max_positions=8)

    region = SensoryRegion(
        input_dim=enc.input_dim,
        encoding_width=enc.encoding_width,
        n_columns=cfg.n_columns,
        n_l4=cfg.n_l4,
        n_l23=cfg.n_l23,
        k_columns=cfg.k_columns,
        voltage_decay=cfg.voltage_decay,
        eligibility_decay=cfg.eligibility_decay,
        synapse_decay=cfg.synapse_decay,
        learning_rate=cfg.learning_rate,
        max_excitability=cfg.max_excitability,
        fb_boost=cfg.fb_boost,
        ltd_rate=cfg.ltd_rate,
        burst_learning_scale=cfg.burst_learning_scale,
        seed=cfg.seed,
    )
    decoder = DendriticDecoder(source_dim=region.n_l23_total)

    prev_l23 = None
    prev_was_boundary = False
    top1_hits: list[bool] = []
    col_counts = np.zeros(n_cols)
    burst_rates: list[float] = []

    start = time.monotonic()
    for token_id, token_str in tokens:
        if token_id == STORY_BOUNDARY:
            region.reset_working_memory()
            enc.reset()
            prev_l23 = None
            prev_was_boundary = True
            continue

        if prev_l23 is not None and not prev_was_boundary:
            preds = decoder.decode(prev_l23, k=1)
            top1_hits.append(len(preds) > 0 and preds[0] == token_id)

        encoding = enc.encode(token_str)
        region.process(encoding)

        n_active = int(region.active_columns.sum())
        if n_active > 0:
            burst_rates.append(
                float(region.bursting_columns.sum()) / n_active,
            )
        col_counts += region.active_columns.astype(np.float64)

        if prev_l23 is not None and not prev_was_boundary:
            decoder.observe(token_id, prev_l23)

        prev_l23 = region.l23.active.copy()
        prev_was_boundary = False

    elapsed = time.monotonic() - start
    n = len(top1_hits)
    q = n // 4
    q1 = sum(top1_hits[:q]) / q if q else 0
    q4 = sum(top1_hits[3 * q :]) / (n - 3 * q) if n > 3 * q else 0
    top1 = sum(top1_hits) / n if n else 0
    burst = float(np.mean(burst_rates)) if burst_rates else 1.0
    dead = int((col_counts == 0).sum())

    print(
        f"{n_cols:>4d} {k_cols:>2d} {ltd_rate:>5.2f} | "
        f"{top1:>6.4f} {burst:>6.4f} {dead:>4d} "
        f"{q1:>6.4f} {q4:>6.4f} | {elapsed:>5.1f}s"
    )

print(f"\nMajority baseline: {majority_frac:.4f}")
