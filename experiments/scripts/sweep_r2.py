#!/usr/bin/env python3
"""Sweep S2 parameters for char-level hierarchy.

Measures S2 column selectivity and context discrimination to find
configs where S2 actually differentiates tokens.
"""

import time

import numpy as np

import step.env  # noqa: F401
from step.config import CortexConfig
from step.cortex.sensory import SensoryRegion
from step.cortex.surprise import SurpriseTracker
from step.data import STORY_BOUNDARY, prepare_tokens_charlevel
from step.encoders.positional import PositionalCharEncoder
from step.probes.representation import RepresentationTracker

tokens = prepare_tokens_charlevel(20000)
alphabet = sorted({ch for _, ch in tokens if _ != STORY_BOUNDARY})


def run_hierarchy(r2_cols, r2_k, r2_lr, r2_ltd, r2_vdecay):
    """Run hierarchy, return S2 selectivity and context discrimination."""
    r1_cfg = CortexConfig(ltd_rate=0.05)
    enc = PositionalCharEncoder("".join(alphabet), max_positions=8)

    r1 = SensoryRegion(
        input_dim=enc.input_dim,
        encoding_width=enc.encoding_width,
        n_columns=r1_cfg.n_columns,
        n_l4=r1_cfg.n_l4,
        n_l23=r1_cfg.n_l23,
        k_columns=r1_cfg.k_columns,
        voltage_decay=r1_cfg.voltage_decay,
        eligibility_decay=r1_cfg.eligibility_decay,
        synapse_decay=r1_cfg.synapse_decay,
        learning_rate=r1_cfg.learning_rate,
        max_excitability=r1_cfg.max_excitability,
        fb_boost=r1_cfg.fb_boost,
        ltd_rate=r1_cfg.ltd_rate,
        burst_learning_scale=r1_cfg.burst_learning_scale,
        seed=0,
    )
    r2 = SensoryRegion(
        input_dim=r1.n_l23_total,
        encoding_width=0,
        n_columns=r2_cols,
        n_l4=4,
        n_l23=4,
        k_columns=r2_k,
        voltage_decay=r2_vdecay,
        eligibility_decay=0.98,
        synapse_decay=0.9999,
        learning_rate=r2_lr,
        ltd_rate=r2_ltd,
        seed=123,
    )

    surprise = SurpriseTracker()
    rep2 = RepresentationTracker(r2.n_columns, r2.n_l4)

    start = time.monotonic()
    for token_id, token_str in tokens:
        if token_id == STORY_BOUNDARY:
            r1.reset_working_memory()
            r2.reset_working_memory()
            enc.reset()
            rep2.reset_context()
            continue

        encoding = enc.encode(token_str)
        r1.process(encoding)

        n_active = int(r1.active_columns.sum())
        n_burst = int(r1.bursting_columns.sum())
        burst_rate = n_burst / max(n_active, 1)
        mod = surprise.update(burst_rate)
        r2.surprise_modulator = mod

        r2.process(r1.firing_rate_l23)
        rep2.observe(token_id, r2.active_columns, r2.active_l4)

    elapsed = time.monotonic() - start
    summ = rep2.summary(r2.ff_weights)
    sel = rep2.column_selectivity()

    return {
        "selectivity": summ.get("column_selectivity_mean", 1.0),
        "ctx_disc": summ.get("context_discrimination", 0.0),
        "sim": summ.get("similarity_mean", 0.0),
        "dead": int((np.array(sel["per_column"]) > 0.99).sum()),
        "elapsed": elapsed,
    }


configs = [
    # (cols, k, lr, ltd, v_decay)
    (16, 2, 0.01, 0.4, 0.8),   # current default
    (16, 2, 0.05, 0.1, 0.8),   # higher lr, lower ltd
    (16, 2, 0.05, 0.05, 0.8),  # match S1 ltd
    (16, 2, 0.05, 0.2, 0.5),   # faster voltage decay
    (16, 4, 0.05, 0.1, 0.8),   # more active cols
    (16, 4, 0.05, 0.05, 0.5),  # more active + faster decay
    (32, 4, 0.05, 0.1, 0.8),   # more cols
    (32, 4, 0.05, 0.05, 0.5),  # more cols + faster
    (32, 4, 0.05, 0.1, 0.5),   # more cols + faster decay
    (32, 8, 0.05, 0.05, 0.5),  # many active cols
]

header = (
    f"{'cols':>4s} {'k':>2s} {'lr':>5s} {'ltd':>5s} {'vd':>4s} | "
    f"{'sel':>5s} {'ctx':>5s} {'sim':>5s} {'dead':>4s} | {'time':>5s}"
)
print(header)
print("-" * 62)

for cols, k, lr, ltd, vd in configs:
    r = run_hierarchy(cols, k, lr, ltd, vd)
    print(
        f"{cols:>4d} {k:>2d} {lr:>5.2f} {ltd:>5.2f} {vd:>4.1f} | "
        f"{r['selectivity']:>5.3f} {r['ctx_disc']:>5.3f} "
        f"{r['sim']:>5.3f} {r['dead']:>4d} | "
        f"{r['elapsed']:>5.1f}s"
    )

print("\nGoal: selectivity < 0.7, ctx_disc > 0.9, dead = 0")
