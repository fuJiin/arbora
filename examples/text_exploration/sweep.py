#!/usr/bin/env python3
"""Sweep ltd_rate / synapse_decay / learning_rate for T1.

PR B's diagnostics pinpointed `ff_weights` saturation (69% near the
clip ceiling) as the acute bottleneck — L4 and L2/3 SDRs collapse on
vowels because the input layer can't differentiate them. This script
sweeps the three knobs most directly controlling ff weight equilibrium
and reports accuracy + saturation + SDR overlap per config.

Measurement matrix per config:
- `test_acc` / `test_bpc` — held-out next-char prediction
- `ff_mean` / `ff_near_0` / `ff_near_1` — ff weight distribution
- `l4_within_vowel` / `l4_across` — L4 SDR overlap (collapse indicator)
- `l23_within_vowel` / `l23_across` — L2/3 SDR overlap

Usage:
    uv run python -m examples.text_exploration.sweep
    uv run python -m examples.text_exploration.sweep --epochs 3 --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from itertools import product
from pathlib import Path

from arbora.config import _default_t1_config, make_sensory_region
from arbora.decoders.dendritic import DendriticDecoder
from arbora.encoders.onehot import OneHotCharEncoder
from arbora.probes.bpc import BPCProbe
from examples.text_exploration.data import (
    DEFAULT_ALPHABET,
    alphabet_filter,
    load_words,
    train_test_split,
)
from examples.text_exploration.diagnostics import (
    character_sdr_overlap,
    weight_distribution,
)
from examples.text_exploration.trainer import T1Trainer


def build_region(
    *,
    input_dim: int,
    ltd_rate: float,
    synapse_decay: float,
    learning_rate: float,
    n_columns: int | None = None,
    k_columns: int | None = None,
    seed: int = 0,
):
    """T1 with overridden saturation-relevant knobs. L5 disabled (not used).

    `n_columns` / `k_columns` override the T1 defaults when set — used
    by the capacity sweep. k scales with cols so the activation fraction
    stays ~6.25%.
    """
    cfg = _default_t1_config()
    cfg.n_l5 = 0
    cfg.ltd_rate = ltd_rate
    cfg.synapse_decay = synapse_decay
    cfg.learning_rate = learning_rate
    if n_columns is not None:
        cfg.n_columns = n_columns
    if k_columns is not None:
        cfg.k_columns = k_columns
    return make_sensory_region(cfg, input_dim=input_dim, encoding_width=0, seed=seed)


def run_config(
    *,
    ltd_rate: float,
    synapse_decay: float,
    learning_rate: float,
    train_words: list[str],
    test_words: list[str],
    epochs: int = 2,
    n_columns: int | None = None,
    k_columns: int | None = None,
    seed: int = 0,
) -> dict:
    encoder = OneHotCharEncoder(chars=DEFAULT_ALPHABET)
    region = build_region(
        input_dim=encoder.input_dim,
        ltd_rate=ltd_rate,
        synapse_decay=synapse_decay,
        learning_rate=learning_rate,
        n_columns=n_columns,
        k_columns=k_columns,
        seed=seed,
    )
    decoder = DendriticDecoder(source_dim=region.n_l23_total, seed=seed)
    bpc = BPCProbe()
    trainer = T1Trainer(region, encoder, decoder=decoder, bpc_probe=bpc)

    for _ in range(epochs):
        for w in train_words:
            trainer.train_word(w, train=True)

    # Eval on held-out: fresh BPC counter, no learning.
    bpc.reset()
    n_correct = 0
    n_chars = 0
    for w in test_words:
        for r in trainer.train_word(w, train=False):
            n_chars += 1
            if r.top1_correct:
                n_correct += 1

    test_acc = n_correct / max(n_chars, 1)
    test_bpc = bpc.bpc

    sdr = character_sdr_overlap(trainer)
    weights = weight_distribution(trainer)

    return {
        "ltd_rate": ltd_rate,
        "synapse_decay": synapse_decay,
        "learning_rate": learning_rate,
        "n_columns": region.n_columns,
        "k_columns": region.k_columns,
        "epochs": epochs,
        "test_acc": test_acc,
        "test_bpc": test_bpc,
        "ff_mean": weights.ff.mean,
        "ff_near_0": weights.ff.frac_near_zero,
        "ff_near_1": weights.ff.frac_near_one,
        "l4_within_vowel": sdr.l4.within_vowel_mean,
        "l4_within_consonant": sdr.l4.within_consonant_mean,
        "l4_across": sdr.l4.across_mean,
        "l23_within_vowel": sdr.l23.within_vowel_mean,
        "l23_across": sdr.l23.across_mean,
    }


def format_row(r: dict) -> str:
    return (
        f"cols={r['n_columns']:>4d} k={r['k_columns']:>2d} ep={r['epochs']} "
        f"ltd={r['ltd_rate']:.2f} dec={r['synapse_decay']:.3f} "
        f"lr={r['learning_rate']:.2f} | "
        f"acc={r['test_acc']:.3f} bpc={r['test_bpc']:.2f} | "
        f"ff(near1)={r['ff_near_1']:.2f} ff(mean)={r['ff_mean']:.2f} | "
        f"l4_vv={r['l4_within_vowel']:.2f} l4_xx={r['l4_across']:.2f} | "
        f"l23_vv={r['l23_within_vowel']:.2f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="T1 saturation / capacity sweep")
    p.add_argument("--epochs", type=int, nargs="+", default=[2])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--csv", type=str, default=None, help="Write results CSV to this path"
    )
    p.add_argument("--ltd", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    p.add_argument("--decay", type=float, nargs="+", default=[1.0, 0.999, 0.995])
    p.add_argument("--lr", type=float, nargs="+", default=[0.05, 0.02])
    p.add_argument(
        "--cols",
        type=int,
        nargs="+",
        default=[None],
        help="Column counts to sweep (None = T1 default 128). k scales with cols.",
    )
    args = p.parse_args()

    words = alphabet_filter(load_words(), DEFAULT_ALPHABET)
    train_words, test_words = train_test_split(words, test_frac=0.2, seed=args.seed)
    configs = list(product(args.ltd, args.decay, args.lr, args.cols, args.epochs))
    print(
        f"{len(configs)} configs "
        f"({len(train_words)} train / {len(test_words)} test words)"
    )

    rows: list[dict] = []
    t_start = time.monotonic()
    for i, (ltd, decay, lr, cols, epochs) in enumerate(configs):
        t0 = time.monotonic()
        # k scales with cols to keep activation fraction ~6.25% (T1 default).
        k = None if cols is None else max(1, cols // 16)
        r = run_config(
            ltd_rate=ltd,
            synapse_decay=decay,
            learning_rate=lr,
            train_words=train_words,
            test_words=test_words,
            epochs=epochs,
            n_columns=cols,
            k_columns=k,
            seed=args.seed,
        )
        rows.append(r)
        dt = time.monotonic() - t0
        print(f"[{i + 1:2d}/{len(configs)}] {format_row(r)} ({dt:.1f}s)")

    total = time.monotonic() - t_start
    print(f"\nTotal: {total:.1f}s")

    # Ranked summary: best by accuracy, best by lowest ff saturation.
    print("\n=== Top 5 by test_acc ===")
    for r in sorted(rows, key=lambda x: -x["test_acc"])[:5]:
        print("  " + format_row(r))
    print("\n=== Top 5 by lowest ff_near_1 ===")
    for r in sorted(rows, key=lambda x: x["ff_near_1"])[:5]:
        print("  " + format_row(r))
    print("\n=== Top 5 by lowest l4_within_vowel (least collapse) ===")
    for r in sorted(rows, key=lambda x: x["l4_within_vowel"])[:5]:
        print("  " + format_row(r))

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
