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
from arbora.cortex.circuit import Circuit
from arbora.decoders.dendritic import DendriticDecoder
from arbora.encoders.onehot import OneHotCharEncoder
from arbora.probes.bpc import BPCProbe
from arbora.probes.core import LaminaProbe
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
    perm_init: float | None = None,
    perm_increment: float | None = None,
    seg_activation_threshold: int | None = None,
    seed: int = 0,
):
    """T1 with overridden saturation- and segment-relevant knobs.

    L5 disabled (not used). `n_columns` / `k_columns` override the T1
    defaults when set (k scales with cols so activation fraction stays
    near 6.25%). `perm_init` / `perm_increment` / `seg_activation_threshold`
    control segment growth dynamics.
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
    if perm_init is not None:
        cfg.perm_init = perm_init
    if perm_increment is not None:
        cfg.perm_increment = perm_increment
    if seg_activation_threshold is not None:
        cfg.seg_activation_threshold = seg_activation_threshold
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
    perm_init: float | None = None,
    perm_increment: float | None = None,
    seg_activation_threshold: int | None = None,
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
        perm_init=perm_init,
        perm_increment=perm_increment,
        seg_activation_threshold=seg_activation_threshold,
        seed=seed,
    )
    decoder = DendriticDecoder(source_dim=region.n_l23_total, seed=seed)
    bpc = BPCProbe()
    trainer = T1Trainer(region, encoder, decoder=decoder, bpc_probe=bpc)

    for _ in range(epochs):
        for w in train_words:
            trainer.train_word(w, train=True)

    # Eval on held-out. Fresh LaminaProbe so recall/precision reflect
    # the trained state only (not noisy training dynamics). Wrap the
    # region in a minimal Circuit so LaminaProbe can walk it.
    probe_circuit = Circuit(encoder)
    probe_circuit.add_region("T1", region, entry=True, input_region=True)
    probe_circuit.finalize()
    lamina = LaminaProbe()

    bpc.reset()
    n_correct = 0
    n_chars = 0
    for w in test_words:
        trainer.reset()
        for c in w:
            r = trainer.step(c, train=False)
            n_chars += 1
            if r.top1_correct:
                n_correct += 1
            # Read L4 predicted+active after step — this is the
            # region's surprise / prediction-match signal.
            lamina.observe(probe_circuit)

    test_acc = n_correct / max(n_chars, 1)
    test_bpc = bpc.bpc
    snap = lamina.snapshot()["T1"]

    sdr = character_sdr_overlap(trainer)
    weights = weight_distribution(trainer)

    return {
        "ltd_rate": ltd_rate,
        "synapse_decay": synapse_decay,
        "learning_rate": learning_rate,
        "n_columns": region.n_columns,
        "k_columns": region.k_columns,
        "perm_init": region.perm_init,
        "perm_increment": region.perm_increment,
        "seg_activation_threshold": region.seg_activation_threshold,
        "epochs": epochs,
        # Region-intrinsic surprise / prediction quality. L4 and L2/3.
        "l4_recall": snap.input.recall,
        "l4_precision": snap.input.precision,
        "l4_sparseness": snap.input.sparseness,
        "l23_recall": snap.association.recall,
        "l23_precision": snap.association.precision,
        "l23_sparseness": snap.association.sparseness,
        "l23_eff_dim": snap.association.eff_dim,
        # Decoder-based — kept for comparison, demoted in ranking.
        "test_acc": test_acc,
        "test_bpc": test_bpc,
        # ff weights.
        "ff_mean": weights.ff.mean,
        "ff_near_0": weights.ff.frac_near_zero,
        "ff_near_1": weights.ff.frac_near_one,
        # Segment utilization.
        "l4_seg_mean": weights.l4_lat_perm.mean,
        "l4_seg_near_0": weights.l4_lat_perm.frac_near_zero,
        "l23_seg_mean": weights.l23_seg_perm.mean,
        "l23_seg_near_0": weights.l23_seg_perm.frac_near_zero,
        # SDR structure (isolated reset+1step).
        "l4_within_vowel": sdr.l4.within_vowel_mean,
        "l4_within_consonant": sdr.l4.within_consonant_mean,
        "l4_across": sdr.l4.across_mean,
        "l23_within_vowel": sdr.l23.within_vowel_mean,
        "l23_across": sdr.l23.across_mean,
    }


def format_row(r: dict) -> str:
    return (
        f"cols={r['n_columns']:>4d} k={r['k_columns']:>2d} ep={r['epochs']} "
        f"pi={r['perm_init']:.2f} pinc={r['perm_increment']:.2f} "
        f"sat={r['seg_activation_threshold']} | "
        f"L4(rec={r['l4_recall']:.2f} pre={r['l4_precision']:.2f}) "
        f"L23(rec={r['l23_recall']:.2f} pre={r['l23_precision']:.2f} "
        f"ed={r['l23_eff_dim']:.1f}) | "
        f"segs(l4m={r['l4_seg_mean']:.2f} l23m={r['l23_seg_mean']:.2f}) | "
        f"l4_vv={r['l4_within_vowel']:.2f} acc={r['test_acc']:.3f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="T1 saturation / capacity / segment sweep")
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
    p.add_argument(
        "--perm-init",
        type=float,
        nargs="+",
        default=[None],
        help="Segment perm_init (None = default 0.6).",
    )
    p.add_argument(
        "--perm-increment",
        type=float,
        nargs="+",
        default=[None],
        help="Segment perm_increment (None = default 0.2).",
    )
    p.add_argument(
        "--seg-threshold",
        type=int,
        nargs="+",
        default=[None],
        help="Segment activation threshold (None = default 2).",
    )
    args = p.parse_args()

    words = alphabet_filter(load_words(), DEFAULT_ALPHABET)
    train_words, test_words = train_test_split(words, test_frac=0.2, seed=args.seed)
    configs = list(
        product(
            args.ltd,
            args.decay,
            args.lr,
            args.cols,
            args.epochs,
            args.perm_init,
            args.perm_increment,
            args.seg_threshold,
        )
    )
    print(
        f"{len(configs)} configs "
        f"({len(train_words)} train / {len(test_words)} test words)"
    )

    rows: list[dict] = []
    t_start = time.monotonic()
    for i, (ltd, decay, lr, cols, epochs, pi, pinc, sat) in enumerate(configs):
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
            perm_init=pi,
            perm_increment=pinc,
            seg_activation_threshold=sat,
            seed=args.seed,
        )
        rows.append(r)
        dt = time.monotonic() - t0
        print(f"[{i + 1:2d}/{len(configs)}] {format_row(r)} ({dt:.1f}s)")

    total = time.monotonic() - t_start
    print(f"\nTotal: {total:.1f}s")

    # Ranked summary: prioritize region-intrinsic surprise metrics.
    # test_acc is kept as the last ranking to spot decoder/region coupling.
    print("\n=== Top 5 by highest L4 recall (lowest surprise) ===")
    for r in sorted(rows, key=lambda x: -x["l4_recall"])[:5]:
        print("  " + format_row(r))
    print("\n=== Top 5 by highest L2/3 recall ===")
    for r in sorted(rows, key=lambda x: -x["l23_recall"])[:5]:
        print("  " + format_row(r))
    print("\n=== Top 5 by highest L2/3 eff_dim (richness) ===")
    for r in sorted(rows, key=lambda x: -x["l23_eff_dim"])[:5]:
        print("  " + format_row(r))
    print("\n=== Top 5 by highest L2/3 segment mean perm (most growth) ===")
    for r in sorted(rows, key=lambda x: -x["l23_seg_mean"])[:5]:
        print("  " + format_row(r))
    print("\n=== Top 5 by test_acc (decoder-coupled, for comparison) ===")
    for r in sorted(rows, key=lambda x: -x["test_acc"])[:5]:
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
