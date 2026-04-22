#!/usr/bin/env python3
"""Train T1 on a dictionary stream, evaluate on held-out words.

ARB-131 M1 success criterion: "T1 predicts next character above chance
on held-out dictionary words, with context-sensitive predictions."

This script is the minimum viable wiring to check that criterion. It
does not build the explorer (PR C) or the full diagnostic-checkpoint
suite (PR B); those layer on top of this substrate.

Usage:
    uv run examples/text_exploration/train.py
    uv run examples/text_exploration/train.py --epochs 5 --test-frac 0.2
"""

from __future__ import annotations

import argparse
import math
import time

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
    context_sensitivity,
    format_diagnostics,
    weight_distribution,
)
from examples.text_exploration.trainer import T1Trainer


def build_t1(encoder: OneHotCharEncoder, seed: int = 0):
    """Build a T1 region from the canonical char-level defaults.

    `_default_t1_config` is tuned for 128 columns at k=8 per column at
    char-level input (~6.25% activation). `encoding_width=0` gives
    full L4-to-column connectivity, which is what we want for single-
    char-per-step input that has no positional substructure.
    """
    cfg = _default_t1_config()
    return make_sensory_region(
        cfg,
        input_dim=encoder.input_dim,
        encoding_width=0,
        seed=seed,
    )


def run_stream(
    trainer: T1Trainer,
    words: list[str],
    *,
    train: bool,
    log_every: int = 0,
) -> tuple[int, int]:
    """Iterate words, reset per word, step each char.

    Returns `(n_chars, n_top1_correct)`. BPC info accumulates on the
    trainer's `bpc_probe`.
    """
    n_chars = 0
    n_correct = 0
    t0 = time.monotonic()
    for w_idx, word in enumerate(words):
        trainer.reset()
        for c in word:
            r = trainer.step(c, train=train)
            n_chars += 1
            if r.top1_correct:
                n_correct += 1
        if log_every and (w_idx + 1) % log_every == 0:
            elapsed = time.monotonic() - t0
            acc = n_correct / max(n_chars, 1)
            print(
                f"  word {w_idx + 1}/{len(words)}: "
                f"chars={n_chars} acc={acc:.3f} "
                f"bpc(recent)={trainer.bpc_probe.recent_bpc:.3f} "
                f"({elapsed:.1f}s)"
            )
    return n_chars, n_correct


def main() -> None:
    parser = argparse.ArgumentParser(
        description="T1 next-char prediction on a dictionary stream"
    )
    parser.add_argument(
        "--words",
        type=str,
        default=None,
        help="Wordlist path (default: data/text/common_words.txt)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Passes over the training wordlist"
    )
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-every",
        type=int,
        default=200,
        help="Print a progress line every N words (0 = silent)",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="After training, run the three diagnostic checkpoints",
    )
    args = parser.parse_args()

    all_words = alphabet_filter(load_words(args.words), DEFAULT_ALPHABET)
    train_words, test_words = train_test_split(
        all_words, test_frac=args.test_frac, seed=args.seed
    )
    print(
        f"Loaded {len(all_words)} words "
        f"(train={len(train_words)}, test={len(test_words)})"
    )

    encoder = OneHotCharEncoder(chars=DEFAULT_ALPHABET)
    region = build_t1(encoder, seed=args.seed)
    # Decoder + BPC are optional — wired in here so the CLI reports
    # accuracy/BPC, but the trainer itself doesn't require them. The
    # decoder reads L2/3 active (per the ticket spec: L2/3 is the
    # site of next-char prediction).
    decoder = DendriticDecoder(source_dim=region.n_l23_total, seed=args.seed)
    bpc = BPCProbe()
    trainer = T1Trainer(region, encoder, decoder=decoder, bpc_probe=bpc)

    # --- Train ---
    print(f"\nTraining for {args.epochs} epoch(s)...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        bpc.reset()
        n_c, n_ok = run_stream(
            trainer, train_words, train=True, log_every=args.log_every
        )
        print(
            f"  epoch done: chars={n_c} train_acc={n_ok / max(n_c, 1):.3f} "
            f"train_bpc={bpc.bpc:.3f}"
        )

    # --- Eval on held-out ---
    print("\nEvaluating on held-out test words...")
    bpc.reset()
    n_c, n_ok = run_stream(trainer, test_words, train=False, log_every=0)
    test_acc = n_ok / max(n_c, 1)
    uniform_bpc = math.log2(len(DEFAULT_ALPHABET))  # ~4.7 bits for 26 chars
    print(
        f"  test_chars={n_c} test_acc={test_acc:.3f} test_bpc={bpc.bpc:.3f} "
        f"(uniform-{len(DEFAULT_ALPHABET)}-char baseline={uniform_bpc:.2f})"
    )
    chance = 1.0 / len(DEFAULT_ALPHABET)
    above_chance = test_acc > chance
    print(f"  above-chance (1/{len(DEFAULT_ALPHABET)}={chance:.3f}): {above_chance}")

    if args.diagnostics:
        print("\nRunning diagnostic checkpoints...\n")
        sdr = character_sdr_overlap(trainer)
        ctx = context_sensitivity(
            trainer,
            prefixes=["c", "ca", "de", "sh", "th", "str"],
        )
        weights = weight_distribution(trainer)
        print(format_diagnostics(sdr, ctx, weights))


if __name__ == "__main__":
    main()
