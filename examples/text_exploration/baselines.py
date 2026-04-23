#!/usr/bin/env python3
"""Baselines for characterizing T1's contribution to next-char prediction.

Three reference points for the "is T1 learning?" and "how does T1 compare?"
questions:

1. **untrained T1 + dendritic decoder** — T1 with learning disabled from
   step 0 (random Hebbian state). Isolates "is the region contributing?"
   from "is the decoder just memorizing."
2. **bigram / trigram Markov** — classical n-gram char-level next-token
   predictors. Zero neural machinery. Puts T1 on a known reference scale
   at the "simple sequence learner" end.
3. **linear probe on L2/3** — train a logistic-regression classifier on
   trained T1's L2/3 active → next-char instead of the DendriticDecoder.
   Isolates "is the decoder the bottleneck" from "is the representation
   the bottleneck."

Uses the same TinyStories pipeline and train/test split as `sweep.py`.
Default config matches the current best baseline (ptd=0.50, ep=20).
"""

from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict

import numpy as np

from arbora.decoders.dendritic import DendriticDecoder
from arbora.encoders.onehot import OneHotCharEncoder
from arbora.probes.bpc import BPCProbe
from examples.text_exploration.data import (
    DEFAULT_ALPHABET,
    alphabet_filter,
    load_natural_chunks,
    load_words,
    shuffle_chunks,
    split_chunks,
    wordlist_chunks,
)
from examples.text_exploration.sweep import build_region
from examples.text_exploration.trainer import T1Trainer

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _load_chunks(dataset: str, *, max_chars: int, alphabet: str) -> list[str]:
    if dataset == "wordlist":
        words = alphabet_filter(load_words(), alphabet)
        return wordlist_chunks(words, append_space=True)
    return load_natural_chunks(dataset, max_chars=max_chars, alphabet=alphabet)


# ---------------------------------------------------------------------------
# Baseline 1: untrained T1 + dendritic decoder
# ---------------------------------------------------------------------------


def _build_trainer(*, learning_enabled: bool, seed: int, **region_kwargs) -> T1Trainer:
    encoder = OneHotCharEncoder(chars=DEFAULT_ALPHABET)
    region = build_region(input_dim=encoder.input_dim, seed=seed, **region_kwargs)
    region.learning_enabled = learning_enabled
    decoder = DendriticDecoder(source_dim=region.n_l23_total, seed=seed)
    return T1Trainer(region, encoder, decoder=decoder, bpc_probe=BPCProbe())


def run_t1_decoder(
    train_chunks: list[str],
    test_chunks: list[str],
    *,
    learning_enabled: bool,
    epochs: int,
    ptd: float,
    ltd: float,
    decay: float,
    lr: float,
    seed: int,
) -> dict:
    """Run T1 + dendritic decoder end-to-end, with region learning toggleable."""
    trainer = _build_trainer(
        learning_enabled=learning_enabled,
        seed=seed,
        ltd_rate=ltd,
        synapse_decay=decay,
        learning_rate=lr,
        pre_trace_decay=ptd,
    )
    train_rng = np.random.default_rng(seed)
    t0 = time.monotonic()
    for _ in range(epochs):
        for chunk in shuffle_chunks(train_chunks, rng=train_rng):
            trainer.reset()
            for c in chunk:
                # `train=True` at the trainer level still lets the decoder
                # learn even when region.learning_enabled is False — the
                # region's flag gates its own Hebbian/segment updates, the
                # decoder's flag is separate.
                trainer.step(c, train=True)

    # Eval
    trainer.bpc_probe.reset()
    n_correct = n_chars = 0
    for chunk in test_chunks:
        trainer.reset()
        for c in chunk:
            r = trainer.step(c, train=False)
            n_chars += 1
            if r.top1_correct:
                n_correct += 1
    return {
        "name": "t1_trained" if learning_enabled else "t1_untrained",
        "test_acc": n_correct / max(n_chars, 1),
        "test_bpc": trainer.bpc_probe.bpc,
        "n_test_chars": n_chars,
        "elapsed_s": time.monotonic() - t0,
    }


# ---------------------------------------------------------------------------
# Baseline 2: Markov n-gram
# ---------------------------------------------------------------------------


def run_markov(
    train_chunks: list[str],
    test_chunks: list[str],
    *,
    order: int,
    alphabet: str = DEFAULT_ALPHABET,
) -> dict:
    """Classical char-level Markov model of given order.

    Counts (context, next-char) over the training stream. At eval,
    looks up p(next | context) and computes BPC + top-1 accuracy.
    Falls back to uniform over `alphabet` when the context is unseen.
    """
    alphabet_size = len(alphabet) + 1  # include "unknown" as a valid token
    uniform_bits = math.log2(alphabet_size)

    # Count n-grams. `counts[context][next_char] = freq`.
    counts: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    context_totals: dict[tuple, int] = defaultdict(int)

    t0 = time.monotonic()
    for chunk in train_chunks:
        if len(chunk) < order + 1:
            continue
        # Start-of-chunk padding with empty tuple context so early-chunk
        # steps can still be scored at eval.
        for i in range(len(chunk) - order):
            ctx = tuple(chunk[i : i + order])
            nxt = chunk[i + order]
            counts[ctx][nxt] += 1
            context_totals[ctx] += 1

    n_correct = n_chars = 0
    total_bits = 0.0
    for chunk in test_chunks:
        if len(chunk) < order + 1:
            continue
        for i in range(len(chunk) - order):
            ctx = tuple(chunk[i : i + order])
            actual = chunk[i + order]
            n_chars += 1

            ctx_total = context_totals.get(ctx, 0)
            if ctx_total == 0:
                # Unseen context — uniform distribution.
                total_bits += uniform_bits
                # No top-1 prediction possible; random baseline = 1/alphabet.
                continue

            # Top-1 prediction from counts.
            ctx_counts = counts[ctx]
            top_char = max(ctx_counts, key=lambda c: ctx_counts[c])
            if top_char == actual:
                n_correct += 1

            # BPC for actual char.
            p_actual = ctx_counts.get(actual, 0) / ctx_total
            if p_actual <= 0:
                # Actual char unseen in this context — smooth to 1e-6.
                total_bits += -math.log2(1e-6)
            else:
                total_bits += -math.log2(p_actual)

    return {
        "name": f"markov_{order}",
        "test_acc": n_correct / max(n_chars, 1),
        "test_bpc": total_bits / max(n_chars, 1),
        "n_test_chars": n_chars,
        "n_unique_contexts": len(counts),
        "elapsed_s": time.monotonic() - t0,
    }


# ---------------------------------------------------------------------------
# Baseline 3: linear probe on L2/3
# ---------------------------------------------------------------------------


def run_linear_probe(
    train_chunks: list[str],
    test_chunks: list[str],
    *,
    epochs: int,
    ptd: float,
    ltd: float,
    decay: float,
    lr: float,
    seed: int,
    alphabet: str = DEFAULT_ALPHABET,
) -> dict:
    """Trained T1 + logistic-regression readout (instead of DendriticDecoder).

    Same region training as `run_t1_decoder(learning_enabled=True, ...)`,
    but the readout is a simple linear classifier on the L2/3 active
    binary vector. Isolates "is the representation the bottleneck" from
    "is the decoder the bottleneck."
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError as e:
        raise ImportError(
            "linear probe requires scikit-learn. Install with `uv sync --all-extras`."
        ) from e

    encoder = OneHotCharEncoder(chars=DEFAULT_ALPHABET)
    region = build_region(
        input_dim=encoder.input_dim,
        seed=seed,
        ltd_rate=ltd,
        synapse_decay=decay,
        learning_rate=lr,
        pre_trace_decay=ptd,
    )
    trainer = T1Trainer(region, encoder, decoder=None, bpc_probe=None)

    t0 = time.monotonic()
    # Train the region (same loop as sweep), but collect features from
    # the final epoch only so they reflect the trained representations.
    train_rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        is_last = epoch == epochs - 1
        train_l23: list[np.ndarray] = []
        train_y: list[int] = []
        for chunk in shuffle_chunks(train_chunks, rng=train_rng):
            trainer.reset()
            prev_l23: np.ndarray | None = None
            for c in chunk:
                if is_last and prev_l23 is not None:
                    train_l23.append(prev_l23)
                    train_y.append(ord(c))
                trainer.step(c, train=True)
                prev_l23 = region.l23.active.copy()

    # Fit the probe.
    X = np.array(train_l23, dtype=np.uint8)
    y = np.array(train_y)
    probe = LogisticRegression(max_iter=200, solver="liblinear", multi_class="auto")
    probe.fit(X, y)

    # Eval: collect test features, score.
    test_l23: list[np.ndarray] = []
    test_y: list[int] = []
    for chunk in test_chunks:
        trainer.reset()
        prev_l23 = None
        for c in chunk:
            if prev_l23 is not None:
                test_l23.append(prev_l23)
                test_y.append(ord(c))
            # train=False freezes region learning for eval.
            trainer.step(c, train=False)
            prev_l23 = region.l23.active.copy()

    Xt = np.array(test_l23, dtype=np.uint8)
    yt = np.array(test_y)
    preds = probe.predict(Xt)
    acc = float((preds == yt).mean())
    # BPC via predicted probabilities.
    probs = probe.predict_proba(Xt)
    class_index = {c: i for i, c in enumerate(probe.classes_)}
    bits = 0.0
    for yi, row in zip(yt, probs, strict=True):
        idx = class_index.get(yi)
        p = row[idx] if idx is not None else 1e-6
        bits += -math.log2(max(p, 1e-6))

    return {
        "name": "linear_probe_l23",
        "test_acc": acc,
        "test_bpc": bits / max(len(yt), 1),
        "n_test_chars": len(yt),
        "elapsed_s": time.monotonic() - t0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description="Baselines for T1 next-char substrate")
    p.add_argument("--dataset", default="tinystories")
    p.add_argument("--max-chars", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--ptd", type=float, default=0.50)
    p.add_argument("--ltd", type=float, default=0.20)
    p.add_argument("--decay", type=float, default=0.999)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument(
        "--baselines",
        nargs="+",
        default=["untrained", "trained", "bigram", "trigram", "linear_probe"],
        choices=["untrained", "trained", "bigram", "trigram", "linear_probe"],
    )
    args = p.parse_args()

    chunks = _load_chunks(
        args.dataset, max_chars=args.max_chars, alphabet=DEFAULT_ALPHABET
    )
    train_chunks, test_chunks = split_chunks(chunks, test_frac=0.2, seed=args.seed)
    print(
        f"Dataset: {args.dataset}  {len(train_chunks)}tr {len(test_chunks)}te"
    )
    print(
        f"Config: ep={args.epochs} ptd={args.ptd} ltd={args.ltd} "
        f"decay={args.decay} lr={args.lr} seed={args.seed}"
    )

    results: list[dict] = []
    for name in args.baselines:
        if name == "untrained":
            r = run_t1_decoder(
                train_chunks,
                test_chunks,
                learning_enabled=False,
                epochs=args.epochs,
                ptd=args.ptd,
                ltd=args.ltd,
                decay=args.decay,
                lr=args.lr,
                seed=args.seed,
            )
        elif name == "trained":
            r = run_t1_decoder(
                train_chunks,
                test_chunks,
                learning_enabled=True,
                epochs=args.epochs,
                ptd=args.ptd,
                ltd=args.ltd,
                decay=args.decay,
                lr=args.lr,
                seed=args.seed,
            )
        elif name in ("bigram", "trigram"):
            order = 1 if name == "bigram" else 2
            r = run_markov(train_chunks, test_chunks, order=order)
        elif name == "linear_probe":
            r = run_linear_probe(
                train_chunks,
                test_chunks,
                epochs=args.epochs,
                ptd=args.ptd,
                ltd=args.ltd,
                decay=args.decay,
                lr=args.lr,
                seed=args.seed,
            )
        results.append(r)
        print(
            f"  {r['name']:>22s}: acc={r['test_acc']:.3f} bpc={r['test_bpc']:.2f} "
            f"n={r['n_test_chars']} ({r['elapsed_s']:.1f}s)"
        )

    print("\n=== Summary (sorted by accuracy) ===")
    for r in sorted(results, key=lambda x: -x["test_acc"]):
        print(f"  {r['name']:>22s}: acc={r['test_acc']:.3f} bpc={r['test_bpc']:.2f}")


if __name__ == "__main__":
    main()
