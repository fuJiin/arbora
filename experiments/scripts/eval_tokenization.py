#!/usr/bin/env python3
"""Compare tokenization and encoding strategies on representation quality.

Runs the same cortex config with multiple tokenization/encoding combos:
1. BPE + CharbitEncoder (808-dim) — baseline
2. Char + CharbitEncoder (808-dim, mostly zeros) — reference
3. Char + OneHotEncoder (compact, ~33-dim) — minimal
4. Char + PositionalCharEncoder (~272-dim) — structured

Evaluates: dendritic decoder accuracy, burst rate, dead columns, learning curve.

Usage: uv run experiments/scripts/eval_tokenization.py [--tokens 20000]
"""

import argparse
import string
import time
from collections import Counter

import numpy as np

import step.env  # noqa: F401
from step.config import CortexConfig
from step.cortex.sensory import SensoryRegion
from step.data import STORY_BOUNDARY, prepare_tokens, prepare_tokens_charlevel
from step.decoders.dendritic import DendriticDecoder
from step.encoders.charbit import CharbitEncoder
from step.encoders.onehot import OneHotCharEncoder
from step.encoders.positional import PositionalCharEncoder

CHARS = string.printable
CHAR_LENGTH = 8
CHAR_WIDTH = len(CHARS) + 1


def _make_region(cfg: CortexConfig, input_dim: int, encoding_width: int):
    """Create a SensoryRegion with the given input configuration."""
    return SensoryRegion(
        input_dim=input_dim,
        encoding_width=encoding_width,
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


def run_eval(
    tokens: list[tuple[int, str]],
    cfg: CortexConfig,
    label: str,
    log_interval: int = 5000,
    *,
    encoder=None,
    input_dim: int = 0,
    encoding_width: int = 0,
) -> dict:
    """Run cortex + dendritic decoder, return comprehensive metrics."""
    if encoder is None:
        encoder = CharbitEncoder(
            length=CHAR_LENGTH,
            width=CHAR_WIDTH,
            chars=CHARS,
        )
        input_dim = CHAR_LENGTH * CHAR_WIDTH
        encoding_width = CHAR_WIDTH
    region = _make_region(cfg, input_dim, encoding_width)
    decoder = DendriticDecoder(source_dim=region.n_l23_total)

    prev_l23: np.ndarray | None = None
    prev_was_boundary = False

    top1_hits: list[bool] = []
    top5_hits: list[bool] = []
    burst_per_token: dict[int, list[float]] = {}
    col_activation_counts = np.zeros(cfg.n_columns)

    start = time.monotonic()
    for t, (token_id, token_str) in enumerate(tokens):
        if token_id == STORY_BOUNDARY:
            region.reset_working_memory()
            if hasattr(encoder, "reset"):
                encoder.reset()
            prev_l23 = None
            prev_was_boundary = True
            continue

        # Decode before processing
        if prev_l23 is not None and not prev_was_boundary:
            preds = decoder.decode(prev_l23, k=5)
            top1_hits.append(len(preds) > 0 and preds[0] == token_id)
            top5_hits.append(token_id in preds)

        # Process
        encoding = encoder.encode(token_str)
        region.process(encoding)

        # Track burst rate per token
        n_active = int(region.active_columns.sum())
        if n_active > 0:
            rate = float(region.bursting_columns.sum()) / n_active
            burst_per_token.setdefault(token_id, []).append(rate)

        # Track column usage
        col_activation_counts += region.active_columns.astype(np.float64)

        # Teach decoder
        if prev_l23 is not None and not prev_was_boundary:
            decoder.observe(token_id, prev_l23)

        prev_l23 = region.l23.active.copy()
        prev_was_boundary = False

        if t > 0 and t % log_interval == 0:
            elapsed = time.monotonic() - start
            n = len(top1_hits)
            # Recent window
            recent = min(log_interval, n)
            w1 = sum(top1_hits[-recent:]) / recent if recent else 0
            w5 = sum(top5_hits[-recent:]) / recent if recent else 0
            print(
                f"  [{label}] t={t:>6d}  "
                f"top1={w1:.3f} top5={w5:.3f}  "
                f"vocab={decoder.n_tokens}  ({elapsed:.1f}s)"
            )

    elapsed = time.monotonic() - start
    n = len(top1_hits)

    # Burst rate stats
    all_burst = [r for rates in burst_per_token.values() for r in rates]
    mean_burst = float(np.mean(all_burst)) if all_burst else 1.0

    # Per-token burst (min 5 occurrences)
    token_burst_means = {
        tid: float(np.mean(rates))
        for tid, rates in burst_per_token.items()
        if len(rates) >= 5
    }

    # Column utilization
    n_dead = int((col_activation_counts == 0).sum())
    rare_mask = (col_activation_counts > 0) & (
        col_activation_counts < len(tokens) * 0.01
    )
    n_rare = int(rare_mask.sum())

    # Learning curve: quarters
    q = n // 4
    quarter_acc = []
    for i in range(4):
        s = i * q
        e = s + q if i < 3 else n
        quarter_acc.append(sum(top1_hits[s:e]) / (e - s) if e > s else 0)

    unique_tokens = len({tid for tid, _ in tokens if tid != STORY_BOUNDARY})

    return {
        "label": label,
        "top1": sum(top1_hits) / n if n else 0,
        "top5": sum(top5_hits) / n if n else 0,
        "n_samples": n,
        "n_unique": unique_tokens,
        "vocab_learned": decoder.n_tokens,
        "mean_burst": mean_burst,
        "n_dead_cols": n_dead,
        "n_rare_cols": n_rare,
        "elapsed": elapsed,
        "quarter_acc": quarter_acc,
        "token_burst_means": token_burst_means,
        "majority_frac": Counter(
            tid for tid, _ in tokens if tid != STORY_BOUNDARY
        ).most_common(1)[0][1]
        / sum(1 for tid, _ in tokens if tid != STORY_BOUNDARY),
    }


def print_results(bpe: dict, char: dict):
    """Print comparison table."""
    print(f"\n{'=' * 65}")
    print("TOKENIZATION COMPARISON")
    print(f"{'=' * 65}")

    header = f"  {'Metric':<30s}  {'BPE':>12s}  {'Char-level':>12s}"
    print(header)
    print(f"  {'-' * 30}  {'-' * 12}  {'-' * 12}")

    bpe_maj = bpe["majority_frac"]
    char_maj = char["majority_frac"]
    bpe_lift_maj = f"{bpe['top1'] / bpe_maj:.2f}x"
    char_lift_maj = f"{char['top1'] / char_maj:.2f}x"
    bpe_lift_uni = f"{bpe['top1'] * bpe['n_unique']:.0f}x"
    char_lift_uni = f"{char['top1'] * char['n_unique']:.0f}x"

    rows = [
        ("Vocabulary size", f"{bpe['n_unique']}", f"{char['n_unique']}"),
        ("Samples", f"{bpe['n_samples']:,}", f"{char['n_samples']:,}"),
        ("", "", ""),
        ("Dendritic top-1", f"{bpe['top1']:.4f}", f"{char['top1']:.4f}"),
        ("Dendritic top-5", f"{bpe['top5']:.4f}", f"{char['top5']:.4f}"),
        ("Majority baseline", f"{bpe_maj:.4f}", f"{char_maj:.4f}"),
        ("Uniform top-1", f"{1 / bpe['n_unique']:.4f}", f"{1 / char['n_unique']:.4f}"),
        ("", "", ""),
        ("Lift over majority", bpe_lift_maj, char_lift_maj),
        ("Lift over uniform", bpe_lift_uni, char_lift_uni),
        ("", "", ""),
        ("Mean burst rate", f"{bpe['mean_burst']:.4f}", f"{char['mean_burst']:.4f}"),
        ("Dead columns", f"{bpe['n_dead_cols']}", f"{char['n_dead_cols']}"),
        ("Rare columns", f"{bpe['n_rare_cols']}", f"{char['n_rare_cols']}"),
    ]

    for label, v1, v2 in rows:
        if not label:
            print()
        else:
            print(f"  {label:<30s}  {v1:>12s}  {v2:>12s}")

    # Learning curves
    print("\n  Learning curve (top-1 by quarter):")
    bpe_q = bpe["quarter_acc"]
    char_q = char["quarter_acc"]
    for i, (b, c) in enumerate(zip(bpe_q, char_q, strict=True)):
        qlabel = ["Q1 (early)", "Q2", "Q3", "Q4 (late)"][i]
        print(f"    {qlabel:<12s}  BPE={b:.4f}  Char={c:.4f}")

    # Best/worst predicted chars
    print("\n  Best-predicted characters (lowest burst rate):")
    sorted_chars = sorted(
        char["token_burst_means"].items(),
        key=lambda x: x[1],
    )
    for tid, rate in sorted_chars[:10]:
        ch = repr(chr(tid))
        print(f"    {ch:>6s}: burst={rate:.3f}")

    print("\n  Worst-predicted characters:")
    for tid, rate in sorted_chars[-10:][::-1]:
        ch = repr(chr(tid))
        print(f"    {ch:>6s}: burst={rate:.3f}")

    print(f"\n{'=' * 65}")

    # Verdict
    print("\nVerdict:")
    if char["top1"] > char["majority_frac"] * 1.2:
        print("  Char-level BEATS majority baseline -> architecture works,")
        print("  BPE failure is a vocab/capacity mismatch, not a fundamental problem.")
    elif char["top1"] > bpe["top1"] * 1.5:
        print("  Char-level significantly better than BPE but below majority.")
        print("  Architecture shows promise, needs more capacity.")
    else:
        print("  Char-level no better than BPE.")
        print("  Bottleneck is not just vocabulary size.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=20000)
    parser.add_argument("--log-interval", type=int, default=5000)
    args = parser.parse_args()

    cfg = CortexConfig()
    print(f"Config: {cfg.n_columns} cols, {cfg.n_l4} L4, {cfg.n_l23} L2/3")
    print(f"L2/3 dim: {cfg.n_columns * cfg.n_l23}\n")

    # --- BPE (CharbitEncoder, 808-dim) ---
    print("=== BPE Tokenization (CharbitEncoder 808-dim) ===")
    bpe_tokens = prepare_tokens(args.tokens)
    bpe_results = run_eval(
        bpe_tokens,
        cfg,
        "BPE",
        log_interval=args.log_interval,
    )

    # --- Char-level with CharbitEncoder (808-dim, mostly zeros) ---
    print("\n=== Char-Level + CharbitEncoder (808-dim) ===")
    char_tokens = prepare_tokens_charlevel(args.tokens)
    charbit_results = run_eval(
        char_tokens,
        cfg,
        "Char808",
        log_interval=args.log_interval,
    )

    # --- Char-level with OneHotCharEncoder (compact) ---
    # Build alphabet from actual data
    alphabet = sorted({ch for _, ch in char_tokens if _ != STORY_BOUNDARY})
    onehot = OneHotCharEncoder("".join(alphabet))
    # Char-level config: halve ltd_rate (local_scale changes with small input)
    from dataclasses import replace

    char_cfg = replace(cfg, ltd_rate=cfg.ltd_rate / 2)
    print(f"\n=== Char-Level + OneHotEncoder ({onehot.input_dim}-dim) ===")
    compact_results = run_eval(
        char_tokens,
        char_cfg,
        "Compact",
        log_interval=args.log_interval,
        encoder=onehot,
        input_dim=onehot.input_dim,
        encoding_width=0,  # triggers full connectivity for small input
    )

    # --- Char-level with PositionalCharEncoder (structured) ---
    pos_enc = PositionalCharEncoder("".join(alphabet), max_positions=8)
    print(f"\n=== Char-Level + PositionalEncoder ({pos_enc.input_dim}-dim) ===")
    positional_results = run_eval(
        char_tokens,
        cfg,
        "Positional",
        log_interval=args.log_interval,
        encoder=pos_enc,
        input_dim=pos_enc.input_dim,
        encoding_width=pos_enc.encoding_width,
    )

    # Print comparison table
    print_results(bpe_results, positional_results)

    # Reference: all char-level encoders
    print("\n  All char-level encoders:")
    for r in [charbit_results, compact_results, positional_results]:
        q = r["quarter_acc"]
        print(
            f"   {r['label']:>10s}: "
            f"top1={r['top1']:.4f} top5={r['top5']:.4f} "
            f"burst={r['mean_burst']:.4f} "
            f"dead={r['n_dead_cols']} "
            f"curve={q[0]:.3f}->{q[3]:.3f}"
        )


if __name__ == "__main__":
    main()
