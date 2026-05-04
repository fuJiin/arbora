#!/usr/bin/env python3
"""ARB-139 plateau analyzer: postprocess accumulator dumps.

Reads data/runs/arb139/plateau/accumulator_n{N}_seed{seed}.pkl and
sdrs_n{N}_seed{seed}.pkl. Computes diagnostics:

  - Histogram of A_w[i] values across all (word, bit) pairs
  - Per-corpus saturation fraction (|sigmoid(A) - 0.5| > 0.45 ⇔ |A| > 2.2)
  - For SimLex word pairs: A values at the top-k bits of those words
  - Cross-corpus: which bits changed between 1M and 5M

Plot output: data/runs/arb139/diagnostics/plateau_*.png
Text summary: data/runs/arb139/diagnostics/plateau_summary.txt
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples.text_exploration.sparse_vs_dense.data import load_simlex

DUMP_DIR = Path("data/runs/arb139/plateau")
OUT_DIR = Path("data/runs/arb139/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_dumps(n_tokens: int, seed: int) -> tuple[dict, dict]:
    acc_path = DUMP_DIR / f"accumulator_n{n_tokens}_seed{seed}.pkl"
    sdr_path = DUMP_DIR / f"sdrs_n{n_tokens}_seed{seed}.pkl"
    with acc_path.open("rb") as f:
        accumulator = pickle.load(f)
    with sdr_path.open("rb") as f:
        sdrs = pickle.load(f)
    return accumulator, sdrs


def analyze_one(n_tokens: int, seed: int) -> dict:
    print(f"\n=== Analyzing n_tokens={n_tokens:,} seed={seed} ===")
    acc, sdrs = load_dumps(n_tokens, seed)

    # Stack all accumulator rows: V × D
    words = sorted(acc.keys())
    A = np.stack([acc[w] for w in words])  # (V, D)
    _V, _D = A.shape

    # Distribution stats
    A_flat = A.flatten()
    print(
        f"  A values: mean={A_flat.mean():+.3f}  std={A_flat.std():.3f}  "
        f"min={A_flat.min():+.3f}  max={A_flat.max():+.3f}"
    )
    print(
        f"  |A| percentiles:  "
        f"p50={np.percentile(np.abs(A_flat), 50):.3f}  "
        f"p90={np.percentile(np.abs(A_flat), 90):.3f}  "
        f"p99={np.percentile(np.abs(A_flat), 99):.3f}"
    )

    # Saturation fraction: a bit is "saturated" if |sigmoid(A) - 0.5| > 0.45,
    # which is |A| > log(0.95/0.05) = 2.94.
    sat_frac = float((np.abs(A_flat) > 2.94).mean())
    print(f"  saturation fraction (|A| > 2.94, σ outside [0.05, 0.95]): {sat_frac:.4f}")
    sat_frac_strict = float((np.abs(A_flat) > 4.6).mean())  # σ outside [0.01, 0.99]
    print(
        f"  strict saturation fraction (|A| > 4.6, σ outside [0.01, 0.99]): {sat_frac_strict:.4f}"
    )

    # For each word, does its top-k contain saturated bits?
    sat_bits_per_word = []
    for w in words:
        sdr = sdrs.get(w)
        if sdr is None:
            continue
        active_bits = np.flatnonzero(sdr)
        if len(active_bits) == 0:
            continue
        a_at_active = acc[w][active_bits]
        sat_at_active = (np.abs(a_at_active) > 2.94).mean()
        sat_bits_per_word.append(sat_at_active)
    sat_bits_arr = np.array(sat_bits_per_word)
    print(
        f"  for top-k bits: mean fraction saturated = {sat_bits_arr.mean():.3f}  "
        f"(p10={np.percentile(sat_bits_arr, 10):.3f} p90={np.percentile(sat_bits_arr, 90):.3f})"
    )

    # SimLex-specific: pull A values at top-k bits for SimLex pair words only
    simlex = load_simlex(vocab=set(words))
    simlex_words = set()
    for a, b, _ in simlex:
        simlex_words.add(a)
        simlex_words.add(b)
    simlex_words &= set(words)

    simlex_a_top = []
    for w in simlex_words:
        sdr = sdrs.get(w)
        if sdr is None:
            continue
        active_bits = np.flatnonzero(sdr)
        a_at_active = acc[w][active_bits]
        simlex_a_top.extend(a_at_active.tolist())
    simlex_a_top = np.asarray(simlex_a_top)
    print(
        f"  SimLex words ({len(simlex_words)}): top-k bit A values  "
        f"mean={simlex_a_top.mean():+.3f}  p50={np.percentile(simlex_a_top, 50):+.3f}  "
        f"p90={np.percentile(simlex_a_top, 90):+.3f}"
    )
    print(
        f"    saturated fraction at SimLex top-k bits: "
        f"{(np.abs(simlex_a_top) > 2.94).mean():.3f}"
    )

    return {
        "n_tokens": n_tokens,
        "A": A,
        "words": words,
        "sdrs": sdrs,
        "sat_frac": sat_frac,
        "sat_frac_strict": sat_frac_strict,
        "sat_bits_arr": sat_bits_arr,
        "simlex_a_top": simlex_a_top,
        "simlex_words": simlex_words,
    }


def cross_compare(r1: dict, r2: dict) -> None:
    """Compare two corpora sizes' accumulator states."""
    print(f"\n=== Cross-compare n={r1['n_tokens']:,} vs n={r2['n_tokens']:,} ===")
    common = sorted(set(r1["words"]) & set(r2["words"]))
    if not common:
        print("  no common words")
        return
    A1 = np.stack([r1["A"][r1["words"].index(w)] for w in common])
    A2 = np.stack([r2["A"][r2["words"].index(w)] for w in common])
    delta = A2 - A1
    print(
        f"  per-bit change |ΔA|:  mean={np.abs(delta).mean():.3f}  "
        f"p50={np.percentile(np.abs(delta), 50):.3f}  "
        f"p90={np.percentile(np.abs(delta), 90):.3f}"
    )

    # How much did top-k change per word?
    bit_changes = []
    for w in common:
        s1 = r1["sdrs"].get(w)
        s2 = r2["sdrs"].get(w)
        if s1 is None or s2 is None:
            continue
        union = (s1 | s2).sum()
        inter = (s1 & s2).sum()
        jaccard = inter / union if union > 0 else 0.0
        bit_changes.append(jaccard)
    bc_arr = np.asarray(bit_changes)
    print(
        f"  top-k jaccard between {r1['n_tokens']:,} and {r2['n_tokens']:,}:  "
        f"mean={bc_arr.mean():.3f}  p10={np.percentile(bc_arr, 10):.3f}  "
        f"frac<0.5: {(bc_arr < 0.5).mean():.3f}"
    )


def make_plots(results: list[dict]) -> None:
    _fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. A_w distribution per corpus size
    ax = axes[0, 0]
    for r in results:
        A_flat = r["A"].flatten()
        ax.hist(
            A_flat,
            bins=80,
            alpha=0.5,
            label=f"n={r['n_tokens']:,}",
            density=True,
        )
    ax.set_xlabel("A_w[i] value")
    ax.set_ylabel("density")
    ax.set_title("Continuous accumulator value distribution")
    ax.axvline(2.94, color="red", lw=0.8, ls="--", alpha=0.5, label="σ=0.95")
    ax.axvline(-2.94, color="red", lw=0.8, ls="--", alpha=0.5)
    ax.legend()
    ax.grid(alpha=0.3)

    # 2. |A| distribution log-scale
    ax = axes[0, 1]
    for r in results:
        A_flat = r["A"].flatten()
        ax.hist(
            np.abs(A_flat),
            bins=80,
            alpha=0.5,
            label=f"n={r['n_tokens']:,}",
            density=True,
        )
    ax.axvline(2.94, color="red", lw=0.8, ls="--", alpha=0.5, label="|A|=2.94 (σ=0.95)")
    ax.axvline(
        4.6, color="darkred", lw=0.8, ls="--", alpha=0.5, label="|A|=4.6 (σ=0.99)"
    )
    ax.set_xlabel("|A_w[i]|")
    ax.set_ylabel("density")
    ax.set_title("Magnitude distribution")
    ax.legend()
    ax.grid(alpha=0.3)

    # 3. Per-word saturation: fraction of top-k bits that are saturated
    ax = axes[1, 0]
    for r in results:
        ax.hist(
            r["sat_bits_arr"],
            bins=20,
            alpha=0.5,
            label=f"n={r['n_tokens']:,}",
            density=True,
        )
    ax.set_xlabel("Fraction of word's top-k bits at saturation")
    ax.set_ylabel("density")
    ax.set_title("Per-word saturation fraction at top-k bits")
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. SimLex-word A values at top-k bits
    ax = axes[1, 1]
    for r in results:
        ax.hist(
            r["simlex_a_top"],
            bins=80,
            alpha=0.5,
            label=f"n={r['n_tokens']:,}",
            density=True,
        )
    ax.axvline(2.94, color="red", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("A_w[i] at top-k bits of SimLex words")
    ax.set_ylabel("density")
    ax.set_title("SimLex words: A values at their active bits")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "plateau_diagnostics.png"
    plt.savefig(out_path, dpi=140)
    print(f"\nWrote {out_path}")


def main() -> None:
    sizes = []
    for p in sorted(DUMP_DIR.glob("accumulator_n*_seed0.pkl")):
        n = int(p.stem.split("_n")[1].split("_seed")[0])
        sizes.append(n)

    if not sizes:
        print(f"No dumps found in {DUMP_DIR}/. Run plateau_dump.py first.")
        return

    print(f"Found dumps for n_tokens: {sizes}")
    results = [analyze_one(n, 0) for n in sizes]

    if len(results) >= 2:
        for i in range(len(results) - 1):
            cross_compare(results[i], results[i + 1])

    make_plots(results)


if __name__ == "__main__":
    main()
