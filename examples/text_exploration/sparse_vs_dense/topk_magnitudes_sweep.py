#!/usr/bin/env python3
"""ARB-139: Path B — top-k magnitudes readout sweep on existing dumps.

For each accumulator dump and each k_eval value, build TWO sparse outputs:

  binary:  top-k positions set to 1, rest 0  → Jaccard similarity
  magnitudes: top-k positions = A_w value, rest 0 → cosine similarity

Compares whether keeping continuous magnitudes at the top-k positions
helps over binary indicators.

Reads from data/runs/arb139/plateau/accumulator_n*_seed0.pkl.
Writes data/runs/arb139/diagnostics/topk_magnitudes_sweep.csv.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np

from examples.text_exploration.sparse_vs_dense.data import load_simlex
from examples.text_exploration.sparse_vs_dense.evaluation import (
    cosine_similarity,
    jaccard_similarity,
)

DUMP_DIR = Path("data/runs/arb139/plateau")
OUT_DIR = Path("data/runs/arb139/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def topk_binary(row: np.ndarray, k: int) -> np.ndarray:
    n = row.size
    idx = np.argpartition(-row, min(k, n - 1))[:k]
    out = np.zeros(n, dtype=np.bool_)
    out[idx] = True
    return out


def topk_magnitudes(row: np.ndarray, k: int) -> np.ndarray:
    """Top-k positions get the row's continuous value; rest 0."""
    n = row.size
    idx = np.argpartition(-row, min(k, n - 1))[:k]
    out = np.zeros(n, dtype=np.float32)
    out[idx] = row[idx].astype(np.float32)
    return out


def evaluate_with_k(
    accumulator: dict[str, np.ndarray],
    pairs: list[tuple[str, str, float]],
    k: int,
) -> tuple[float, float, float]:
    """Returns (simlex_jaccard_binary, simlex_cosine_topk_mag, simlex_cosine_full)."""
    from scipy.stats import spearmanr

    binary_codes = {w: topk_binary(v, k) for w, v in accumulator.items()}
    mag_codes = {w: topk_magnitudes(v, k) for w, v in accumulator.items()}

    pred_jac, pred_mag, pred_full, human = [], [], [], []
    for a, b, score in pairs:
        if a not in accumulator or b not in accumulator:
            continue
        pred_jac.append(jaccard_similarity(binary_codes[a], binary_codes[b]))
        pred_mag.append(cosine_similarity(mag_codes[a], mag_codes[b]))
        pred_full.append(cosine_similarity(accumulator[a], accumulator[b]))
        human.append(score)

    if len(pred_jac) < 2:
        return float("nan"), float("nan"), float("nan")

    rho_jac, _ = spearmanr(pred_jac, human)
    rho_mag, _ = spearmanr(pred_mag, human)
    rho_full, _ = spearmanr(pred_full, human)
    return float(rho_jac), float(rho_mag), float(rho_full)


def main() -> None:
    rows: list[dict] = []
    k_values = [10, 20, 40, 60, 80, 120, 160, 240, 320, 512]

    for acc_path in sorted(DUMP_DIR.glob("accumulator_n*_seed0.pkl")):
        n_tokens = int(acc_path.stem.split("_n")[1].split("_seed")[0])
        print(f"\n=== n_tokens={n_tokens:,} ===")
        with acc_path.open("rb") as f:
            accumulator = pickle.load(f)
        vocab_set = set(accumulator.keys())
        pairs = load_simlex(vocab=vocab_set)
        print(f"  loaded {len(accumulator)} rows, {len(pairs)} SimLex pairs")

        for k in k_values:
            jac, mag, full = evaluate_with_k(accumulator, pairs, k)
            rows.append(
                {
                    "n_tokens": n_tokens,
                    "k_eval": k,
                    "simlex_jaccard_binary": jac,
                    "simlex_cosine_topk_mag": mag,
                    "simlex_cosine_full": full,
                    "n_pairs": len(pairs),
                }
            )
            print(
                f"  k={k:4d}  jaccard_binary={jac:+.4f}  "
                f"cosine_topk_mag={mag:+.4f}  cosine_full={full:+.4f}"
            )

    out_csv = OUT_DIR / "topk_magnitudes_sweep.csv"
    keys = sorted({k for r in rows for k in r})
    with out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
