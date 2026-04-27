#!/usr/bin/env python3
"""ARB-139 plateau diagnostic: vary k at eval time only.

Loads the pre-computed continuous accumulator dumps from
`data/runs/arb139/plateau/` and re-evaluates SimLex with different k
values applied at readout time. The training k stays fixed (k=40, what
the trained accumulator was learned with).

What we're testing:
- If SimLex(jaccard) climbs as k_eval grows, the trained accumulator
  has finer structure that the original k=40 readout was throwing away
  → top-k discretization at training/eval is the bottleneck.
- If SimLex(jaccard) is flat across k_eval, the underlying state has
  a structural ceiling → capacity (D or training k) is the limit.

Output: data/runs/arb139/diagnostics/k_eval_sweep.csv
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


def top_k_sdr(row: np.ndarray, k: int) -> np.ndarray:
    n = row.size
    idx = np.argpartition(-row, min(k, n - 1))[:k]
    sdr = np.zeros(n, dtype=bool)
    sdr[idx] = True
    return sdr


def evaluate_with_k(
    accumulator: dict[str, np.ndarray],
    pairs: list[tuple[str, str, float]],
    k: int,
) -> tuple[float, float]:
    """Returns (simlex_jaccard, simlex_cosine_continuous)."""
    from scipy.stats import spearmanr

    # Build SDRs at this k
    sdrs = {w: top_k_sdr(v, k) for w, v in accumulator.items()}

    pred_jac, pred_cos, human = [], [], []
    for a, b, score in pairs:
        if a not in sdrs or b not in sdrs:
            continue
        sa, sb = sdrs[a], sdrs[b]
        pred_jac.append(jaccard_similarity(sa, sb))
        # Cosine on continuous state — k_eval-independent baseline
        va, vb = accumulator[a], accumulator[b]
        pred_cos.append(cosine_similarity(va, vb))
        human.append(score)

    if len(pred_jac) < 2:
        return float("nan"), float("nan")

    rho_jac, _ = spearmanr(pred_jac, human)
    rho_cos, _ = spearmanr(pred_cos, human)
    return float(rho_jac), float(rho_cos)


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
        print(f"  loaded {len(accumulator)} accumulator rows, "
              f"{len(pairs)} SimLex pairs in vocab")

        for k in k_values:
            jac, cos = evaluate_with_k(accumulator, pairs, k)
            rows.append(
                {
                    "n_tokens": n_tokens,
                    "k_eval": k,
                    "simlex_jaccard": jac,
                    "simlex_cosine_continuous": cos,
                    "simlex_n_pairs": len(pairs),
                }
            )
            print(f"  k_eval={k:4d}  simlex(jaccard)={jac:+.4f}  "
                  f"simlex(cosine_cont)={cos:+.4f}")

    out_csv = OUT_DIR / "k_eval_sweep.csv"
    keys = sorted({k for r in rows for k in r})
    with out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
