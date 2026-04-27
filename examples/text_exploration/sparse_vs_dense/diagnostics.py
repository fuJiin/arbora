#!/usr/bin/env python3
"""ARB-139: postprocess diagnostics on cached embedding dumps.

Three diagnostics, all from existing dumps (no new training):

1. **Per-bit usage histogram** — for each (model, n_tokens), count how many
   words have each bit position in their top-k. If heavy-tailed, the
   representation has "hot bits" used by many words (vocabulary mixing).
   If uniform, bits are used roughly equally (fair use).

2. **Per-word code stability across sizes** — for each word in vocabs of
   consecutive training sizes, compute Jaccard of its sparse code at size
   N and size N+1. High → codes are stable, troughs are evaluation noise.
   Low → codes shift substantively across sizes.

3. **Pairwise overlap distribution** — at each (model, n_tokens), compute
   Jaccard distribution over a sample of word pairs. Should peak near 0
   for unrelated words; the width tells us discriminability.

Output: data/runs/arb139/diagnostics/*.csv + summary.txt
"""

from __future__ import annotations

import csv
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

DUMP_DIR = Path("data/runs/arb139/dumps")
OUT_DIR = Path("data/runs/arb139/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_dump_filename(p: Path) -> tuple[str, int, int] | None:
    """Match e.g. 'sparse_skipgram_hebbian_n1000000_s0.pkl' → (model, n, seed)."""
    m = re.match(r"^(.+?)_n(\d+)_s(\d+)\.pkl$", p.name)
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def is_sparse_dump(payload: dict[str, np.ndarray]) -> bool:
    """Sparse dumps store boolean ndarrays."""
    if not payload:
        return False
    first = next(iter(payload.values()))
    return first.dtype == np.bool_


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    union = (a | b).sum()
    if union == 0:
        return 0.0
    return float((a & b).sum()) / float(union)


def diag_per_bit_usage() -> None:
    """For each sparse dump, histogram of how many words use each bit."""
    rows: list[dict] = []
    histos: dict[tuple[str, int, int], np.ndarray] = {}
    for p in sorted(DUMP_DIR.glob("*.pkl")):
        meta = parse_dump_filename(p)
        if meta is None:
            continue
        model, n_tokens, seed = meta
        with p.open("rb") as f:
            payload = pickle.load(f)
        if not is_sparse_dump(payload):
            continue
        # Build the V × D matrix of bool codes.
        V = len(payload)
        if V == 0:
            continue
        first = next(iter(payload.values()))
        D = first.size
        usage = np.zeros(D, dtype=np.int64)
        for code in payload.values():
            usage += code.astype(np.int64)
        histos[(model, n_tokens, seed)] = usage
        # Summary stats.
        # - mean: V × k_active / D (uniform expectation)
        # - max: how concentrated the most-used bit is
        # - tail_5pct: mean usage of top-5% bits / mean usage overall (>1 = hot bits)
        sorted_usage = np.sort(usage)[::-1]
        top_5pct = int(D * 0.05)
        top_5pct_mean = sorted_usage[:top_5pct].mean()
        tail_concentration = top_5pct_mean / max(usage.mean(), 1e-9)
        rows.append(
            {
                "model": model,
                "n_tokens": n_tokens,
                "seed": seed,
                "vocab_size": V,
                "n_dims": D,
                "mean_words_per_bit": float(usage.mean()),
                "max_words_per_bit": int(usage.max()),
                "min_words_per_bit": int(usage.min()),
                "top5pct_concentration": float(tail_concentration),
                "n_zero_bits": int((usage == 0).sum()),
                "n_full_bits": int((usage == V).sum()),
            }
        )

    out_csv = OUT_DIR / "per_bit_usage.csv"
    if rows:
        keys = sorted({k for r in rows for k in r})
        with out_csv.open("w") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {out_csv} ({len(rows)} rows)")
    return histos


def diag_code_stability_across_sizes() -> None:
    """For each (model, seed), compute Jaccard of each word's code between
    consecutive corpus sizes. If shared word's code at 1M differs sharply
    from its code at 500k, the representation is unstable across sizes.
    """
    # Group dumps by (model, seed)
    by_key: dict[tuple[str, int], list[tuple[int, dict]]] = defaultdict(list)
    for p in sorted(DUMP_DIR.glob("*.pkl")):
        meta = parse_dump_filename(p)
        if meta is None:
            continue
        model, n_tokens, seed = meta
        with p.open("rb") as f:
            payload = pickle.load(f)
        if not is_sparse_dump(payload):
            continue
        by_key[(model, seed)].append((n_tokens, payload))

    rows: list[dict] = []
    for (model, seed), entries in by_key.items():
        entries.sort(key=lambda e: e[0])
        for i in range(len(entries) - 1):
            n_a, codes_a = entries[i]
            n_b, codes_b = entries[i + 1]
            common = set(codes_a) & set(codes_b)
            if not common:
                continue
            sims = [jaccard(codes_a[w], codes_b[w]) for w in common]
            sims_arr = np.asarray(sims)
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "n_a": n_a,
                    "n_b": n_b,
                    "n_words": len(common),
                    "jaccard_mean": float(sims_arr.mean()),
                    "jaccard_std": float(sims_arr.std()),
                    "jaccard_p10": float(np.percentile(sims_arr, 10)),
                    "jaccard_p50": float(np.percentile(sims_arr, 50)),
                    "jaccard_p90": float(np.percentile(sims_arr, 90)),
                    "frac_high_stability": float((sims_arr > 0.7).mean()),
                    "frac_low_stability": float((sims_arr < 0.2).mean()),
                }
            )

    out_csv = OUT_DIR / "code_stability_across_sizes.csv"
    if rows:
        keys = sorted({k for r in rows for k in r})
        with out_csv.open("w") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {out_csv} ({len(rows)} rows)")


def diag_pairwise_overlap_distribution(sample_size: int = 1000) -> None:
    """For each sparse dump, distribution of pairwise Jaccard similarities
    across a random sample of word pairs. Tells us:
    - Mean: are codes well-separated on average?
    - Std: how varied are similarities?
    - Tail (95th percentile): what does the most-similar pair look like?
    """
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for p in sorted(DUMP_DIR.glob("*.pkl")):
        meta = parse_dump_filename(p)
        if meta is None:
            continue
        model, n_tokens, seed = meta
        with p.open("rb") as f:
            payload = pickle.load(f)
        if not is_sparse_dump(payload):
            continue
        words = list(payload.keys())
        V = len(words)
        if V < 100:
            continue
        # Sample word pairs.
        n_pairs = min(sample_size, V * (V - 1) // 2)
        pairs_a = rng.integers(0, V, n_pairs)
        pairs_b = rng.integers(0, V, n_pairs)
        # Avoid self-pairs.
        mask = pairs_a != pairs_b
        pairs_a = pairs_a[mask]
        pairs_b = pairs_b[mask]
        sims = np.array(
            [jaccard(payload[words[a]], payload[words[b]])
             for a, b in zip(pairs_a, pairs_b)]
        )
        rows.append(
            {
                "model": model,
                "n_tokens": n_tokens,
                "seed": seed,
                "n_pairs_sampled": len(sims),
                "jaccard_mean": float(sims.mean()),
                "jaccard_std": float(sims.std()),
                "jaccard_p50": float(np.percentile(sims, 50)),
                "jaccard_p95": float(np.percentile(sims, 95)),
                "jaccard_p99": float(np.percentile(sims, 99)),
                "frac_zero_overlap": float((sims == 0).mean()),
                "frac_high_overlap": float((sims > 0.5).mean()),
            }
        )

    out_csv = OUT_DIR / "pairwise_overlap.csv"
    if rows:
        keys = sorted({k for r in rows for k in r})
        with out_csv.open("w") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {out_csv} ({len(rows)} rows)")


def main() -> None:
    print("=== Per-bit usage histograms ===")
    diag_per_bit_usage()
    print("\n=== Code stability across sizes ===")
    diag_code_stability_across_sizes()
    print("\n=== Pairwise overlap distribution ===")
    diag_pairwise_overlap_distribution()
    print(f"\nAll diagnostics written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
