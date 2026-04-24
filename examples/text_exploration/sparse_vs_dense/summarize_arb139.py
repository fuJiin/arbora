#!/usr/bin/env python3
"""Summarize ARB-139 sweep CSVs into a comparison table.

Reads any CSV produced by `compare.py --csv` and aggregates rows by
(model, n_tokens), reporting mean across seeds. Designed to work on
partial results — if a phase didn't complete, it shows what's there.

Usage:
    uv run python -m examples.text_exploration.sparse_vs_dense.summarize_arb139 \\
        data/runs/arb139/light_seed0.csv data/runs/arb139/t1_seed0.csv \\
        data/runs/arb139/variance_seed1.csv data/runs/arb139/variance_seed2.csv

If no args given, defaults to all CSVs under data/runs/arb139/.
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

DEFAULT_DIR = Path("data/runs/arb139")

HEADLINE_METRICS = [
    ("simlex_spearman", "simlex"),
    ("analogy_top1", "analogy"),
    ("cap_mean_sim", "coll_sim"),
    ("cap_collision_frac", "coll_frac"),
    ("cap_eff_dim", "eff_dim"),
    ("bundling_capacity", "bundle_k*"),
    ("bundling_margin_at_k8", "bundle@k8"),
    ("bundling_margin_at_k32", "bundle@k32"),
    ("partial_cue_retention", "pc_ret"),
    ("elapsed_s", "train_s"),
]


def load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        if not p.exists():
            print(f"[skip] {p} (missing)", file=sys.stderr)
            continue
        with p.open() as f:
            for row in csv.DictReader(f):
                rows.append(row)
    return rows


def coerce(v: str) -> float | None:
    if v == "" or v is None:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def aggregate(rows: list[dict]) -> dict:
    """Group by (model, n_tokens), compute mean +/- std across seeds."""
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["model"], int(r["n_tokens"]))
        grouped[key].append(r)
    agg: dict[tuple[str, int], dict[str, tuple[float, float, int]]] = {}
    for key, group in grouped.items():
        out: dict[str, tuple[float, float, int]] = {}
        for col, _label in HEADLINE_METRICS:
            vals = [coerce(r.get(col, "")) for r in group]
            vals_clean = [v for v in vals if v is not None]
            if vals_clean:
                m = mean(vals_clean)
                s = stdev(vals_clean) if len(vals_clean) > 1 else 0.0
                out[col] = (m, s, len(vals_clean))
            else:
                out[col] = (float("nan"), 0.0, 0)
        agg[key] = out
    return agg


def fmt_cell(stat: tuple[float, float, int], decimals: int = 3) -> str:
    m, s, n = stat
    if n == 0 or m != m:  # nan
        return " - "
    if n > 1:
        return f"{m:>.{decimals}f}±{s:.{decimals}f}"
    return f"{m:>.{decimals}f}"


def print_table(agg: dict) -> None:
    models_order = [
        "word2vec",
        "random_indexing",
        "brown_cluster",
        "sparse_skipgram_hebbian",
        "t1_sparse",
    ]
    token_counts = sorted({k[1] for k in agg.keys()})
    models_present = [m for m in models_order if any(k[0] == m for k in agg.keys())]

    for col, label in HEADLINE_METRICS:
        print(f"\n=== {label} ({col}) ===")
        # Header
        header = f"{'model':>17s} | " + " | ".join(
            f"{n:>11,}" for n in token_counts
        )
        print(header)
        print("-" * len(header))
        for m in models_present:
            cells: list[str] = []
            for n in token_counts:
                stat = agg.get((m, n), {}).get(col)
                if stat is None:
                    cells.append(" - ")
                else:
                    decimals = 0 if col in ("bundling_capacity",) else (
                        1 if col in ("cap_eff_dim", "elapsed_s") else 3
                    )
                    cells.append(fmt_cell(stat, decimals))
            row = f"{m:>17s} | " + " | ".join(f"{c:>11s}" for c in cells)
            print(row)


def main() -> None:
    args = [Path(a) for a in sys.argv[1:]]
    if not args:
        args = sorted(DEFAULT_DIR.glob("*.csv"))
        if not args:
            print(f"No CSVs in {DEFAULT_DIR}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading: {[str(p) for p in args]}")
    rows = load_rows(args)
    print(f"Loaded {len(rows)} rows.")
    if not rows:
        return
    agg = aggregate(rows)
    print_table(agg)


if __name__ == "__main__":
    main()
