#!/usr/bin/env python3
"""ARB-139: plot within-run SimLex curves at different EMA alphas.

Reads all data/runs/arb139/within_run*.csv and overlays them on one
plot. Saves PNG to data/runs/arb139/diagnostics/within_run_curves.png.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
RUN_DIR = ROOT / "data" / "runs" / "arb139"
OUT_DIR = RUN_DIR / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_curve(csv_path: Path) -> list[tuple[int, float]]:
    out: list[tuple[int, float]] = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            n = int(row["n_tokens"])
            sim = float(row["simlex_spearman"])
            out.append((n, sim))
    out.sort()
    return out


def parse_label_from_name(name: str) -> str:
    """Build a human-readable label from the CSV filename."""
    m = re.match(r"within_run_alpha_(.+)\.csv", name)
    if m:
        return f"α={m.group(1).replace('_', '.')}"
    if name == "within_run_sigmoid.csv":
        return "sigmoid-bounded (no EMA)"
    return "α=0.01"  # the original within_run_simlex.csv


def main() -> None:
    csvs = sorted(RUN_DIR.glob("within_run*.csv"))
    if not csvs:
        print("No within_run*.csv found.")
        return

    fig, ax = plt.subplots(figsize=(11, 6))
    for p in csvs:
        label = parse_label_from_name(p.name)
        curve = load_curve(p)
        if not curve:
            continue
        xs = [n for n, _ in curve]
        ys = [s for _, s in curve]
        # Highlight the sigmoid-bounded curve with a thicker line.
        is_sigmoid = "sigmoid" in label
        ax.plot(
            xs, ys, marker="o", lw=3 if is_sigmoid else 1.5,
            alpha=0.95 if is_sigmoid else 0.7,
            label=f"{label}  (n={len(curve)} ckpts)",
        )

    ax.axhline(0, color="black", lw=0.5)
    ax.set_xlabel("training tokens consumed")
    ax.set_ylabel("SimLex-999 Spearman correlation")
    ax.set_title(
        "SSH within-run SimLex trajectory at different EMA α values\n"
        "(text8, vocab=5000, single-table, modulated, 50k checkpoint stride)"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()

    out_path = OUT_DIR / "within_run_curves.png"
    plt.savefig(out_path, dpi=140)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
