#!/usr/bin/env python3
"""Sweep S1/M1 parameters on TinyDialogues for optimal BPC + turn-taking.

Key bottlenecks identified from initial 10k run:
  - BPC ~5.2 (near random for 56 chars, log2(56)=5.8)
  - M1 capacity: 32 cols k=1 for 56 chars → many chars share columns
  - Dendritic acc ~10% (decoder barely learning)

Explores:
  - S1 n_columns: 32 vs 48 vs 64 (more capacity for 56 chars)
  - S1 k_columns: 2 vs 4 (sparser = better discrimination?)
  - M1 n_columns: 32 vs 48 vs 64 (more output capacity)
  - M1 learning_rate: 0.10 vs 0.15 vs 0.25
  - S1 learning_rate: 0.03 vs 0.05 vs 0.10

Usage:
    uv run experiments/scripts/td_sweep.py
    uv run experiments/scripts/td_sweep.py --quick     # fast subset
    uv run experiments/scripts/td_sweep.py --tokens 20000  # longer runs
"""

import argparse
import time

import step.env  # noqa: F401
from step.config import (
    CortexConfig,
    _default_region2_config,
    make_motor_region,
    make_sensory_region,
)
from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.modulators import SurpriseTracker, ThalamicGate
from step.cortex.topology import Topology
from step.data import prepare_tokens_tinydialogues
from step.encoders.positional import PositionalCharEncoder


def run_config(cfg, tokens, encoder):
    """Run a single configuration and return metrics dict."""
    # S1
    s1_cfg = CortexConfig(
        n_columns=cfg["s1_cols"],
        k_columns=cfg["s1_k"],
        learning_rate=cfg["s1_lr"],
        ltd_rate=0.05,
    )
    region1 = make_sensory_region(
        s1_cfg,
        encoder.input_dim,
        encoder.encoding_width,
    )

    # S2
    r2_cfg = _default_region2_config()
    r2_input_dim = region1.n_l23_total * 4
    region2 = make_sensory_region(r2_cfg, r2_input_dim, seed=123)

    # M1: override cols and learning rate from sweep config
    m1_cfg = CortexConfig(
        n_columns=cfg["m1_cols"],
        k_columns=1,  # WTA proven best
        learning_rate=cfg["m1_lr"],
        ltd_rate=cfg["m1_lr"],  # symmetric LTP/LTD
    )
    motor = make_motor_region(m1_cfg, region1.n_l23_total, seed=456)

    # BG
    bg = BasalGanglia(
        context_dim=region1.n_columns + 1,
        learning_rate=0.1,
        seed=789,
    )

    # Wire
    cortex = Topology(encoder, diagnostics_interval=5000)
    cortex.add_region("S1", region1, entry=True)
    cortex.add_region("S2", region2)
    cortex.connect(
        "S1",
        "S2",
        "feedforward",
        buffer_depth=4,
        burst_gate=True,
        surprise_tracker=SurpriseTracker(),
    )
    cortex.connect("S2", "S1", "apical", thalamic_gate=ThalamicGate())
    cortex.add_region("M1", motor, basal_ganglia=bg)
    cortex.connect("S1", "M1", "feedforward", surprise_tracker=SurpriseTracker())
    # M1→S1 apical only if M1 n_l23_total matches S2's (S2→S1 already set)
    if motor.n_l23_total == region2.n_l23_total:
        cortex.connect("M1", "S1", "apical", thalamic_gate=ThalamicGate())

    t0 = time.time()
    result = cortex.run(tokens, log_interval=100000)
    elapsed = time.time() - t0

    s1 = result.per_region["S1"]
    m1 = result.per_region["M1"]

    # M1 accuracy (last 20%)
    accs = m1.motor_accuracies
    tail = accs[int(len(accs) * 0.8) :] if accs else []
    m1_acc = sum(tail) / len(tail) if tail else 0.0

    # Dendritic accuracy (last 20%)
    den = s1.dendritic_accuracies
    den_tail = den[int(len(den) * 0.8) :] if den else []
    den_acc = sum(den_tail) / len(den_tail) if den_tail else 0.0

    # Turn-taking
    int_rate = m1.turn_interruptions / max(m1.turn_input_steps, 1)
    speak_rate = m1.turn_correct_speak / max(m1.turn_eom_steps, 1)
    unr_rate = m1.turn_unresponsive / max(m1.turn_eom_steps, 1)

    return {
        **cfg,
        "bpc": s1.bpc,
        "bpc_recent": s1.bpc_recent,
        "den_acc": den_acc,
        "m1_acc": m1_acc,
        "int_rate": int_rate,
        "speak_rate": speak_rate,
        "unr_rate": unr_rate,
        "elapsed": elapsed,
    }


def generate_configs(mode):
    """Generate sweep configurations."""
    if mode == "quick":
        return [
            # Baseline
            {
                "s1_cols": 32,
                "s1_k": 4,
                "s1_lr": 0.05,
                "m1_cols": 32,
                "m1_lr": 0.15,
                "label": "baseline",
            },
            # More S1 capacity
            {
                "s1_cols": 64,
                "s1_k": 4,
                "s1_lr": 0.05,
                "m1_cols": 32,
                "m1_lr": 0.15,
                "label": "s1_64col",
            },
            # More M1 capacity
            {
                "s1_cols": 32,
                "s1_k": 4,
                "s1_lr": 0.05,
                "m1_cols": 64,
                "m1_lr": 0.15,
                "label": "m1_64col",
            },
            # Both expanded
            {
                "s1_cols": 64,
                "s1_k": 4,
                "s1_lr": 0.05,
                "m1_cols": 64,
                "m1_lr": 0.15,
                "label": "both_64col",
            },
        ]

    configs = []

    # Phase 1: S1 column count (capacity for 56 chars)
    for s1_cols in [32, 48, 64]:
        configs.append(
            {
                "s1_cols": s1_cols,
                "s1_k": 4,
                "s1_lr": 0.05,
                "m1_cols": 32,
                "m1_lr": 0.15,
                "label": f"s1_{s1_cols}col",
            }
        )

    # Phase 2: S1 sparsity (k_columns)
    for s1_k in [2, 4, 6]:
        configs.append(
            {
                "s1_cols": 32,
                "s1_k": s1_k,
                "s1_lr": 0.05,
                "m1_cols": 32,
                "m1_lr": 0.15,
                "label": f"s1_k{s1_k}",
            }
        )

    # Phase 3: S1 learning rate
    for s1_lr in [0.03, 0.05, 0.10]:
        configs.append(
            {
                "s1_cols": 32,
                "s1_k": 4,
                "s1_lr": s1_lr,
                "m1_cols": 32,
                "m1_lr": 0.15,
                "label": f"s1_lr{s1_lr}",
            }
        )

    # Phase 4: M1 column count
    for m1_cols in [32, 48, 64]:
        configs.append(
            {
                "s1_cols": 32,
                "s1_k": 4,
                "s1_lr": 0.05,
                "m1_cols": m1_cols,
                "m1_lr": 0.15,
                "label": f"m1_{m1_cols}col",
            }
        )

    # Phase 5: M1 learning rate
    for m1_lr in [0.10, 0.15, 0.25]:
        configs.append(
            {
                "s1_cols": 32,
                "s1_k": 4,
                "s1_lr": 0.05,
                "m1_cols": 32,
                "m1_lr": m1_lr,
                "label": f"m1_lr{m1_lr}",
            }
        )

    # Phase 6: Best combos (hypothesized)
    configs.append(
        {
            "s1_cols": 64,
            "s1_k": 4,
            "s1_lr": 0.05,
            "m1_cols": 64,
            "m1_lr": 0.15,
            "label": "both_64col",
        }
    )
    configs.append(
        {
            "s1_cols": 64,
            "s1_k": 4,
            "s1_lr": 0.10,
            "m1_cols": 64,
            "m1_lr": 0.25,
            "label": "both_64_fastlr",
        }
    )

    # Deduplicate
    seen = set()
    unique = []
    for c in configs:
        key = (c["s1_cols"], c["s1_k"], c["s1_lr"], c["m1_cols"], c["m1_lr"])
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def print_results(results):
    """Print results as a formatted table."""
    print(f"\n{'=' * 110}")
    print(
        f"{'label':<20} {'s1col':>5} {'s1k':>3} {'s1lr':>5} "
        f"{'m1col':>5} {'m1lr':>5} "
        f"| {'BPC':>5} {'BPCr':>5} {'den%':>5} "
        f"| {'M1%':>5} {'int%':>5} {'spk%':>5} {'unr%':>5} "
        f"| {'time':>5}"
    )
    print("-" * 110)

    for r in results:
        print(
            f"{r['label']:<20} "
            f"{r['s1_cols']:>5} {r['s1_k']:>3} {r['s1_lr']:>5.3f} "
            f"{r['m1_cols']:>5} {r['m1_lr']:>5.3f} "
            f"| {r['bpc']:>5.2f} {r['bpc_recent']:>5.2f} "
            f"{r['den_acc']:>5.1%} "
            f"| {r['m1_acc']:>5.1%} "
            f"{r['int_rate']:>5.1%} "
            f"{r['speak_rate']:>5.1%} "
            f"{r['unr_rate']:>5.1%} "
            f"| {r['elapsed']:>5.1f}s"
        )

    print("=" * 110)

    # Find best
    best_bpc = min(results, key=lambda r: r["bpc"])
    best_m1 = max(results, key=lambda r: r["m1_acc"])
    print(f"\nBest BPC:  {best_bpc['label']} ({best_bpc['bpc']:.2f})")
    print(f"Best M1:   {best_m1['label']} ({best_m1['m1_acc']:.1%})")


def main():
    parser = argparse.ArgumentParser(
        description="Sweep S1/M1 params on TinyDialogues",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick subset",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=10000,
        help="Chars per run",
    )
    parser.add_argument(
        "--speak-window",
        type=int,
        default=10,
        help="EOM speak window",
    )
    args = parser.parse_args()

    mode = "quick" if args.quick else "full"
    configs = generate_configs(mode)
    print(f"Running {len(configs)} configs ({mode}) on {args.tokens} chars\n")

    # Load tokens once
    tokens = prepare_tokens_tinydialogues(
        args.tokens,
        speak_window=args.speak_window,
    )
    alphabet = sorted({ch for _, ch in tokens if _ >= 0})
    encoder = PositionalCharEncoder("".join(alphabet), max_positions=8)
    print(f"Vocab: {len(alphabet)} chars, encoder dim: {encoder.input_dim}")

    results = []
    for i, cfg in enumerate(configs):
        print(
            f"\n{'=' * 60}\n"
            f"[{i + 1}/{len(configs)}] {cfg['label']} "
            f"(S1:{cfg['s1_cols']}col k={cfg['s1_k']} lr={cfg['s1_lr']} "
            f"M1:{cfg['m1_cols']}col lr={cfg['m1_lr']})\n"
            f"{'=' * 60}"
        )

        r = run_config(cfg, tokens, encoder)
        results.append(r)

        print(
            f"  => BPC={r['bpc']:.2f} den={r['den_acc']:.1%} "
            f"M1={r['m1_acc']:.1%} int={r['int_rate']:.1%} "
            f"spk={r['speak_rate']:.1%} ({r['elapsed']:.1f}s)"
        )

    print_results(results)


if __name__ == "__main__":
    main()
