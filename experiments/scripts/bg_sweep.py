#!/usr/bin/env python3
"""Sweep BasalGanglia parameters across datasets and training lengths.

Explores:
  - BG learning rate: how fast should the gate learn?
  - segment_length: frequency of synthetic turn boundaries
  - speak_window: how long M1 gets to practice speaking
  - Dataset: babylm (synthetic boundaries) vs tinystories (natural stories)
  - Token count: single story vs multiple stories

Usage:
    uv run experiments/scripts/bg_sweep.py
    uv run experiments/scripts/bg_sweep.py --quick     # fast subset
"""

import argparse
import time

import step.env  # noqa: F401
from step.config import (
    CortexConfig,
    _default_motor_config,
    _default_region2_config,
    make_motor_region,
    make_sensory_region,
)
from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.modulators import SurpriseTracker, ThalamicGate
from step.cortex.topology import ConnectionRole, Topology
from step.data import inject_eom_tokens, prepare_tokens_charlevel
from step.encoders.positional import PositionalCharEncoder

CONFIGS = {
    "quick": [
        # Quick sanity check: 3 configs
        {
            "bg_lr": 0.01,
            "tokens": 5000,
            "segment": 100,
            "window": 10,
            "dataset": "babylm",
        },
        {
            "bg_lr": 0.1,
            "tokens": 5000,
            "segment": 100,
            "window": 10,
            "dataset": "babylm",
        },
        {
            "bg_lr": 0.1,
            "tokens": 5000,
            "segment": 0,
            "window": 10,
            "dataset": "tinystories",
        },
    ],
    "full": None,  # generated below
}


def generate_full_configs():
    """Generate full sweep grid."""
    configs = []

    # Phase 1: BG learning rate sweep (babylm, synthetic boundaries)
    for bg_lr in [0.01, 0.05, 0.1, 0.2]:
        configs.append(
            {
                "bg_lr": bg_lr,
                "tokens": 10000,
                "segment": 100,
                "window": 10,
                "dataset": "babylm",
            }
        )

    # Phase 2: segment_length sweep (best lr from phase 1, but run all)
    for segment in [50, 100, 200, 500]:
        configs.append(
            {
                "bg_lr": 0.1,
                "tokens": 10000,
                "segment": segment,
                "window": 10,
                "dataset": "babylm",
            }
        )

    # Phase 3: speak_window sweep
    for window in [5, 10, 20]:
        configs.append(
            {
                "bg_lr": 0.1,
                "tokens": 10000,
                "segment": 100,
                "window": window,
                "dataset": "babylm",
            }
        )

    # Phase 4: TinyStories with natural boundaries
    for bg_lr in [0.05, 0.1, 0.2]:
        for n_tokens in [5000, 20000]:
            configs.append(
                {
                    "bg_lr": bg_lr,
                    "tokens": n_tokens,
                    "segment": 0,  # natural boundaries only
                    "window": 10,
                    "dataset": "tinystories",
                }
            )

    # Phase 5: Scale test — longer runs
    for n_tokens in [20000, 50000]:
        configs.append(
            {
                "bg_lr": 0.1,
                "tokens": n_tokens,
                "segment": 100,
                "window": 10,
                "dataset": "babylm",
            }
        )

    # Deduplicate
    seen = set()
    unique = []
    for c in configs:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


CONFIGS["full"] = generate_full_configs()


def run_config(cfg):
    """Run a single configuration and return metrics dict."""
    tokens = prepare_tokens_charlevel(cfg["tokens"], dataset=cfg["dataset"])
    tokens = inject_eom_tokens(
        tokens,
        segment_length=cfg["segment"],
        speak_window=cfg["window"],
    )

    alphabet = sorted({ch for _, ch in tokens if _ >= 0})
    encoder = PositionalCharEncoder("".join(alphabet), max_positions=8)

    cortex_cfg = CortexConfig(ltd_rate=0.05)
    region1 = make_sensory_region(
        cortex_cfg,
        encoder.input_dim,
        encoder.encoding_width,
    )

    r2_cfg = _default_region2_config()
    r2_input_dim = region1.n_l23_total * 4  # buffer_depth=4
    region2 = make_sensory_region(r2_cfg, r2_input_dim, seed=123)

    m1_cfg = _default_motor_config()
    motor = make_motor_region(m1_cfg, region1.n_l23_total, seed=456)

    bg = BasalGanglia(
        context_dim=region1.n_columns + 1,  # per-col burst + overall burst frac
        learning_rate=cfg["bg_lr"],
        seed=789,
    )

    surprise = SurpriseTracker()
    cortex = Topology(encoder, diagnostics_interval=1000)
    cortex.add_region("S1", region1, entry=True)
    cortex.add_region("S2", region2)
    cortex.connect(
        "S1",
        "S2",
        ConnectionRole.FEEDFORWARD,
        buffer_depth=4,
        burst_gate=True,
        surprise_tracker=surprise,
    )
    cortex.connect("S2", "S1", ConnectionRole.APICAL, thalamic_gate=ThalamicGate())
    cortex.add_region("M1", motor, basal_ganglia=bg)
    cortex.connect(
        "S1", "M1", ConnectionRole.FEEDFORWARD, surprise_tracker=SurpriseTracker()
    )
    m1_gate = ThalamicGate()
    cortex.connect("M1", "S1", ConnectionRole.APICAL, thalamic_gate=m1_gate)

    t0 = time.time()
    result = cortex.run(tokens, log_interval=100000)  # suppress per-step logging
    elapsed = time.time() - t0

    m1 = result.per_region["M1"]

    # Compute rates
    int_rate = m1.turn_interruptions / max(m1.turn_input_steps, 1)
    unr_rate = m1.turn_unresponsive / max(m1.turn_eom_steps, 1)
    speak_rate = m1.turn_correct_speak / max(m1.turn_eom_steps, 1)
    ramble_rate = m1.turn_rambles / max(m1.turn_eom_steps, 1)

    # BG gate stats
    bg_gates = m1.bg_gate_values
    bg_mean = sum(bg_gates) / len(bg_gates) if bg_gates else 0.5
    bg_min = min(bg_gates) if bg_gates else 0.5
    bg_max = max(bg_gates) if bg_gates else 0.5

    # M1 accuracy (last 20% of motor_accuracies)
    accs = m1.motor_accuracies
    tail = accs[int(len(accs) * 0.8) :] if accs else []
    m1_acc = sum(tail) / len(tail) if tail else 0.0

    return {
        **cfg,
        "n_tokens_actual": len(tokens),
        "n_eom": sum(1 for tid, _ in tokens if tid == -2),
        "int_rate": int_rate,
        "unr_rate": unr_rate,
        "speak_rate": speak_rate,
        "ramble_rate": ramble_rate,
        "bg_mean": bg_mean,
        "bg_min": bg_min,
        "bg_max": bg_max,
        "m1_acc": m1_acc,
        "eom_steps": m1.turn_eom_steps,
        "input_steps": m1.turn_input_steps,
        "elapsed": elapsed,
    }


def print_results(results):
    """Print results as a formatted table."""
    print("\n" + "=" * 130)
    print(
        f"{'dataset':<12} {'tokens':>6} {'seg':>4} {'win':>4} {'bg_lr':>6} "
        f"| {'int%':>5} {'unr%':>5} {'spk%':>5} {'ram%':>5} "
        f"| {'bg_mn':>5} {'bg_lo':>5} {'bg_hi':>5} "
        f"| {'M1acc':>5} {'eom':>5} {'inp':>5} {'time':>5}"
    )
    print("-" * 130)

    for r in results:
        line = (
            f"{r['dataset']:<12} "
            f"{r['tokens']:>6} {r['segment']:>4} "
            f"{r['window']:>4} {r['bg_lr']:>6.3f} "
            f"| {r['int_rate']:>5.1%} "
            f"{r['unr_rate']:>5.1%} "
            f"{r['speak_rate']:>5.1%} "
            f"{r['ramble_rate']:>5.1%} "
            f"| {r['bg_mean']:>5.2f} "
            f"{r['bg_min']:>5.2f} {r['bg_max']:>5.2f} "
            f"| {r['m1_acc']:>5.1%} "
            f"{r['eom_steps']:>5} "
            f"{r['input_steps']:>5} "
            f"{r['elapsed']:>5.1f}s"
        )
        print(line)

    print("=" * 130)


def main():
    parser = argparse.ArgumentParser(description="Sweep BG parameters")
    parser.add_argument("--quick", action="store_true", help="Run quick subset only")
    args = parser.parse_args()

    mode = "quick" if args.quick else "full"
    configs = CONFIGS[mode]
    print(f"Running {len(configs)} configurations ({mode} mode)...\n")

    results = []
    for i, cfg in enumerate(configs):
        label = (
            f"[{i + 1}/{len(configs)}] "
            f"{cfg['dataset']} {cfg['tokens']}tok seg={cfg['segment']} "
            f"win={cfg['window']} bg_lr={cfg['bg_lr']}"
        )
        print(f"\n{'=' * 60}")
        print(label)
        print(f"{'=' * 60}")

        r = run_config(cfg)
        results.append(r)

        # Print inline summary
        print(
            f"  => int={r['int_rate']:.1%} unr={r['unr_rate']:.1%} "
            f"spk={r['speak_rate']:.1%} bg={r['bg_mean']:.2f} "
            f"[{r['bg_min']:.2f}-{r['bg_max']:.2f}] "
            f"M1={r['m1_acc']:.1%} ({r['elapsed']:.1f}s)"
        )

    print_results(results)


if __name__ == "__main__":
    main()
