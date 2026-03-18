#!/usr/bin/env python3
"""Sweep echo training configurations to find stable convergence.

Tests combinations of:
- batch_reward: apply reward once per episode (mean) vs per-step
- eligibility_clip: cap eligibility trace magnitudes
- reward_baseline_decay: adaptive baseline subtraction
- goal_consolidation_scale: how fast goal weights learn
- curriculum: start with short words, advance when stable

Usage:
    uv run experiments/scripts/echo_sweep.py
    uv run experiments/scripts/echo_sweep.py --episodes 2000 --sensory-tokens 50000
    uv run experiments/scripts/echo_sweep.py --config batch_clip  # run single config
"""

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np

import step.env  # noqa: F401
from step.config import (
    _default_motor_config,
    _default_pfc_config,
    _default_premotor_config,
    _default_region2_config,
    _default_region3_config,
    _default_s1_config,
    make_motor_region,
    make_pfc_region,
    make_premotor_region,
    make_sensory_region,
)
from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.modulators import SurpriseTracker, ThalamicGate
from step.cortex.stages import SENSORY_STAGE
from step.cortex.topology import Topology
from step.data import inject_eom_tokens, prepare_tokens_charlevel
from step.encoders.positional import PositionalCharEncoder


@dataclass
class EchoConfig:
    """Configuration for one echo sweep trial."""

    name: str
    batch_reward: bool = False
    eligibility_clip: float = 0.0
    reward_baseline_decay: float = 0.0
    goal_consolidation_scale: float = 0.3
    curriculum: bool = False
    babbling_noise: float = 0.2
    mismatch_scale: float = 0.1  # EchoReward mismatch penalty scale


# Sweep configurations — each tests a specific hypothesis
CONFIGS = {
    # Baseline: current behavior (should oscillate)
    "baseline": EchoConfig(
        name="baseline",
    ),
    # Fix 1: batch reward only
    "batch": EchoConfig(
        name="batch",
        batch_reward=True,
    ),
    # Fix 2: eligibility clamping only
    "clip": EchoConfig(
        name="clip",
        eligibility_clip=0.05,
    ),
    # Fix 3: reward baseline only
    "baseline_sub": EchoConfig(
        name="baseline_sub",
        reward_baseline_decay=0.95,
    ),
    # Fix 4: batch + clip (combined)
    "batch_clip": EchoConfig(
        name="batch_clip",
        batch_reward=True,
        eligibility_clip=0.05,
    ),
    # Fix 5: batch + baseline subtraction
    "batch_baseline": EchoConfig(
        name="batch_baseline",
        batch_reward=True,
        reward_baseline_decay=0.95,
    ),
    # Fix 6: all three fixes together
    "all_fixes": EchoConfig(
        name="all_fixes",
        batch_reward=True,
        eligibility_clip=0.05,
        reward_baseline_decay=0.95,
    ),
    # Fix 7: slower goal consolidation (0.1x instead of 0.3x)
    "slow_goal": EchoConfig(
        name="slow_goal",
        goal_consolidation_scale=0.1,
    ),
    # Fix 8: batch + slow goal
    "batch_slow": EchoConfig(
        name="batch_slow",
        batch_reward=True,
        goal_consolidation_scale=0.1,
    ),
    # Fix 9: curriculum (start with 2-char words)
    "curriculum": EchoConfig(
        name="curriculum",
        curriculum=True,
    ),
    # Fix 10: batch + clip + curriculum
    "batch_clip_curriculum": EchoConfig(
        name="batch_clip_curriculum",
        batch_reward=True,
        eligibility_clip=0.05,
        curriculum=True,
    ),
    # Fix 11: batch + baseline + slow goal (conservative)
    "conservative": EchoConfig(
        name="conservative",
        batch_reward=True,
        reward_baseline_decay=0.95,
        goal_consolidation_scale=0.1,
    ),
    # Fix 12: symmetric reward (mismatch penalty = 50% of match)
    "symmetric": EchoConfig(
        name="symmetric",
        mismatch_scale=0.5,
    ),
    # Fix 13: symmetric + batch (test reward symmetry with batch)
    "sym_batch": EchoConfig(
        name="sym_batch",
        batch_reward=True,
        mismatch_scale=0.5,
    ),
    # Fix 14: symmetric + clip
    "sym_clip": EchoConfig(
        name="sym_clip",
        eligibility_clip=0.05,
        mismatch_scale=0.5,
    ),
    # Fix 15: symmetric + curriculum (the full package)
    "sym_curriculum": EchoConfig(
        name="sym_curriculum",
        curriculum=True,
        mismatch_scale=0.5,
    ),
    # Fix 16: batch + clip + symmetric reward (strongest combo)
    "best_combo": EchoConfig(
        name="best_combo",
        batch_reward=True,
        eligibility_clip=0.05,
        mismatch_scale=0.5,
        curriculum=True,
    ),
}


def build_topology(encoder, *, timeline_interval=0):
    """Build topology (same as cortex_staged.py but without timeline)."""
    s1_cfg = _default_s1_config()
    s1 = make_sensory_region(s1_cfg, encoder.input_dim, encoder.encoding_width)

    r2_cfg = _default_region2_config()
    s2 = make_sensory_region(r2_cfg, s1.n_l23_total * 4, seed=123)

    r3_cfg = _default_region3_config()
    s3 = make_sensory_region(r3_cfg, s2.n_l23_total * 8, seed=789)

    m2_cfg = _default_premotor_config()
    m2_n_l23 = m2_cfg.n_columns * m2_cfg.n_l23

    m1_cfg = _default_motor_config()
    output_vocab = [ord(ch) for ch in encoder._char_to_idx]
    m1 = make_motor_region(m1_cfg, m2_n_l23, seed=456)
    m1._output_vocab = np.array(output_vocab, dtype=np.int64)
    m1.n_output_tokens = len(output_vocab)
    n_l23 = m1.n_l23_total
    m1.output_weights = m1._rng.uniform(0, 0.01, size=(n_l23, len(output_vocab)))
    m1.output_mask = (m1._rng.random((n_l23, len(output_vocab))) < 0.5).astype(
        np.float64
    )
    m1.output_weights *= m1.output_mask
    m1._output_eligibility = np.zeros((n_l23, len(output_vocab)))

    cortex = Topology(
        encoder,
        enable_timeline=False,
        timeline_interval=1,
        diagnostics_interval=1000,
    )

    cortex.add_region("S1", s1, entry=True)
    cortex.add_region("S2", s2)
    cortex.add_region("S3", s3)
    bg = BasalGanglia(context_dim=s1.n_columns + 1, learning_rate=0.05, seed=789)
    cortex.add_region("M1", m1, basal_ganglia=bg)

    pfc_cfg = _default_pfc_config()
    pfc = make_pfc_region(pfc_cfg, s2.n_l23_total + s3.n_l23_total, seed=999)
    cortex.add_region("PFC", pfc)

    m2 = make_premotor_region(m2_cfg, s2.n_l23_total + pfc.n_l23_total, seed=321)
    cortex.add_region("M2", m2)

    cortex.connect("S1", "S2", "feedforward", buffer_depth=4, burst_gate=True)
    cortex.connect("S2", "S3", "feedforward", buffer_depth=8, burst_gate=True)
    cortex.connect("S2", "PFC", "feedforward")
    cortex.connect("S3", "PFC", "feedforward")
    cortex.connect("S2", "M2", "feedforward")
    cortex.connect("PFC", "M2", "feedforward")
    cortex.connect("M2", "M1", "feedforward")
    cortex.connect("S1", "S2", "surprise", surprise_tracker=SurpriseTracker())
    cortex.connect("S2", "S3", "surprise", surprise_tracker=SurpriseTracker())
    cortex.connect("S1", "M1", "surprise", surprise_tracker=SurpriseTracker())
    cortex.connect("S2", "S1", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("S3", "S2", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("M1", "M2", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("M2", "PFC", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("S1", "M1", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("M1", "S1", "apical", thalamic_gate=ThalamicGate())

    return cortex


def run_sensory_warmup(cortex, tokens, n_tokens):
    """Quick sensory training to bootstrap representations."""
    print(f"  Sensory warmup: {n_tokens:,} tokens...")
    SENSORY_STAGE.configure(cortex)
    stage_tokens = tokens[:n_tokens]
    cortex.run(stage_tokens, log_interval=n_tokens // 5)
    print("  Sensory warmup complete.")


def run_echo_trial(cortex, tokens, cfg: EchoConfig, n_episodes: int):
    """Run one echo trial with the given configuration."""
    # Configure M1 with this trial's parameters
    m1 = None
    for _name, s in cortex._regions.items():
        if s.motor:
            m1 = s.region
            break
    if m1 is None:
        raise ValueError("No motor region found")

    m1._eligibility_clip = cfg.eligibility_clip
    m1._reward_baseline_decay = cfg.reward_baseline_decay
    m1._reward_baseline = 0.0
    m1.goal_consolidation_scale = cfg.goal_consolidation_scale

    # Enable all regions for echo
    for _name, s in cortex._regions.items():
        s.region.learning_enabled = True

    # Build EchoReward kwargs
    echo_kwargs = {}
    if cfg.mismatch_scale != 0.1:
        echo_kwargs["mismatch_scale"] = cfg.mismatch_scale

    result = cortex.run_echo(
        tokens,
        n_episodes,
        max_word_len=6,
        min_word_len=2,
        log_interval=max(n_episodes // 20, 10),
        batch_reward=cfg.batch_reward,
        curriculum=cfg.curriculum,
        echo_reward_kwargs=echo_kwargs or None,
    )

    return result


def analyze_results(results: dict[str, dict]) -> None:
    """Print comparison table of all configurations."""
    print("\n" + "=" * 70)
    print("ECHO SWEEP RESULTS")
    print("=" * 70)

    # Table header
    print(
        f"{'Config':<25} {'Avg%':>6} {'Peak%':>6} "
        f"{'Last50%':>7} {'StdDev':>7} {'Trend':>7}"
    )
    print("-" * 70)

    ranked = []
    for name, result in sorted(results.items()):
        matches = result["matches"]
        if not matches:
            continue
        avg = 100 * sum(matches) / len(matches)
        peak = 100 * max(matches)
        last50 = matches[-50:] if len(matches) >= 50 else matches
        last50_avg = 100 * sum(last50) / len(last50)
        std = 100 * float(np.std(matches))

        # Trend: compare first half to second half
        half = len(matches) // 2
        first_half = sum(matches[:half]) / max(half, 1)
        second_half = sum(matches[half:]) / max(len(matches) - half, 1)
        trend = 100 * (second_half - first_half)

        # Oscillation metric: count sign changes in rolling average
        window = 50
        if len(matches) > window * 2:
            rolling = [
                sum(matches[i : i + window]) / window
                for i in range(0, len(matches) - window, window)
            ]
            sign_changes = sum(
                1
                for i in range(1, len(rolling))
                if (rolling[i] - rolling[i - 1])
                * (rolling[i - 1] - rolling[max(i - 2, 0)])
                < 0
            )
        else:
            sign_changes = 0

        trend_str = f"{trend:+.1f}"
        print(
            f"{name:<25} {avg:>5.1f}% {peak:>5.1f}% "
            f"{last50_avg:>6.1f}% {std:>6.1f}% {trend_str:>7} "
            f"osc={sign_changes}"
        )
        ranked.append((name, last50_avg, std, sign_changes))

    # Rank by: high last50 avg, low std, low oscillation
    print("\n" + "-" * 70)
    ranked.sort(key=lambda x: (-x[1], x[2], x[3]))
    best = ranked[0]
    print(f"Best: {best[0]} (last50={best[1]:.1f}%, std={best[2]:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Echo training sweep")
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Echo episodes per config (default: 1000)",
    )
    parser.add_argument(
        "--sensory-tokens",
        type=int,
        default=50000,
        help="Sensory warmup tokens (default: 50000)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        choices=list(CONFIGS.keys()),
        help="Run a single config (default: run all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/echo_sweep.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    all_tokens = prepare_tokens_charlevel(1_000_000, dataset="babylm")
    alphabet = sorted({ch for _, ch in all_tokens if _ >= 0})
    encoder = PositionalCharEncoder("".join(alphabet), max_positions=8)
    tokens = all_tokens[: max(args.sensory_tokens, 100_000)]
    tokens = inject_eom_tokens(tokens, segment_length=200)

    # Select configs to run
    configs_to_run = (
        {args.config: CONFIGS[args.config]} if args.config else CONFIGS
    )

    results = {}

    for cfg_name, cfg in configs_to_run.items():
        print(f"\n{'=' * 60}")
        print(f"CONFIG: {cfg_name}")
        print(f"  batch_reward={cfg.batch_reward}")
        print(f"  eligibility_clip={cfg.eligibility_clip}")
        print(f"  reward_baseline_decay={cfg.reward_baseline_decay}")
        print(f"  goal_consolidation_scale={cfg.goal_consolidation_scale}")
        print(f"  curriculum={cfg.curriculum}")
        print(f"  mismatch_scale={cfg.mismatch_scale}")
        print(f"{'=' * 60}")

        # Fresh topology for each config (fair comparison)
        cortex = build_topology(encoder)
        cortex.finalize()

        # Sensory warmup (same for all configs)
        run_sensory_warmup(cortex, tokens, args.sensory_tokens)

        # Run echo
        start = time.monotonic()
        result = run_echo_trial(cortex, tokens, cfg, args.episodes)
        elapsed = time.monotonic() - start

        results[cfg_name] = {
            "matches": result["matches"],
            "rewards": result["rewards"],
            "avg_match": result["avg_match"],
            "echo_summary": result["echo_summary"],
            "elapsed_seconds": elapsed,
            "config": {
                "batch_reward": cfg.batch_reward,
                "eligibility_clip": cfg.eligibility_clip,
                "reward_baseline_decay": cfg.reward_baseline_decay,
                "goal_consolidation_scale": cfg.goal_consolidation_scale,
                "curriculum": cfg.curriculum,
                "mismatch_scale": cfg.mismatch_scale,
            },
        }

        print(
            f"\n  {cfg_name}: avg_match={result['avg_match']:.1%}, "
            f"elapsed={elapsed:.1f}s"
        )

    # Analyze
    analyze_results(results)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
