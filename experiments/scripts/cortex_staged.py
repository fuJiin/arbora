#!/usr/bin/env python3
"""Staged developmental training for the cortex model.

Runs training stages sequentially, each with different learning
configurations. Checkpoints flow between stages.

Usage:
    # Run all stages from scratch
    uv run experiments/scripts/cortex_staged.py

    # Run a specific stage (loads its prerequisite checkpoint)
    uv run experiments/scripts/cortex_staged.py --stage sensory

    # Resume from a checkpoint, run remaining stages
    uv run experiments/scripts/cortex_staged.py --resume stage1_sensory

    # Override tokens for a stage
    uv run experiments/scripts/cortex_staged.py --stage sensory --tokens 500000
"""

import argparse
import os
import time

import numpy as np

import step.env  # noqa: F401
from step.config import (
    _default_motor_config,
    _default_region2_config,
    _default_region3_config,
    _default_s1_config,
    make_motor_region,
    make_sensory_region,
)
from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.modulators import SurpriseTracker, ThalamicGate
from step.cortex.stages import (
    BABBLING_STAGE,
    GUIDED_BABBLING_STAGE,
    SENSORY_STAGE,
    TrainingStage,
)
from step.cortex.topology import Topology
from step.data import inject_eom_tokens, prepare_tokens_charlevel
from step.encoders.positional import PositionalCharEncoder
from step.runs import save_run

CKPT_DIR = "experiments/checkpoints"

# Ordered list of all stages
ALL_STAGES = [
    SENSORY_STAGE,
    BABBLING_STAGE,
    GUIDED_BABBLING_STAGE,
]

STAGE_MAP = {s.name: s for s in ALL_STAGES}


def build_topology(encoder, *, log_interval=100, timeline_interval=100):
    """Build the full topology with all regions and connections.

    All regions and connections are created upfront. Stages control
    which are active via freeze/enable APIs.
    """
    s1_cfg = _default_s1_config()
    s1 = make_sensory_region(s1_cfg, encoder.input_dim, encoder.encoding_width)

    r2_cfg = _default_region2_config()
    s2 = make_sensory_region(r2_cfg, s1.n_l23_total * 4, seed=123)

    r3_cfg = _default_region3_config()
    s3 = make_sensory_region(r3_cfg, s2.n_l23_total * 8, seed=789)

    m1_cfg = _default_motor_config()
    # Build M1 vocabulary from encoder's character set
    output_vocab = [ord(ch) for ch in encoder._char_to_idx]
    m1 = make_motor_region(m1_cfg, s1.n_l23_total, seed=456)
    # Set vocabulary for L5 output mapping
    m1._output_vocab = np.array(output_vocab, dtype=np.int64)
    m1.n_output_tokens = len(output_vocab)
    # Reinitialize L5 weights with correct vocab size
    n_l23 = m1.n_l23_total
    m1.output_weights = m1._rng.uniform(0, 0.01, size=(n_l23, len(output_vocab)))
    m1.output_mask = (m1._rng.random((n_l23, len(output_vocab))) < 0.5).astype(np.float64)
    m1.output_weights *= m1.output_mask
    m1._output_eligibility = np.zeros((n_l23, len(output_vocab)))

    cortex = Topology(
        encoder,
        enable_timeline=timeline_interval > 0,
        timeline_interval=max(timeline_interval, 1),
        diagnostics_interval=log_interval,
    )

    # Regions
    cortex.add_region("S1", s1, entry=True)
    cortex.add_region("S2", s2)
    cortex.add_region("S3", s3)
    bg = BasalGanglia(
        context_dim=s1.n_columns + 1,
        learning_rate=0.05,
        seed=789,
    )
    cortex.add_region("M1", m1, basal_ganglia=bg)

    # Feedforward
    cortex.connect("S1", "S2", "feedforward", buffer_depth=4, burst_gate=True)
    cortex.connect("S2", "S3", "feedforward", buffer_depth=8, burst_gate=True)
    cortex.connect("S1", "M1", "feedforward")

    # Surprise
    cortex.connect("S1", "S2", "surprise", surprise_tracker=SurpriseTracker())
    cortex.connect("S2", "S3", "surprise", surprise_tracker=SurpriseTracker())
    cortex.connect("S1", "M1", "surprise", surprise_tracker=SurpriseTracker())

    # Apical feedback (with thalamic gating)
    cortex.connect("S2", "S1", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("S3", "S2", "apical", thalamic_gate=ThalamicGate())
    cortex.connect("M1", "S1", "apical", thalamic_gate=ThalamicGate())

    return cortex


def load_data(n_tokens):
    """Load BabyLM dataset with EOM injection.

    Always samples at least 100k chars to discover the full alphabet,
    ensuring consistent encoder dimensions across stages/checkpoints.
    """
    # Discover full alphabet from a large sample — need enough to find
    # all chars (rare chars only appear after many documents)
    vocab_sample = max(n_tokens, 1_000_000)
    all_tokens = prepare_tokens_charlevel(vocab_sample, dataset="babylm")
    alphabet = sorted({ch for _, ch in all_tokens if _ >= 0})
    encoder = PositionalCharEncoder("".join(alphabet), max_positions=8)

    # Use only the requested amount for training
    if n_tokens < vocab_sample:
        tokens = all_tokens[:n_tokens]
    else:
        tokens = all_tokens
    tokens = inject_eom_tokens(tokens, segment_length=200)
    return tokens, encoder


def resolve_checkpoint(name):
    """Resolve a checkpoint name to a full path."""
    if name is None:
        return None
    path = name
    if not path.endswith(".ckpt"):
        path = os.path.join(CKPT_DIR, f"{path}.ckpt")
    return path


def run_stage(
    cortex, stage, tokens, *, log_interval=100,
):
    """Configure and run a single training stage."""
    print(f"\n{'='*60}")
    print(f"Stage: {stage.name} — {stage.description}")
    print(f"  Tokens: {stage.n_tokens:,}")
    print(f"  Learning: {', '.join(stage.learning_regions) or 'none'}")
    print(f"{'='*60}\n")

    # Load checkpoint if specified
    ckpt_path = resolve_checkpoint(stage.load_checkpoint)
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        cortex.load_checkpoint(ckpt_path)

    # Apply stage configuration (freeze/unfreeze, enable/disable)
    stage.configure(cortex)

    # Choose run mode: babbling (autoregressive) or corpus-driven
    is_babbling = stage.babbling_noise > 0 or stage.force_motor_active

    if is_babbling:
        # Autoregressive babbling: M1 drives, hears itself through S1
        print(f"  Mode: autoregressive babbling (noise={stage.babbling_noise})")
        result = cortex.run_babbling(
            stage.n_tokens, log_interval=log_interval,
        )
        elapsed = result["elapsed_seconds"]
    else:
        # Corpus-driven: standard token-by-token processing
        stage_tokens = tokens[:stage.n_tokens]
        if len(stage_tokens) < stage.n_tokens:
            print(
                f"  Warning: only {len(stage_tokens):,} tokens available "
                f"(requested {stage.n_tokens:,})"
            )

        start = time.monotonic()
        result = cortex.run(stage_tokens, log_interval=log_interval)
        elapsed = time.monotonic() - start

    print(f"\nStage '{stage.name}' completed in {elapsed:.1f}s")

    # Save checkpoint
    if stage.save_checkpoint:
        ckpt_path = resolve_checkpoint(stage.save_checkpoint)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        cortex.save_checkpoint(ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    # Save run data (babbling returns dict, corpus returns CortexResult)
    if not is_babbling:
        run_dir = save_run(
            name=f"staged-{stage.name}-{stage.n_tokens // 1000}k",
            timelines=dict(cortex.timelines),
            diagnostics=dict(cortex.diagnostics),
            result=result,
            region_configs={},
            meta_extra={
                "stage": stage.name,
                "n_tokens": stage.n_tokens,
                "learning_regions": stage.learning_regions,
                "encoder": "PositionalCharEncoder",
            },
        )
        print(f"Run saved: {run_dir}")
    else:
        print(
            f"  Babbling summary: "
            f"{len(result['tokens_produced'])} chars produced, "
            f"{len(result['unique_tokens'])} unique"
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="Staged developmental training")
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        choices=list(STAGE_MAP.keys()),
        help="Run a single stage (default: run all stages in order)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=None,
        help="Override token count for the stage(s)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from this checkpoint (skip earlier stages)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Logging interval (default: 100)",
    )
    parser.add_argument(
        "--timeline-interval",
        type=int,
        default=100,
        help="Timeline capture interval (default: 100, 0=disable)",
    )
    args = parser.parse_args()

    # Determine which stages to run
    if args.stage:
        stages = [STAGE_MAP[args.stage]]
    elif args.resume:
        # Find the stage that saves this checkpoint and start from the next one
        resume_idx = None
        for i, s in enumerate(ALL_STAGES):
            if s.save_checkpoint == args.resume:
                resume_idx = i + 1
                break
        if resume_idx is None:
            # Assume resume is a stage name checkpoint, inject it
            print(f"Loading checkpoint '{args.resume}' before first stage")
            stages = list(ALL_STAGES)
            stages[0] = TrainingStage(
                name=stages[0].name,
                description=stages[0].description,
                n_tokens=stages[0].n_tokens,
                learning_regions=stages[0].learning_regions,
                connections=stages[0].connections,
                load_checkpoint=args.resume,
                save_checkpoint=stages[0].save_checkpoint,
            )
        else:
            stages = ALL_STAGES[resume_idx:]
            if not stages:
                print("All stages already completed!")
                return
    else:
        stages = list(ALL_STAGES)

    # Override token counts if specified
    if args.tokens:
        stages = [
            TrainingStage(
                name=s.name,
                description=s.description,
                n_tokens=args.tokens,
                learning_regions=s.learning_regions,
                connections=s.connections,
                load_checkpoint=s.load_checkpoint,
                save_checkpoint=s.save_checkpoint,
                babbling_noise=s.babbling_noise,
                force_motor_active=s.force_motor_active,
                reward_source=s.reward_source,
            )
            for s in stages
        ]

    # Load data (max tokens across all stages)
    max_tokens = max(s.n_tokens for s in stages)
    tokens, encoder = load_data(max_tokens)

    # Build topology
    cortex = build_topology(
        encoder,
        log_interval=args.log_interval,
        timeline_interval=args.timeline_interval,
    )

    # Run stages
    for stage in stages:
        run_stage(cortex, stage, tokens, log_interval=args.log_interval)

    print("\nAll stages complete.")


if __name__ == "__main__":
    main()
