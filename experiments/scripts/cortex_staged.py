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
from dataclasses import replace

import step.env  # noqa: F401
from step.cortex.canonical import build_canonical_circuit
from step.cortex.stages import (
    BABBLING_STAGE,
    GUIDED_BABBLING_STAGE,
    SENSORY_STAGE,
)
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


def build_circuit(encoder, *, log_interval=100, timeline_interval=100):
    """Build the full circuit with all regions and connections.

    Thin wrapper around build_canonical_circuit(). Stages control
    which regions are active via freeze/enable APIs.
    """
    return build_canonical_circuit(
        encoder,
        log_interval=log_interval,
        timeline_interval=timeline_interval,
        finalize=False,
    )


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
    tokens = all_tokens[:n_tokens] if n_tokens < vocab_sample else all_tokens
    tokens = inject_eom_tokens(tokens, segment_length=200)
    return tokens, encoder


def _extract_vocabulary(tokens, min_count=3, min_length=2):
    """Extract common words from corpus tokens for caregiver reward."""
    from collections import Counter

    words = Counter()
    current = []
    for token_id, ch in tokens:
        if token_id < 0 or ch in (" ", ".", ",", "!", "?", "'", "-", ""):  # boundary
            if len(current) >= min_length:
                words["".join(current)] += 1
            current.clear()
        else:
            current.append(ch)
    return {w for w, c in words.items() if c >= min_count}


def resolve_checkpoint(name):
    """Resolve a checkpoint name to a full path."""
    if name is None:
        return None
    path = name
    if not path.endswith(".ckpt"):
        path = os.path.join(CKPT_DIR, f"{path}.ckpt")
    return path


def run_stage(
    cortex,
    stage,
    tokens,
    *,
    log_interval=100,
):
    """Configure and run a single training stage."""
    print(f"\n{'=' * 60}")
    print(f"Stage: {stage.name} — {stage.description}")
    print(f"  Tokens: {stage.n_tokens:,}")
    print(f"  Learning: {', '.join(stage.learning_regions) or 'none'}")
    print(f"{'=' * 60}\n")

    # Load checkpoint if specified
    ckpt_path = resolve_checkpoint(stage.load_checkpoint)
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        cortex.load_checkpoint(ckpt_path)

    # Apply stage configuration (freeze/unfreeze, enable/disable)
    stage.configure(cortex)

    # Seed caregiver vocabulary from corpus if applicable
    if cortex._reward_source is not None:
        from step.cortex.reward import CaregiverReward

        if isinstance(cortex._reward_source, CaregiverReward):
            vocab = _extract_vocabulary(tokens)
            cortex._reward_source.seed_vocabulary(vocab)
            print(f"  Seeded caregiver with {len(vocab)} words from corpus")

    # Choose run mode: babbling (interleaved) or corpus-driven
    is_babbling = stage.babbling_noise > 0 or stage.force_motor_active

    if is_babbling:
        # Interleaved: alternate listening (corpus) and babbling (autoregressive)
        # babble_ratio: fraction of total steps that are babbling
        # n_babble_target = len(tokens) * ratio / (1 - ratio) = stage.n_tokens
        # => ratio = n_tokens / (n_tokens + len(tokens))
        n_listen = len(tokens)
        total = n_listen + stage.n_tokens
        babble_ratio = stage.n_tokens / total if total > 0 else 0.0
        print(
            f"  Mode: interleaved listen+babble "
            f"(noise={stage.babbling_noise}, ratio={babble_ratio:.2f})"
        )
        result = cortex.run(
            tokens,
            babble_ratio=babble_ratio,
            log_interval=log_interval,
        )
    else:
        # Corpus-driven: standard token-by-token processing
        stage_tokens = tokens[: stage.n_tokens]
        if len(stage_tokens) < stage.n_tokens:
            print(
                f"  Warning: only {len(stage_tokens):,} tokens available "
                f"(requested {stage.n_tokens:,})"
            )

        result = cortex.run(stage_tokens, log_interval=log_interval)

    print(f"\nStage '{stage.name}' completed in {result.elapsed_seconds:.1f}s")

    # Save checkpoint
    if stage.save_checkpoint:
        ckpt_path = resolve_checkpoint(stage.save_checkpoint)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        cortex.save_checkpoint(ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    # Save run data (now always CortexResult)
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

    if is_babbling and result.babble_tokens_produced:
        print(
            f"  Babbling summary: "
            f"{len(result.babble_tokens_produced)} chars produced, "
            f"{len(result.babble_unique_tokens)} unique"
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
            stages[0] = replace(stages[0], load_checkpoint=args.resume)
        else:
            stages = ALL_STAGES[resume_idx:]
            if not stages:
                print("All stages already completed!")
                return
    else:
        stages = list(ALL_STAGES)

    # Override token counts if specified
    if args.tokens:
        stages = [replace(s, n_tokens=args.tokens) for s in stages]

    # Load data (max tokens across all stages)
    max_tokens = max(s.n_tokens for s in stages)
    tokens, encoder = load_data(max_tokens)

    # Build circuit
    cortex = build_circuit(
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
