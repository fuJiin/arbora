#!/usr/bin/env python3
"""Run cortex model and save outputs for later dashboard generation.

Usage:
    uv run experiments/scripts/cortex_run.py --tokens 5000 --char-level --hierarchy
    uv run experiments/scripts/cortex_run.py --tokens 20000 --char-level --hierarchy \
        --buffer-depth 4 --burst-gate --apical --name my-experiment
    uv run experiments/scripts/cortex_run.py --dataset tinydialogues --tokens 50000 \
        --buffer-depth 4 --burst-gate --apical --gate-feedback
"""

import argparse
import string

import step.env  # noqa: F401
from step.config import (
    CortexConfig,
    _default_motor_config,
    _default_region2_config,
    _default_s1_config,
    make_motor_region,
    make_sensory_region,
)
from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.modulators import SurpriseTracker, ThalamicGate
from step.cortex.topology import Topology
from step.data import (
    inject_eom_tokens,
    prepare_tokens,
    prepare_tokens_charlevel,
    prepare_tokens_personachat,
    prepare_tokens_tinydialogues,
)
from step.encoders.charbit import CharbitEncoder
from step.encoders.positional import PositionalCharEncoder
from step.runs import auto_name, auto_tags, save_run

CHARS = string.printable
CHAR_LENGTH = 8
CHAR_WIDTH = len(CHARS) + 1


def main():
    parser = argparse.ArgumentParser(description="Run cortex and save outputs")
    parser.add_argument("--tokens", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--name", type=str, default=None, help="Run name (auto-generated if omitted)"
    )
    parser.add_argument(
        "--hierarchy", action="store_true", help="Run two-region hierarchy"
    )
    parser.add_argument(
        "--char-level",
        action="store_true",
        help="Use character-level tokenization with positional encoding",
    )
    parser.add_argument(
        "--buffer-depth",
        type=int,
        default=1,
        help="Temporal buffer depth for S1→S2 feedforward (default: 1 = direct)",
    )
    parser.add_argument(
        "--burst-gate",
        action="store_true",
        help="Gate feedforward signal by bursting columns (novel events only)",
    )
    parser.add_argument(
        "--apical",
        action="store_true",
        help="Enable S2→S1 apical feedback connection",
    )
    parser.add_argument(
        "--gate-feedback",
        action="store_true",
        help="Thalamic gating: suppress feedback until receiver stabilizes",
    )
    parser.add_argument(
        "--motor",
        action="store_true",
        help="Add M1 motor region: S1→M1 feedforward, M1→S1 apical feedback",
    )
    parser.add_argument(
        "--reward",
        action="store_true",
        help="Enable reward modulation: M1→S1 dopaminergic turn-taking reward",
    )
    parser.add_argument(
        "--eom",
        action="store_true",
        help="Inject EOM tokens at story boundaries for turn-taking training",
    )
    parser.add_argument(
        "--eom-segment",
        type=int,
        default=0,
        help="Synthetic turn boundary every N tokens (0=natural boundaries only)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["tinydialogues", "personachat", "babylm"],
        help="Use a specific dataset (implies --char-level --hierarchy --motor --eom)",
    )
    parser.add_argument(
        "--speak-window",
        type=int,
        default=10,
        help="EOM speak window for TinyDialogues (default: 10)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Save checkpoint after training (name or path)",
    )
    args = parser.parse_args()

    # Dataset presets: dialogue datasets imply full architecture
    # (must match REPL's build_model() for checkpoint compatibility)
    if args.dataset in ("tinydialogues", "personachat", "babylm"):
        args.char_level = True
        args.hierarchy = True
        args.motor = True
        args.eom = True
        # Only set S2 flags if user hasn't overridden them
        if args.buffer_depth == 1:
            args.buffer_depth = 4
        if not args.burst_gate:
            args.burst_gate = True
        if not args.apical:
            args.apical = True
        if not args.gate_feedback:
            args.gate_feedback = True

    cortex_cfg = CortexConfig()

    if args.dataset == "tinydialogues":
        tokens = prepare_tokens_tinydialogues(
            args.tokens, speak_window=args.speak_window,
        )
        alphabet = sorted({ch for _, ch in tokens if _ >= 0})
        encoder = PositionalCharEncoder("".join(alphabet), max_positions=8)
        input_dim = encoder.input_dim
        encoding_width = encoder.encoding_width
        cortex_cfg = _default_s1_config()
    elif args.dataset == "personachat":
        tokens = prepare_tokens_personachat(
            args.tokens, speak_window=args.speak_window,
        )
        alphabet = sorted({ch for _, ch in tokens if _ >= 0})
        encoder = PositionalCharEncoder("".join(alphabet), max_positions=8)
        input_dim = encoder.input_dim
        encoding_width = encoder.encoding_width
        cortex_cfg = _default_s1_config()
    elif args.dataset == "babylm":
        tokens = prepare_tokens_charlevel(args.tokens, dataset="babylm")
        alphabet = sorted({ch for _, ch in tokens if _ >= 0})
        encoder = PositionalCharEncoder("".join(alphabet), max_positions=8)
        input_dim = encoder.input_dim
        encoding_width = encoder.encoding_width
        cortex_cfg = _default_s1_config()
    elif args.char_level:
        tokens = prepare_tokens_charlevel(args.tokens)
        alphabet = sorted({ch for _, ch in tokens if _ != -1})
        encoder = PositionalCharEncoder("".join(alphabet), max_positions=8)
        input_dim = encoder.input_dim
        encoding_width = encoder.encoding_width
        cortex_cfg = CortexConfig(ltd_rate=0.05)
    else:
        tokens = prepare_tokens(args.tokens)
        encoder = CharbitEncoder(length=CHAR_LENGTH, width=CHAR_WIDTH, chars=CHARS)
        input_dim = CHAR_LENGTH * CHAR_WIDTH
        encoding_width = CHAR_WIDTH

    if args.eom and args.dataset not in ("tinydialogues", "personachat"):
        # BabyLM has few natural boundaries — use synthetic segmentation
        if args.dataset == "babylm" and args.eom_segment == 0:
            args.eom_segment = 200
        tokens = inject_eom_tokens(tokens, segment_length=args.eom_segment)

    if args.hierarchy:
        cortex, result = _run_hierarchy(
            tokens,
            cortex_cfg,
            encoder,
            input_dim,
            encoding_width,
            args,
        )
    else:
        cortex, result = _run_single(
            tokens,
            cortex_cfg,
            encoder,
            input_dim,
            encoding_width,
            args,
        )

    # Build region configs for metadata
    region_configs = _build_region_configs(cortex_cfg, args)

    # Determine run name
    name = args.name or auto_name(
        hierarchy=args.hierarchy,
        char_level=args.char_level,
        n_tokens=len(tokens),
        buffer_depth=args.buffer_depth,
        burst_gate=args.burst_gate,
        apical=args.apical,
        gate_feedback=args.gate_feedback,
        motor=args.motor,
        dataset=args.dataset,
    )

    tags = auto_tags(
        hierarchy=args.hierarchy,
        char_level=args.char_level,
        buffer_depth=args.buffer_depth,
        burst_gate=args.burst_gate,
        apical=args.apical,
        gate_feedback=args.gate_feedback,
        motor=args.motor,
        dataset=args.dataset,
    )

    run_dir = save_run(
        name=name,
        timelines=dict(cortex.timelines),
        diagnostics=dict(cortex.diagnostics),
        result=result,
        region_configs=region_configs,
        meta_extra={
            "tags": tags,
            "n_tokens": len(tokens),
            "encoder": type(encoder).__name__,
        },
    )

    # Save checkpoint if requested
    if args.checkpoint:
        import os
        ckpt_dir = "experiments/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = args.checkpoint
        if not ckpt_path.endswith(".ckpt"):
            ckpt_path = os.path.join(ckpt_dir, f"{ckpt_path}.ckpt")
        cortex.save_checkpoint(ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    print(f"Run '{name}' saved to {run_dir}")


def _run_single(tokens, cortex_cfg, encoder, input_dim, encoding_width, args):
    region = _make_region(cortex_cfg, input_dim, encoding_width)

    cortex = Topology(
        encoder,
        enable_timeline=True,
        diagnostics_interval=args.log_interval,
    )
    cortex.add_region("S1", region, entry=True)

    print(f"\nRunning cortex on {len(tokens):,} tokens...")
    result = cortex.run(tokens, log_interval=args.log_interval)

    return cortex, result


def _run_hierarchy(tokens, cortex_cfg, encoder, input_dim, encoding_width, args):
    region1 = _make_region(cortex_cfg, input_dim, encoding_width)
    r2_cfg = _default_region2_config()
    r2_input_dim = region1.n_l23_total * args.buffer_depth
    region2 = make_sensory_region(r2_cfg, r2_input_dim, seed=123)

    surprise = SurpriseTracker()

    cortex = Topology(
        encoder,
        enable_timeline=True,
        diagnostics_interval=args.log_interval,
    )
    cortex.add_region("S1", region1, entry=True)
    cortex.add_region("S2", region2)
    cortex.connect(
        "S1",
        "S2",
        "feedforward",
        buffer_depth=args.buffer_depth,
        burst_gate=args.burst_gate,
    )
    cortex.connect("S1", "S2", "surprise", surprise_tracker=surprise)
    if args.apical:
        gate = ThalamicGate() if args.gate_feedback else None
        cortex.connect("S2", "S1", "apical", thalamic_gate=gate)

    if args.motor:
        m1_cfg = _default_motor_config()
        motor = make_motor_region(m1_cfg, region1.n_l23_total, seed=456)
        bg = BasalGanglia(
            context_dim=region1.n_columns + 1,  # per-col burst + overall burst frac
            seed=789,
        ) if args.reward else None
        cortex.add_region("M1", motor, basal_ganglia=bg)
        cortex.connect("S1", "M1", "feedforward")
        cortex.connect("S1", "M1", "surprise", surprise_tracker=SurpriseTracker())
        # M1→S1 apical only if dimensions match S2→S1 (both send to same target)
        if not args.apical or motor.n_l23_total == region2.n_l23_total:
            m1_gate = ThalamicGate() if args.gate_feedback else None
            cortex.connect("M1", "S1", "apical", thalamic_gate=m1_gate)

    print(f"\nRunning hierarchy on {len(tokens):,} tokens...")
    result = cortex.run(tokens, log_interval=args.log_interval)

    return cortex, result


def _make_region(cortex_cfg, input_dim, encoding_width):
    return make_sensory_region(cortex_cfg, input_dim, encoding_width)


def _build_region_configs(cortex_cfg, args):
    configs = {
        "S1": {
            "n_columns": cortex_cfg.n_columns,
            "k_columns": cortex_cfg.k_columns,
            "n_l4": cortex_cfg.n_l4,
            "n_l23": cortex_cfg.n_l23,
            "learning_rate": cortex_cfg.learning_rate,
            "ltd_rate": cortex_cfg.ltd_rate,
            "voltage_decay": cortex_cfg.voltage_decay,
        },
    }
    if args.hierarchy:
        r2_cfg = _default_region2_config()
        configs["S2"] = {
            "n_columns": r2_cfg.n_columns,
            "k_columns": r2_cfg.k_columns,
            "n_l4": r2_cfg.n_l4,
            "n_l23": r2_cfg.n_l23,
            "learning_rate": r2_cfg.learning_rate,
            "ltd_rate": r2_cfg.ltd_rate,
            "voltage_decay": r2_cfg.voltage_decay,
            "buffer_depth": args.buffer_depth,
            "burst_gate": args.burst_gate,
            "apical": args.apical,
        }
    if args.motor:
        m1_cfg = _default_motor_config()
        configs["M1"] = {
            "n_columns": m1_cfg.n_columns,
            "k_columns": m1_cfg.k_columns,
            "n_l4": m1_cfg.n_l4,
            "n_l23": m1_cfg.n_l23,
            "learning_rate": m1_cfg.learning_rate,
            "ltd_rate": m1_cfg.ltd_rate,
            "voltage_decay": m1_cfg.voltage_decay,
            "motor": True,
        }
    return configs


if __name__ == "__main__":
    main()
