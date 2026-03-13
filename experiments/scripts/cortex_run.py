#!/usr/bin/env python3
"""Run cortex model and save outputs for later dashboard generation.

Usage:
    uv run experiments/scripts/cortex_run.py --tokens 5000 --char-level --hierarchy
    uv run experiments/scripts/cortex_run.py --tokens 20000 --char-level --hierarchy \
        --buffer-depth 4 --burst-gate --apical --name my-experiment
"""

import argparse
import string

import step.env  # noqa: F401
from step.config import CortexConfig, _default_motor_config, _default_region2_config
from step.cortex.motor import MotorRegion
from step.cortex.sensory import SensoryRegion
from step.cortex.surprise import SurpriseTracker, ThalamicGate
from step.cortex.topology import Topology
from step.data import prepare_tokens, prepare_tokens_charlevel
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
    args = parser.parse_args()

    cortex_cfg = CortexConfig()

    if args.char_level:
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
    )

    tags = auto_tags(
        hierarchy=args.hierarchy,
        char_level=args.char_level,
        buffer_depth=args.buffer_depth,
        burst_gate=args.burst_gate,
        apical=args.apical,
        gate_feedback=args.gate_feedback,
        motor=args.motor,
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
    region2 = SensoryRegion(
        input_dim=r2_input_dim,
        encoding_width=0,
        n_columns=r2_cfg.n_columns,
        n_l4=r2_cfg.n_l4,
        n_l23=r2_cfg.n_l23,
        k_columns=r2_cfg.k_columns,
        voltage_decay=r2_cfg.voltage_decay,
        eligibility_decay=r2_cfg.eligibility_decay,
        synapse_decay=r2_cfg.synapse_decay,
        learning_rate=r2_cfg.learning_rate,
        ltd_rate=r2_cfg.ltd_rate,
        seed=123,
    )

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
        motor = MotorRegion(
            input_dim=region1.n_l23_total,
            n_columns=m1_cfg.n_columns,
            n_l4=m1_cfg.n_l4,
            n_l23=m1_cfg.n_l23,
            k_columns=m1_cfg.k_columns,
            voltage_decay=m1_cfg.voltage_decay,
            eligibility_decay=m1_cfg.eligibility_decay,
            synapse_decay=m1_cfg.synapse_decay,
            learning_rate=m1_cfg.learning_rate,
            ltd_rate=m1_cfg.ltd_rate,
            seed=456,
        )
        cortex.add_region("M1", motor)
        cortex.connect("S1", "M1", "feedforward")
        cortex.connect("S1", "M1", "surprise", surprise_tracker=SurpriseTracker())
        m1_gate = ThalamicGate() if args.gate_feedback else None
        cortex.connect("M1", "S1", "apical", thalamic_gate=m1_gate)

    print(f"\nRunning hierarchy on {len(tokens):,} tokens...")
    result = cortex.run(tokens, log_interval=args.log_interval)

    return cortex, result


def _make_region(cortex_cfg, input_dim, encoding_width):
    return SensoryRegion(
        input_dim=input_dim,
        n_columns=cortex_cfg.n_columns,
        n_l4=cortex_cfg.n_l4,
        n_l23=cortex_cfg.n_l23,
        k_columns=cortex_cfg.k_columns,
        voltage_decay=cortex_cfg.voltage_decay,
        eligibility_decay=cortex_cfg.eligibility_decay,
        synapse_decay=cortex_cfg.synapse_decay,
        learning_rate=cortex_cfg.learning_rate,
        max_excitability=cortex_cfg.max_excitability,
        fb_boost=cortex_cfg.fb_boost,
        ltd_rate=cortex_cfg.ltd_rate,
        encoding_width=encoding_width,
        burst_learning_scale=cortex_cfg.burst_learning_scale,
        n_fb_segments=cortex_cfg.n_fb_segments,
        n_lat_segments=cortex_cfg.n_lat_segments,
        n_synapses_per_segment=cortex_cfg.n_synapses_per_segment,
        perm_threshold=cortex_cfg.perm_threshold,
        perm_init=cortex_cfg.perm_init,
        perm_increment=cortex_cfg.perm_increment,
        perm_decrement=cortex_cfg.perm_decrement,
        seg_activation_threshold=cortex_cfg.seg_activation_threshold,
        prediction_gain=cortex_cfg.prediction_gain,
        n_apical_segments=cortex_cfg.n_apical_segments,
        seed=cortex_cfg.seed,
    )


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
