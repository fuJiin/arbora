#!/usr/bin/env python3
"""Train a minimal T1 -> M1 circuit on MiniGrid-Empty-5x5.

Phase 0: verify obs -> encode -> process -> action pipeline works
end-to-end. No reward wiring yet.

Usage:
    uv run experiments/scripts/minigrid/train_minigrid.py
    uv run experiments/scripts/minigrid/train_minigrid.py --episodes 500
"""

from __future__ import annotations

import argparse

from arbora.basal_ganglia import BasalGangliaRegion
from arbora.cortex import SensoryRegion
from arbora.cortex.circuit import Circuit, ConnectionRole
from arbora.cortex.motor import MotorRegion
from arbora.probes.core import LaminaProbe
from examples.minigrid.agent import MiniGridAgent
from examples.minigrid.encoder import MiniGridEncoder
from examples.minigrid.env import MiniGridEnv
from examples.minigrid.harness import MiniGridHarness


def build_circuit(encoder: MiniGridEncoder) -> Circuit:
    """Build T1 -> BG -> M1 circuit for MiniGrid."""
    t1 = SensoryRegion(
        input_dim=encoder.input_dim,
        encoding_width=encoder.encoding_width,
        n_columns=64,
        n_l4=4,
        n_l23=4,
        n_l5=0,
        k_columns=4,
        seed=42,
    )
    bg = BasalGangliaRegion(
        input_dim=t1.n_l23_total,
        n_actions=7,
        seed=789,
    )
    m1 = MotorRegion(
        input_dim=t1.n_l23_total,
        n_columns=16,
        n_l4=0,
        n_l23=4,
        k_columns=2,
        n_output_tokens=7,
        seed=456,
    )
    circuit = Circuit(encoder)
    circuit.add_region("T1", t1, entry=True, input_region=True)
    circuit.add_region("BG", bg)
    circuit.add_region("M1", m1, output_region=True)
    circuit.connect(t1.output_port, bg.input_port, ConnectionRole.FEEDFORWARD)
    circuit.connect(t1.output_port, m1.input_port, ConnectionRole.FEEDFORWARD)
    circuit.connect(bg.output_port, m1.input_port, ConnectionRole.MODULATORY)
    circuit.finalize()
    return circuit


def main() -> None:
    parser = argparse.ArgumentParser(description="Train on MiniGrid-Empty-5x5")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--env", type=str, default="MiniGrid-Empty-5x5-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=500)
    args = parser.parse_args()

    print(f"MiniGrid Phase 0: {args.env}, {args.episodes} episodes")

    encoder = MiniGridEncoder()
    circuit = build_circuit(encoder)
    env = MiniGridEnv(args.env, max_episodes=args.episodes, seed=args.seed)
    agent = MiniGridAgent(encoder=encoder, circuit=circuit)
    probe = LaminaProbe()

    harness = MiniGridHarness(
        env, agent, probes=[probe], log_interval=args.log_interval
    )
    result = harness.run()

    # Print final probe snapshot
    snap = result.probe_snapshots.get("lamina", {})
    for region_name, region_snap in snap.items():
        print(
            f"  {region_name}: "
            f"recall={region_snap.input.recall:.3f} "
            f"prec={region_snap.input.precision:.3f} "
            f"sparse={region_snap.input.sparseness:.3f} "
            f"dim={region_snap.association.eff_dim:.1f}"
        )


if __name__ == "__main__":
    main()
