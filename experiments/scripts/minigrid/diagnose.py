#!/usr/bin/env python3
"""Diagnose why STEP agent underperforms random on MiniGrid."""

from __future__ import annotations

from collections import Counter

import numpy as np

from step.agent.minigrid import MiniGridAgent
from step.cortex import SensoryRegion
from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.circuit import Circuit, ConnectionRole
from step.cortex.motor import MotorRegion
from step.encoders.minigrid import MiniGridEncoder
from step.environment.minigrid import MiniGridEnv


def build_circuit(encoder: MiniGridEncoder) -> Circuit:
    s1 = SensoryRegion(
        input_dim=encoder.input_dim,
        encoding_width=encoder.encoding_width,
        n_columns=64, n_l4=4, n_l23=4, n_l5=0,
        k_columns=4, seed=42,
    )
    m1 = MotorRegion(
        input_dim=s1.n_l23_total,
        n_columns=16, n_l4=0, n_l23=4,
        k_columns=2, n_output_tokens=7, seed=456,
    )
    bg = BasalGanglia(context_dim=s1.n_columns + 1, seed=789)
    circuit = Circuit(encoder)
    circuit.add_region("S1", s1, entry=True)
    circuit.add_region("M1", m1, basal_ganglia=bg)
    circuit.connect(s1.l23, m1.input_lamina, ConnectionRole.FEEDFORWARD)
    circuit.finalize()
    return circuit


def main():
    n_episodes = 50
    encoder = MiniGridEncoder()
    circuit = build_circuit(encoder)
    env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=n_episodes, seed=0)
    agent = MiniGridAgent(encoder=encoder, circuit=circuit)

    # Get M1 region
    m1 = circuit.region("M1")
    bg = circuit._regions["M1"].basal_ganglia

    obs = env.reset()
    agent.reset()

    action_counts = Counter()
    m1_outputs = []  # (token_id, confidence)
    bg_gates = []
    purposeful_count = 0
    random_fallback_count = 0
    total_steps = 0
    last_ep = 0

    while not env.done:
        agent.step(obs)

        # Check M1 output before decode_action
        m_id, conf = m1.get_population_output()
        m1_outputs.append((m_id, conf))
        if m1.last_gate is not None:
            bg_gates.append(m1.last_gate)

        action = agent.decode_action()
        action_counts[action] += 1
        if m_id >= 0:
            purposeful_count += 1
        else:
            random_fallback_count += 1

        obs, reward = env.step(action)
        total_steps += 1

        if reward != 0.0:
            agent.apply_reward(reward)

        if env.episode_count > last_ep:
            last_ep = env.episode_count
            agent.reset()

    # Analysis
    print(f"=== Diagnosis: {n_episodes} episodes, {total_steps} steps ===\n")

    print(f"Purposeful (M1 output): {purposeful_count}/{total_steps} = {purposeful_count/total_steps:.1%}")
    print(f"Random fallback:        {random_fallback_count}/{total_steps} = {random_fallback_count/total_steps:.1%}")

    print(f"\nAction distribution:")
    action_names = ["left", "right", "forward", "pickup", "drop", "toggle", "done"]
    for i in range(7):
        pct = action_counts[i] / total_steps * 100
        bar = "█" * int(pct / 2)
        print(f"  {i} ({action_names[i]:>7}): {action_counts[i]:>5} ({pct:5.1f}%) {bar}")

    print(f"\nM1 confidence stats:")
    confs = [c for _, c in m1_outputs if c > 0]
    if confs:
        print(f"  Mean: {np.mean(confs):.4f}")
        print(f"  Max:  {np.max(confs):.4f}")
        print(f"  Min:  {np.min(confs):.4f}")
    else:
        print(f"  No positive confidence outputs")

    m1_action_counts = Counter()
    for m_id, _ in m1_outputs:
        if m_id >= 0:
            m1_action_counts[m_id] += 1
    if m1_action_counts:
        print(f"\nM1 action distribution (when purposeful):")
        for i in range(7):
            if m1_action_counts[i] > 0:
                print(f"  {i} ({action_names[i]:>7}): {m1_action_counts[i]}")

    if bg_gates:
        print(f"\nBG gate stats:")
        print(f"  Mean: {np.mean(bg_gates):.4f}")
        print(f"  Std:  {np.std(bg_gates):.4f}")
        print(f"  Min:  {np.min(bg_gates):.4f}")
        print(f"  Max:  {np.max(bg_gates):.4f}")

    print(f"\nM1 output_weights stats:")
    print(f"  Shape: {m1.output_weights.shape}")
    print(f"  Mean:  {m1.output_weights.mean():.6f}")
    print(f"  Max:   {m1.output_weights.max():.6f}")
    print(f"  Nonzero: {(m1.output_weights != 0).sum()}/{m1.output_weights.size}")

    print(f"\nM1 ff_weights stats:")
    print(f"  Shape: {m1.ff_weights.shape}")
    print(f"  Mean:  {m1.ff_weights.mean():.6f}")
    print(f"  Max:   {m1.ff_weights.max():.6f}")

    # Check output threshold
    print(f"\nM1 output_threshold: {m1.output_threshold}")
    print(f"M1 output_scores (last step): {m1.output_scores}")


if __name__ == "__main__":
    main()
