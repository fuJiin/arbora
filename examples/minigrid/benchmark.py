#!/usr/bin/env python3
"""Benchmark: STEP agent vs random baseline on MiniGrid-Empty-5x5.

Tracks per-episode success rate and step count to see if reward
actually drives learning.

Usage:
    uv run experiments/scripts/minigrid/benchmark.py
    uv run experiments/scripts/minigrid/benchmark.py --episodes 2000
"""

from __future__ import annotations

import argparse

import numpy as np

from arbora.basal_ganglia import BasalGangliaRegion
from arbora.cortex import SensoryRegion
from arbora.cortex.circuit import Circuit, ConnectionRole
from arbora.cortex.motor import MotorRegion
from examples.minigrid.agent import MiniGridAgent
from examples.minigrid.encoder import MiniGridEncoder
from examples.minigrid.env import MiniGridEnv


def build_circuit(encoder: MiniGridEncoder) -> Circuit:
    s1 = SensoryRegion(
        input_dim=encoder.input_dim,
        encoding_width=encoder.encoding_width,
        n_columns=64,
        n_l4=4,
        n_l23=4,
        n_l5=0,
        k_columns=4,
        seed=42,
    )
    m1 = MotorRegion(
        input_dim=s1.n_l23_total,
        n_columns=16,
        n_l4=0,
        n_l23=4,
        k_columns=2,
        n_output_tokens=7,
        seed=456,
    )
    bg = BasalGangliaRegion(input_dim=s1.n_l23_total, n_actions=7, seed=789)
    circuit = Circuit(encoder)
    circuit.add_region("S1", s1, entry=True)
    circuit.add_region("BG", bg)
    circuit.add_region("M1", m1)
    circuit.connect(s1.output_port, bg.input_port, ConnectionRole.FEEDFORWARD)
    circuit.connect(s1.output_port, m1.input_port, ConnectionRole.FEEDFORWARD)
    circuit.connect(bg.output_port, m1.input_port, ConnectionRole.MODULATORY)
    circuit.finalize()
    return circuit


def run_random(n_episodes: int, seed: int = 0) -> dict:
    env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=n_episodes, seed=seed)
    rng = np.random.default_rng(seed)
    env.reset()
    successes = []
    steps_list = []
    current_steps = 0
    while not env.done:
        _obs, _ = env.step(int(rng.integers(7)))
        current_steps += 1
        if env.episode_count > len(successes):
            successes.append(env.last_episode_terminated)
            steps_list.append(current_steps)
            current_steps = 0
    return {"successes": successes, "steps": steps_list}


def run_step(n_episodes: int, seed: int = 0) -> dict:
    encoder = MiniGridEncoder()
    circuit = build_circuit(encoder)
    env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=n_episodes, seed=seed)
    agent = MiniGridAgent(encoder=encoder, circuit=circuit)
    obs = env.reset()
    agent.reset()
    successes = []
    steps_list = []
    current_steps = 0
    last_ep = 0
    while not env.done:
        agent.step(obs)
        action = agent.decode_action()
        obs, reward = env.step(action)
        current_steps += 1
        if reward != 0.0:
            agent.apply_reward(reward)
        if env.episode_count > last_ep:
            successes.append(env.last_episode_terminated)
            steps_list.append(current_steps)
            current_steps = 0
            last_ep = env.episode_count
            agent.reset()
    return {"successes": successes, "steps": steps_list}


def rolling(values: list, window: int = 50) -> list[float]:
    rates = []
    for i in range(len(values)):
        chunk = values[max(0, i - window + 1) : i + 1]
        rates.append(sum(chunk) / len(chunk))
    return rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    args = parser.parse_args()
    n = args.episodes

    print(f"Benchmark: {n} episodes on MiniGrid-Empty-5x5-v0\n")

    print("Random baseline...")
    rand = run_random(n)
    print("STEP agent...")
    step = run_step(n)

    rand_rates = rolling(rand["successes"])
    step_rates = rolling(step["successes"])
    rand_steps = rolling(rand["steps"])
    step_steps = rolling(step["steps"])

    header = (
        f"\n{'Episode':>10} {'Rand %':>8} {'STEP %':>8}"
        f" {'Delta':>8} {'Rand steps':>12} {'STEP steps':>12}"
    )
    print(header)
    print("-" * 65)
    checkpoints = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000]
    for ep in checkpoints:
        if ep <= n:
            r = rand_rates[ep - 1]
            s = step_rates[ep - 1]
            rs = rand_steps[ep - 1]
            ss = step_steps[ep - 1]
            print(
                f"{ep:>10} {r:>8.1%} {s:>8.1%} {s - r:>+8.1%} {rs:>12.1f} {ss:>12.1f}"
            )

    rs = sum(rand["successes"])
    ss = sum(step["successes"])
    print(f"\nTotal: Random={rs}/{n} ({rs / n:.1%})  STEP={ss}/{n} ({ss / n:.1%})")
    rm = np.mean(rand["steps"])
    sm = np.mean(step["steps"])
    print(f"Mean steps: Random={rm:.1f}  STEP={sm:.1f}")


if __name__ == "__main__":
    main()
