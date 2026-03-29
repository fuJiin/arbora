#!/usr/bin/env python3
"""Visualize STEP agent navigating MiniGrid. Saves animated GIF.

Usage:
    uv run experiments/scripts/minigrid/visualize.py
    uv run experiments/scripts/minigrid/visualize.py --episodes 5 --out agent.gif
    uv run experiments/scripts/minigrid/visualize.py --random  # random baseline
"""

from __future__ import annotations

import argparse

import gymnasium as gym
import minigrid  # noqa: F401
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from step.agent.minigrid import MiniGridAgent
from step.cortex import SensoryRegion
from step.cortex.basal_ganglia import BasalGanglia
from step.cortex.circuit import Circuit, ConnectionRole
from step.cortex.motor import MotorRegion
from step.encoders.minigrid import MiniGridEncoder
from step.environment.minigrid import MiniGridObs

ACTION_NAMES = ["left", "right", "fwd", "pick", "drop", "toggle", "done"]


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


def annotate_frame(
    frame: np.ndarray,
    step: int,
    episode: int,
    action: int,
    reward: float,
    conf: float,
    epsilon: float,
    is_random: bool,
) -> Image.Image:
    """Add text overlay to a rendered frame."""
    # Scale up for readability
    img = Image.fromarray(frame)
    img = img.resize((320, 320), Image.NEAREST)
    draw = ImageDraw.Draw(img)

    # Build overlay text
    action_str = ACTION_NAMES[action] if 0 <= action < 7 else "?"
    source = "rand" if is_random else f"M1 ({conf:.2f})"
    lines = [
        f"ep={episode} t={step}",
        f"act={action_str} [{source}]",
    ]
    if reward > 0:
        lines.append(f"REWARD={reward:.2f}")

    # Draw text with background
    y = 2
    for line in lines:
        bbox = draw.textbbox((0, 0), line)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([2, y, w + 6, y + h + 4], fill=(0, 0, 0, 180))
        draw.text((4, y + 2), line, fill=(255, 255, 255))
        y += h + 6

    return img


def run_and_capture(
    n_episodes: int = 3,
    use_random: bool = False,
    pretrain_episodes: int = 200,
    seed: int = 0,
) -> list[Image.Image]:
    """Run agent and capture frames. Optionally pretrain first."""
    gym_env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
    rng = np.random.default_rng(seed)
    frames: list[Image.Image] = []

    encoder = MiniGridEncoder()
    circuit = build_circuit(encoder)
    agent = MiniGridAgent(encoder=encoder, circuit=circuit)
    m1 = circuit.region("M1")

    # Pretrain (no rendering)
    if pretrain_episodes > 0 and not use_random:
        print(f"  Pretraining for {pretrain_episodes} episodes...")
        from step.environment.minigrid import MiniGridEnv
        from step.harness.minigrid.train import MiniGridHarness

        pre_env = MiniGridEnv(
            "MiniGrid-Empty-5x5-v0",
            max_episodes=pretrain_episodes,
            seed=seed,
        )
        harness = MiniGridHarness(pre_env, agent, log_interval=10000)
        harness.run()
        print(f"  Pretrained. Epsilon now: {agent._epsilon:.3f}")

    # Capture episodes
    for ep in range(n_episodes):
        obs_dict, _ = gym_env.reset(seed=seed + pretrain_episodes + ep)
        obs = MiniGridObs(
            image=obs_dict["image"],
            direction=int(obs_dict["direction"]),
            mission=obs_dict.get("mission", ""),
        )
        agent.reset()
        step_count = 0
        done = False

        while not done:
            frame = gym_env.render()

            if use_random:
                action = int(rng.integers(7))
                conf = 0.0
                is_random_action = True
            else:
                agent.step(obs)
                # Check if this will be random (epsilon) or M1
                old_epsilon = agent._epsilon
                action = agent.decode_action()
                is_random_action = agent._epsilon < old_epsilon  # epsilon decayed = was random
                _, conf = m1.get_population_output()

            obs_dict, reward, terminated, truncated, _ = gym_env.step(action)
            obs = MiniGridObs(
                image=obs_dict["image"],
                direction=int(obs_dict["direction"]),
                mission=obs_dict.get("mission", ""),
            )

            if not use_random and reward != 0.0:
                agent.apply_reward(float(reward))

            img = annotate_frame(
                frame, step_count, ep, action, float(reward),
                conf, agent._epsilon if not use_random else 1.0,
                is_random_action,
            )
            frames.append(img)
            step_count += 1
            done = terminated or truncated

            if step_count > 100:
                break

        # Add final frame
        frame = gym_env.render()
        img = annotate_frame(
            frame, step_count, ep, -1, float(reward),
            0.0, 0.0, False,
        )
        # Flash green on success, red on failure
        if terminated:
            overlay = Image.new("RGBA", img.size, (0, 255, 0, 60))
            img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
        frames.append(img)
        # Hold final frame
        for _ in range(10):
            frames.append(img)

    gym_env.close()
    return frames


def save_gif(frames: list[Image.Image], path: str, fps: int = 8):
    """Save frames as animated GIF."""
    if not frames:
        print("No frames to save")
        return
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0,
    )
    print(f"Saved {len(frames)} frames to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--pretrain", type=int, default=500)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=6)
    args = parser.parse_args()

    mode = "random" if args.random else "step"
    out = args.out or f"minigrid_{mode}.gif"

    print(f"Capturing {args.episodes} episodes ({mode} agent)...")
    frames = run_and_capture(
        n_episodes=args.episodes,
        use_random=args.random,
        pretrain_episodes=args.pretrain if not args.random else 0,
        seed=args.seed,
    )
    save_gif(frames, out, fps=args.fps)


if __name__ == "__main__":
    main()
