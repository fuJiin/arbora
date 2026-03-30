"""MiniGrid training harness with probe-based telemetry.

Usage::

    harness = MiniGridHarness(
        env=MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=100),
        agent=MiniGridAgent(encoder=encoder, circuit=circuit),
        probes=[LaminaProbe()],
    )
    result = harness.run()
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING

from arbor.agent import TrainResult
from arbor.probes.core import LaminaProbe, Probe

if TYPE_CHECKING:
    from examples.minigrid.agent import MiniGridAgent
    from examples.minigrid.env import MiniGridEnv


class MiniGridHarness:
    """Train a Circuit on MiniGrid episodes with probe telemetry.

    Simpler than ChatTrainHarness: no EOM/boundary signaling, no
    decoder training, no turn-taking. The agent always acts, and
    episode boundaries trigger circuit reset.

    Args:
        env: MiniGrid environment wrapper.
        agent: MiniGridAgent wrapping encoder + circuit.
        probes: Probe instances for telemetry.
        log_interval: Print progress every N steps.
    """

    def __init__(
        self,
        env: MiniGridEnv,
        agent: MiniGridAgent,
        *,
        probes: Sequence[Probe] = (),
        log_interval: int = 100,
    ):
        self._env = env
        self._agent = agent
        self._probes = probes
        self._log_interval = log_interval

        # Resolve typed probes for logging
        self._lamina_probe: LaminaProbe | None = None
        for p in probes:
            if isinstance(p, LaminaProbe) and self._lamina_probe is None:
                self._lamina_probe = p

    def run(self) -> TrainResult:
        """Execute the training loop across episodes."""
        env = self._env
        agent = self._agent
        probes = self._probes
        start = time.monotonic()

        obs = env.reset()
        agent.reset()
        t = 0
        last_ep = 0

        while not env.done:
            # Agent: encode + process (motor_active=True always)
            agent.step(obs)

            # Probes observe circuit state
            for probe in probes:
                probe.observe(agent.circuit, step=t)

            # Decode action + step env
            action = agent.decode_action()
            obs, reward = env.step(action)

            # Route reward to circuit (BG + motor regions)
            if reward != 0.0:
                agent.apply_reward(reward)

            # Episode boundary detection (env auto-resets internally)
            episode_ended = env.episode_count > last_ep
            if episode_ended:
                last_ep = env.episode_count
                agent.reset()
                for probe in probes:
                    if hasattr(probe, "boundary"):
                        probe.boundary()
                    if hasattr(probe, "episode_end"):
                        probe.episode_end(
                            success=env.last_episode_terminated,
                            steps=env.episode_steps,
                            reward=env.episode_reward,
                        )

            # Periodic logging
            if t > 0 and t % self._log_interval == 0:
                self._log(t, time.monotonic() - start)

            t += 1

        # Build result
        elapsed = time.monotonic() - start
        result = TrainResult(elapsed_seconds=elapsed)
        for probe in probes:
            result.probe_snapshots[probe.name] = probe.snapshot()

        self._log_final(t, elapsed)
        return result

    def _log(self, t: int, elapsed: float) -> None:
        """Print periodic progress line."""
        env = self._env
        lamina_str = ""
        if self._lamina_probe is not None:
            for _rn, snap in self._lamina_probe.snapshot().items():
                lamina_str = (
                    f"recall={snap.input.recall:.2f} "
                    f"prec={snap.input.precision:.2f} "
                    f"dim={snap.association.eff_dim:.1f}"
                )
                break
        print(f"  t={t:,} ep={env.episode_count} {lamina_str} ({elapsed:.1f}s)")

    def _log_final(self, total_steps: int, elapsed: float) -> None:
        """Print summary at end of training."""
        env = self._env
        print(
            f"  done: {total_steps:,} steps, "
            f"{env.episode_count} episodes ({elapsed:.1f}s)"
        )
