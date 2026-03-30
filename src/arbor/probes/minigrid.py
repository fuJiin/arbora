"""MiniGrid-specific motor probe — episode-level and step-level KPIs.

Top 3 motor KPIs:
1. Episode success rate — rolling success over last N episodes
2. Purposeful ratio — fraction of M1 outputs vs random fallback
3. Output weight consolidation rate — dW after apply_reward

Plus: action distribution, per-episode reward/steps, BG tonic DA.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from arbor.cortex.motor import MotorRegion

if TYPE_CHECKING:
    from arbor.cortex.circuit import Circuit


@dataclass
class MiniGridMotorSnapshot:
    """Per-region motor metrics for MiniGrid environments."""

    episode_successes: list[bool] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)
    episode_steps: list[int] = field(default_factory=list)
    action_counts: dict[int, int] = field(default_factory=dict)
    purposeful_steps: int = 0
    total_steps: int = 0
    consolidation_rates: list[float] = field(default_factory=list)
    tonic_da_history: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Overall success rate."""
        if not self.episode_successes:
            return 0.0
        return sum(self.episode_successes) / len(self.episode_successes)

    @property
    def rolling_success(self, window: int = 50) -> float:
        """Rolling success rate over last `window` episodes."""
        if not self.episode_successes:
            return 0.0
        tail = self.episode_successes[-window:]
        return sum(tail) / len(tail)

    @property
    def purposeful_ratio(self) -> float:
        """Fraction of steps where M1 produced output above threshold."""
        if self.total_steps == 0:
            return 0.0
        return self.purposeful_steps / self.total_steps

    @property
    def mean_consolidation_rate(self) -> float:
        """Mean output weight change per reward event."""
        if not self.consolidation_rates:
            return 0.0
        return float(np.mean(self.consolidation_rates))


class MiniGridMotorProbe:
    """Track MiniGrid-specific motor KPIs.

    Step-level: action distribution, purposeful ratio, BG tonic DA.
    Episode-level: success rate, episode reward, episode length.
    Reward-level: output weight consolidation rate (dW/dR).
    """

    name: str = "minigrid_motor"

    def __init__(self):
        self._action_counts: Counter = Counter()
        self._purposeful_steps = 0
        self._total_steps = 0

        self._episode_successes: list[bool] = []
        self._episode_rewards: list[float] = []
        self._episode_steps: list[int] = []
        self._consolidation_rates: list[float] = []
        self._tonic_da_history: list[float] = []

        # Track output weights for consolidation rate
        self._prev_output_weights: dict[str, np.ndarray] = {}

    def observe(self, circuit: Circuit, **kwargs) -> None:
        """Observe motor state after each step."""
        self._total_steps += 1

        for name, s in circuit._regions.items():
            if not s.motor:
                continue
            if not isinstance(s.region, MotorRegion):
                continue
            motor = s.region

            # Purposeful: did M1 produce output?
            m_id, _conf = motor.last_output
            if m_id >= 0:
                self._purposeful_steps += 1
                self._action_counts[m_id] += 1

            # Snapshot weights before next reward (for consolidation rate)
            if name not in self._prev_output_weights:
                self._prev_output_weights[name] = motor.output_weights.copy()

        # Track BG tonic DA
        from arbor.basal_ganglia import BasalGangliaRegion

        for _name, s in circuit._regions.items():
            if isinstance(s.region, BasalGangliaRegion):
                self._tonic_da_history.append(s.region._tonic_da)

    def episode_end(self, *, success: bool, steps: int, reward: float) -> None:
        """Record episode-level metrics. Called by harness at episode boundary."""
        self._episode_successes.append(success)
        self._episode_rewards.append(reward)
        self._episode_steps.append(steps)

    def boundary(self) -> None:
        """Track consolidation rate at episode boundaries."""
        # Measure weight change since last boundary
        # (apply_reward was called during the episode)
        # Import here to avoid circular
        pass

    def _measure_consolidation(self, circuit: Circuit) -> None:
        """Measure output weight change since last snapshot."""
        for name, s in circuit._regions.items():
            if not s.motor or not isinstance(s.region, MotorRegion):
                continue
            motor = s.region
            if name in self._prev_output_weights:
                dw = np.linalg.norm(
                    motor.output_weights - self._prev_output_weights[name]
                )
                if dw > 0:
                    self._consolidation_rates.append(float(dw))
            self._prev_output_weights[name] = motor.output_weights.copy()

    def snapshot(self) -> dict[str, MiniGridMotorSnapshot]:
        """Return accumulated motor KPIs."""
        return {
            "minigrid": MiniGridMotorSnapshot(
                episode_successes=self._episode_successes.copy(),
                episode_rewards=self._episode_rewards.copy(),
                episode_steps=self._episode_steps.copy(),
                action_counts=dict(self._action_counts),
                purposeful_steps=self._purposeful_steps,
                total_steps=self._total_steps,
                consolidation_rates=self._consolidation_rates.copy(),
                tonic_da_history=self._tonic_da_history.copy(),
            )
        }
