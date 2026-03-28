"""MiniGrid environment types and wrapper.

MiniGridObs is the typed observation from MiniGrid gymnasium environments.
MiniGridEnv wraps a gymnasium MiniGrid env for multi-episode training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import gymnasium as gym


@dataclass(frozen=True, slots=True)
class MiniGridObs:
    """Single-step observation from a MiniGrid environment.

    Fields mirror the gymnasium MiniGrid observation dict:
    - image: 7x7x3 symbolic grid (object_type, color, state per cell)
    - direction: agent facing direction (0=right, 1=down, 2=left, 3=up)
    - mission: text instruction (unused for Empty/DoorKey envs)
    """

    image: np.ndarray  # (7, 7, 3) uint8
    direction: int  # 0-3
    mission: str = ""


def _to_obs(gym_obs: dict) -> MiniGridObs:
    """Convert gymnasium dict observation to typed MiniGridObs."""
    return MiniGridObs(
        image=gym_obs["image"],
        direction=int(gym_obs["direction"]),
        mission=gym_obs.get("mission", ""),
    )


class MiniGridEnv:
    """Wraps a gymnasium MiniGrid environment for multi-episode training.

    Conforms to the Environment protocol (reset/step/done). Runs
    ``max_episodes`` episodes in sequence, auto-resetting the underlying
    gym env between episodes.

    Usage::

        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=100)
        obs = env.reset()
        while not env.done:
            obs, reward = env.step(action)
            if obs is None:          # episode boundary
                ...

    Args:
        env_id: Gymnasium environment ID.
        max_episodes: Training ends after this many episodes.
        seed: Random seed for the underlying environment.
    """

    def __init__(
        self,
        env_id: str = "MiniGrid-Empty-5x5-v0",
        *,
        max_episodes: int = 100,
        seed: int = 0,
    ):
        import gymnasium as gym
        import minigrid  # noqa: F401 — registers envs

        self._gym_env: gym.Env = gym.make(env_id)
        self._max_episodes = max_episodes
        self._seed = seed

        self._episode_count = 0
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._terminated = False
        self._truncated = False
        self._done = False

    @property
    def done(self) -> bool:
        """True when max_episodes have been completed."""
        return self._done

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def episode_steps(self) -> int:
        return self._episode_steps

    @property
    def episode_reward(self) -> float:
        return self._episode_reward

    def reset(self) -> MiniGridObs:
        """Reset the gym env and return the first observation."""
        gym_obs, _info = self._gym_env.reset(seed=self._seed + self._episode_count)
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._terminated = False
        self._truncated = False
        return _to_obs(gym_obs)

    def step(self, action: int) -> tuple[MiniGridObs, float]:
        """Step the gym env. Returns (obs, reward).

        When an episode ends (terminated or truncated), this method
        increments the episode counter and auto-resets. The returned
        obs is from the new episode.
        """
        gym_obs, reward, terminated, truncated, _info = self._gym_env.step(action)
        self._episode_steps += 1
        self._episode_reward += float(reward)

        if terminated or truncated:
            self._episode_count += 1
            if self._episode_count >= self._max_episodes:
                self._done = True
            self._terminated = terminated
            self._truncated = truncated
            # Auto-reset for next episode
            if not self._done:
                gym_obs, _info = self._gym_env.reset(
                    seed=self._seed + self._episode_count
                )
                self._episode_steps = 0
                self._episode_reward = 0.0

        return _to_obs(gym_obs), float(reward)

    @property
    def last_episode_terminated(self) -> bool:
        """True if the most recent episode ended by reaching the goal."""
        return self._terminated

    @property
    def last_episode_truncated(self) -> bool:
        """True if the most recent episode was truncated (max steps)."""
        return self._truncated
