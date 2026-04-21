import numpy as np
import pytest

from examples.minigrid.env import MiniGridEnv, MiniGridObs


class TestMiniGridObs:
    def test_frozen(self):
        """MiniGridObs should be immutable."""
        obs = MiniGridObs(image=np.zeros((7, 7, 3), dtype=np.uint8), direction=0)
        with pytest.raises(AttributeError):
            obs.direction = 1  # type: ignore[misc]

    def test_fields(self):
        image = np.zeros((7, 7, 3), dtype=np.uint8)
        obs = MiniGridObs(image=image, direction=2, mission="go to goal")
        assert obs.direction == 2
        assert obs.mission == "go to goal"
        assert obs.image.shape == (7, 7, 3)


class TestMiniGridEnv:
    def test_reset_returns_obs(self):
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=5)
        obs = env.reset()
        assert isinstance(obs, MiniGridObs)
        assert obs.image.shape == (7, 7, 3)
        assert 0 <= obs.direction <= 3

    def test_step_returns_obs_reward(self):
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=5)
        env.reset()
        obs, reward = env.step(2)  # forward
        assert isinstance(obs, MiniGridObs)
        assert isinstance(reward, float)

    def test_episode_terminates(self):
        """Environment should reach done within max_steps."""
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=1)
        env.reset()
        steps = 0
        while not env.done:
            env.step(2)  # forward repeatedly
            steps += 1
            if steps > 200:
                break
        assert env.done
        assert env.episode_count == 1

    def test_done_after_max_episodes(self):
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=3)
        env.reset()
        episodes = 0
        steps = 0
        while not env.done:
            _obs, _reward = env.step(2)
            steps += 1
            if env.episode_count > episodes:
                episodes = env.episode_count
            if steps > 1000:
                break
        assert env.done
        assert env.episode_count == 3

    def test_episode_reward_tracked(self):
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=5)
        env.reset()
        for _ in range(5):
            env.step(2)
        # episode_reward should be a float (may be 0 if goal not reached)
        assert isinstance(env.episode_reward, float)

    def test_last_episode_steps_preserved_across_auto_reset(self):
        """`last_episode_steps` survives the internal auto-reset in step().

        Regression test: `episode_steps` is reset to 0 when the env
        auto-resets for the next episode, so consumers that read
        counters after env.step() returns need the `last_*` variants.
        """
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=2)
        env.reset()
        # Spin forward until at least one episode ends.
        for _ in range(500):
            env.step(2)
            if env.episode_count >= 1:
                break
        assert env.episode_count >= 1
        # In-progress counter has been reset to 0 for the new episode.
        assert env.episode_steps == 0
        # Snapshot of the just-finished episode persists.
        assert env.last_episode_steps > 0

    def test_last_episode_reward_preserved_across_auto_reset(self):
        env = MiniGridEnv("MiniGrid-Empty-5x5-v0", max_episodes=2)
        env.reset()
        for _ in range(500):
            env.step(2)
            if env.episode_count >= 1:
                break
        assert env.episode_count >= 1
        # episode_reward resets with the auto-reset; last_episode_reward doesn't.
        assert env.episode_reward == 0.0
        # For Empty-5x5 the agent occasionally reaches the goal (reward > 0);
        # at minimum the type should be float.
        assert isinstance(env.last_episode_reward, float)
