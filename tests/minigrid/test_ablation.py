"""Smoke tests for the HC v1 ablation runner (ARB-118)."""

from examples.minigrid.ablation import (
    AblationResult,
    ArmResult,
    EpisodeEvent,
    EpisodeProbe,
    run_ablation,
    run_arm,
)
from examples.minigrid.presets import (
    build_baseline_circuit,
    build_hippocampal_circuit,
)


class TestEpisodeProbe:
    def test_records_events(self):
        probe = EpisodeProbe()
        probe.episode_end(success=True, steps=10, reward=0.5)
        probe.episode_end(success=False, steps=100, reward=0.0)
        assert len(probe.events) == 2
        assert probe.events[0] == EpisodeEvent(True, 10, 0.5)
        assert probe.events[1] == EpisodeEvent(False, 100, 0.0)

    def test_boundary_and_observe_are_noop(self):
        """Duck-typed by MiniGridHarness; must not raise."""
        probe = EpisodeProbe()
        probe.observe(None, step=0)
        probe.boundary()
        assert probe.events == []

    def test_snapshot_returns_events(self):
        probe = EpisodeProbe()
        probe.episode_end(True, 5, 1.0)
        snap = probe.snapshot()
        assert snap["events"] == probe.events


class TestArmResult:
    def test_success_rate(self):
        r = ArmResult(
            name="x",
            seed=0,
            events=[
                EpisodeEvent(True, 10, 0.5),
                EpisodeEvent(False, 100, 0.0),
                EpisodeEvent(True, 20, 0.9),
            ],
        )
        assert abs(r.success_rate - 2 / 3) < 1e-9

    def test_success_rate_empty(self):
        assert ArmResult(name="x", seed=0).success_rate == 0.0

    def test_time_to_first_success_returns_episode_index(self):
        r = ArmResult(
            name="x",
            seed=0,
            events=[
                EpisodeEvent(False, 100, 0.0),
                EpisodeEvent(False, 100, 0.0),
                EpisodeEvent(True, 30, 0.7),
            ],
        )
        assert r.time_to_first_success == 3

    def test_time_to_first_success_none_if_no_success(self):
        r = ArmResult(
            name="x",
            seed=0,
            events=[EpisodeEvent(False, 100, 0.0)],
        )
        assert r.time_to_first_success is None

    def test_mean_steps(self):
        r = ArmResult(
            name="x",
            seed=0,
            events=[EpisodeEvent(False, 10, 0.0), EpisodeEvent(True, 20, 1.0)],
        )
        assert r.mean_steps == 15.0


class TestRunArm:
    def test_baseline_runs_to_completion(self):
        """Baseline arm runs a few episodes on the smallest MiniGrid env."""
        r = run_arm(
            "baseline",
            build_baseline_circuit,
            env_id="MiniGrid-Empty-5x5-v0",
            episodes=2,
            seed=0,
        )
        assert r.name == "baseline"
        assert r.seed == 0
        assert len(r.events) >= 1  # episodes may be cut off mid-way

    def test_hippocampal_runs_to_completion(self):
        """With-HC arm runs a few episodes end-to-end without crashing.

        Load-bearing smoke test for ARB-116 + ARB-117 + ARB-118: the full
        HC pipeline (EC → DG → CA3 → CA1 → EC.reverse) plugged into a
        Circuit with BG and M1 must survive a real training loop.
        """
        r = run_arm(
            "hippocampal",
            build_hippocampal_circuit,
            env_id="MiniGrid-Empty-5x5-v0",
            episodes=2,
            seed=0,
        )
        assert r.name == "hippocampal"
        assert len(r.events) >= 1


class TestRunAblation:
    def test_small_ablation_produces_both_arms(self):
        """Runs 1 seed x 2 episodes of each arm on MiniGrid-Empty-5x5."""
        result = run_ablation(
            "MiniGrid-Empty-5x5-v0",
            n_seeds=1,
            episodes_per_seed=2,
            verbose=False,
        )
        assert isinstance(result, AblationResult)
        assert len(result.baseline) == 1
        assert len(result.hippocampal) == 1
        assert result.baseline[0].name == "baseline"
        assert result.hippocampal[0].name == "hippocampal"

    def test_format_table_renders(self):
        result = run_ablation(
            "MiniGrid-Empty-5x5-v0",
            n_seeds=1,
            episodes_per_seed=2,
            verbose=False,
        )
        text = result.format_table()
        assert "baseline" in text
        assert "hippocampal" in text
        assert "MiniGrid-Empty-5x5-v0" in text
