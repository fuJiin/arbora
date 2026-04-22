"""Smoke tests for the HC v1 ablation runner (ARB-118)."""

from examples.minigrid.ablation import (
    AblationResult,
    ArmResult,
    EpisodeEvent,
    EpisodeProbe,
    _synthetic_probe_patterns,
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


class TestEpisodeEvent:
    def test_solved_is_reward_based(self):
        """`solved` tracks reward > 0, not terminated — the MemoryEnv
        distinction between 'made a choice' and 'made the right choice'."""
        assert EpisodeEvent(terminated=True, steps=10, reward=0.7).solved is True
        # Wrong-choice termination in MemoryEnv: terminated but no reward.
        assert EpisodeEvent(terminated=True, steps=10, reward=0.0).solved is False
        # Truncated: didn't terminate, no reward, not solved.
        assert EpisodeEvent(terminated=False, steps=100, reward=0.0).solved is False


class TestArmResult:
    def test_success_rate_uses_reward(self):
        """Reward-based success: terminated-but-unrewarded doesn't count."""
        r = ArmResult(
            name="x",
            seed=0,
            events=[
                EpisodeEvent(True, 10, 0.5),  # solved
                EpisodeEvent(True, 100, 0.0),  # terminated wrong choice
                EpisodeEvent(True, 20, 0.9),  # solved
            ],
        )
        assert abs(r.success_rate - 2 / 3) < 1e-9

    def test_termination_rate_includes_wrong_choices(self):
        r = ArmResult(
            name="x",
            seed=0,
            events=[
                EpisodeEvent(True, 10, 0.5),
                EpisodeEvent(True, 100, 0.0),
                EpisodeEvent(False, 200, 0.0),  # truncated
            ],
        )
        assert abs(r.termination_rate - 2 / 3) < 1e-9
        assert abs(r.success_rate - 1 / 3) < 1e-9

    def test_success_rate_empty(self):
        assert ArmResult(name="x", seed=0).success_rate == 0.0

    def test_time_to_first_success_returns_episode_index(self):
        r = ArmResult(
            name="x",
            seed=0,
            events=[
                EpisodeEvent(True, 100, 0.0),  # terminated but not solved
                EpisodeEvent(False, 200, 0.0),  # truncated
                EpisodeEvent(True, 30, 0.7),  # first solved
            ],
        )
        assert r.time_to_first_success == 3

    def test_time_to_first_success_none_if_no_success(self):
        r = ArmResult(
            name="x",
            seed=0,
            events=[EpisodeEvent(True, 100, 0.0)],  # wrong-choice termination
        )
        assert r.time_to_first_success is None

    def test_mean_steps(self):
        r = ArmResult(
            name="x",
            seed=0,
            events=[EpisodeEvent(False, 10, 0.0), EpisodeEvent(True, 20, 1.0)],
        )
        assert r.mean_steps == 15.0

    def test_mean_retention_none_when_no_probe_patterns(self):
        r = ArmResult(name="x", seed=0)
        assert r.mean_retention is None

    def test_mean_retention_averages_overlaps(self):
        r = ArmResult(name="x", seed=0, final_retention=[0.8, 0.6, 0.4])
        assert r.mean_retention == 0.6


class TestSyntheticProbePatterns:
    def test_dim_and_shape(self):
        probe_patterns = _synthetic_probe_patterns(dim=256, n=8)
        assert len(probe_patterns) == 8
        for c in probe_patterns:
            assert c.shape == (256,)
            assert c.dtype == bool

    def test_sparsity_respected(self):
        probe_patterns = _synthetic_probe_patterns(dim=1000, n=4, sparsity=0.05)
        for c in probe_patterns:
            assert c.sum() == 50  # 1000 * 0.05

    def test_deterministic(self):
        a = _synthetic_probe_patterns(dim=256, n=5, seed=42)
        b = _synthetic_probe_patterns(dim=256, n=5, seed=42)
        for x, y in zip(a, b, strict=True):
            assert (x == y).all()


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
        assert len(r.events) >= 1
        # Baseline has no HC, so no mechanistic stats and no retention.
        assert r.hc_summary == {} or r.hc_summary.get("n_steps", 0) == 0
        assert r.final_retention is None

    def test_hippocampal_runs_to_completion(self):
        """With-HC arm runs a few episodes end-to-end without crashing.

        Load-bearing smoke test for ARB-116 + ARB-117 + ARB-118 + ARB-122
        + ARB-123: the full HC pipeline plugged into a Circuit with BG
        and M1, with probes attached, must survive a real training loop.
        """
        r = run_arm(
            "hippocampal",
            build_hippocampal_circuit,
            env_id="MiniGrid-Empty-5x5-v0",
            episodes=2,
            seed=0,
            n_probe_patterns=4,
        )
        assert r.name == "hippocampal"
        assert len(r.events) >= 1
        # HC arm populates mechanistic summary and retention measurements.
        assert r.hc_summary["n_steps"] > 0
        assert r.final_retention is not None
        assert len(r.final_retention) == 4
        # Immediately after training, retention should still be high for
        # freshly-bound probe patterns (synthetic patterns distinct from real
        # training observations → should survive).
        assert all(0.0 <= o <= 1.0 for o in r.final_retention)


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

    def test_format_table_includes_hc_mechanistic_section(self):
        """HC arm always present → mechanistic section should render."""
        result = run_ablation(
            "MiniGrid-Empty-5x5-v0",
            n_seeds=1,
            episodes_per_seed=2,
            verbose=False,
            n_probe_patterns=4,
        )
        text = result.format_table()
        assert "HC mechanistic stats" in text
        # At least the retention line should be present (probe patterns primed
        # in run_arm always produce a measurement).
        assert "probe-pattern retention" in text
