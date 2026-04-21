"""Tests for CanaryTracker — non-destructive CA3 retention measurement."""

import numpy as np
import pytest

from arbora.hippocampus import HippocampalRegion
from arbora.probes import CanaryTracker


def _hc(input_dim: int = 64, seed: int = 0) -> HippocampalRegion:
    return HippocampalRegion(
        input_dim=input_dim,
        ec_dim=200,
        dg_dim=800,
        ca3_dim=200,
        seed=seed,
    )


def _obs(seed: int, dim: int = 64) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=dim)


class TestInit:
    def test_rejects_empty_canaries(self):
        with pytest.raises(ValueError):
            CanaryTracker(_hc(), [])

    def test_primes_each_canary_and_snapshots_ca3_state(self):
        hc = _hc()
        canaries = [_obs(0), _obs(1), _obs(2)]
        tracker = CanaryTracker(hc, canaries)
        assert len(tracker.initial_states) == 3
        # Each snapshot is a bool array matching CA3 dim.
        for s in tracker.initial_states:
            assert s.dtype == np.bool_
            assert s.shape == (hc.ca3.dim,)

    def test_canaries_are_copied(self):
        hc = _hc()
        obs = _obs(0)
        tracker = CanaryTracker(hc, [obs])
        obs[0] = 999.0  # mutate external
        assert tracker.canaries[0][0] != 999.0


class TestMeasure:
    def test_immediate_measure_returns_near_one(self):
        """Measuring right after setup should show near-perfect retention."""
        hc = _hc()
        tracker = CanaryTracker(hc, [_obs(0), _obs(1), _obs(2)])
        overlaps = tracker.measure()
        assert len(overlaps) == 3
        # Deterministic pipeline: re-encoding a canary immediately after
        # priming should yield identical CA3 state when measurement is
        # truly non-destructive.
        for o in overlaps:
            assert o == pytest.approx(1.0, abs=0.01)

    def test_measure_does_not_modify_learned_state(self):
        """Non-destructive: lateral_weights/state must be byte-identical
        before and after a measure() call."""
        hc = _hc()
        tracker = CanaryTracker(hc, [_obs(0), _obs(1)])

        # Do some training-like activity so there's non-trivial state.
        for i in range(5):
            hc.process(_obs(100 + i))

        lat_before = hc.ca3.lateral_weights.copy()
        state_before = hc.ca3.state.copy()
        match_before = hc.last_match
        ec_before = hc.last_ec_pattern.copy()
        dg_before = hc.last_dg_pattern.copy()
        out_before = hc.output_port.firing_rate.copy()

        _ = tracker.measure()

        np.testing.assert_array_equal(hc.ca3.lateral_weights, lat_before)
        np.testing.assert_array_equal(hc.ca3.state, state_before)
        assert hc.last_match == match_before
        np.testing.assert_array_equal(hc.last_ec_pattern, ec_before)
        np.testing.assert_array_equal(hc.last_dg_pattern, dg_before)
        np.testing.assert_array_equal(hc.output_port.firing_rate, out_before)

    def test_retention_degrades_after_reset_memory(self):
        """Wiping CA3 memory should collapse canary retention."""
        hc = _hc()
        tracker = CanaryTracker(hc, [_obs(0), _obs(1)])
        overlaps_pre = tracker.measure()

        hc.reset_memory()
        overlaps_post = tracker.measure()

        # After reset_memory, CA3 lateral_weights are zero. retrieve()
        # with no learned weights won't reproduce the bound pattern;
        # overlaps should drop.
        assert all(o == pytest.approx(1.0, abs=0.01) for o in overlaps_pre)
        assert all(o < 0.99 for o in overlaps_post)

    def test_measure_is_idempotent(self):
        """Two consecutive measures should return identical overlaps."""
        hc = _hc()
        tracker = CanaryTracker(hc, [_obs(0), _obs(1), _obs(2)])
        a = tracker.measure()
        b = tracker.measure()
        assert a == b
