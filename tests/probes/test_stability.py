"""Tests for CortexStabilityTracker — non-destructive L2/3 drift measurement."""

import numpy as np
import pytest

from arbora.cortex import SensoryRegion
from arbora.probes import CortexStabilityTracker


def _s1(input_dim: int = 984, seed: int = 42) -> SensoryRegion:
    """Small S1 matching the MiniGrid ablation's shared config."""
    return SensoryRegion(
        input_dim=input_dim,
        encoding_width=20,
        n_columns=64,
        n_l4=4,
        n_l23=4,
        n_l5=0,
        k_columns=4,
        seed=seed,
    )


def _sparse_encoding(seed: int, dim: int = 984, k: int = 148) -> np.ndarray:
    """Synthetic sparse binary encoding at the MiniGrid encoder shape."""
    rng = np.random.default_rng(seed)
    out = np.zeros(dim, dtype=np.bool_)
    out[rng.choice(dim, size=k, replace=False)] = True
    return out


class TestInit:
    def test_rejects_empty_encodings(self):
        with pytest.raises(ValueError):
            CortexStabilityTracker(_s1(), [])

    def test_primes_snapshots_l23_active(self):
        s1 = _s1()
        encodings = [_sparse_encoding(i) for i in range(3)]
        tracker = CortexStabilityTracker(s1, encodings)
        assert len(tracker.initial_states) == 3
        for snap in tracker.initial_states:
            assert snap.dtype == np.bool_
            assert snap.shape == (s1.n_l23_total,)

    def test_encodings_are_copied(self):
        s1 = _s1()
        enc = _sparse_encoding(0)
        tracker = CortexStabilityTracker(s1, [enc])
        enc[0] = True  # mutate external
        # Internal copy should not reflect the mutation.
        assert tracker.encodings[0][0] == _sparse_encoding(0)[0]


class TestNonDestructive:
    def test_priming_does_not_perturb_s1_state(self):
        """Constructing the tracker must leave S1 byte-identical."""
        s1 = _s1()
        # Warm up S1 a bit so it has non-trivial state.
        for i in range(3):
            s1.process(_sparse_encoding(1000 + i))

        # Snapshot all relevant state before tracker construction.
        ff_before = s1.ff_weights.copy()
        l23_fr_before = s1.l23.firing_rate.copy()
        l23_act_before = s1.l23.active.copy()
        l4_fr_before = s1.l4.firing_rate.copy()
        active_cols_before = s1.active_columns.copy()
        bursting_before = s1.bursting_columns.copy()

        encodings = [_sparse_encoding(i) for i in range(3)]
        _ = CortexStabilityTracker(s1, encodings)

        np.testing.assert_array_equal(s1.ff_weights, ff_before)
        np.testing.assert_array_equal(s1.l23.firing_rate, l23_fr_before)
        np.testing.assert_array_equal(s1.l23.active, l23_act_before)
        np.testing.assert_array_equal(s1.l4.firing_rate, l4_fr_before)
        np.testing.assert_array_equal(s1.active_columns, active_cols_before)
        np.testing.assert_array_equal(s1.bursting_columns, bursting_before)

    def test_measure_does_not_perturb_s1_state(self):
        """Repeated measure() calls must leave S1 byte-identical."""
        s1 = _s1()
        encodings = [_sparse_encoding(i) for i in range(3)]
        tracker = CortexStabilityTracker(s1, encodings)

        # Some additional training activity.
        for i in range(5):
            s1.process(_sparse_encoding(500 + i))

        ff_before = s1.ff_weights.copy()
        l23_fr_before = s1.l23.firing_rate.copy()
        l23_act_before = s1.l23.active.copy()

        _ = tracker.measure()

        np.testing.assert_array_equal(s1.ff_weights, ff_before)
        np.testing.assert_array_equal(s1.l23.firing_rate, l23_fr_before)
        np.testing.assert_array_equal(s1.l23.active, l23_act_before)


class TestMeasure:
    def test_immediate_measure_returns_near_one(self):
        """Immediately after setup, re-encoding the references should
        reproduce the exact L2/3 pattern (deterministic + no drift)."""
        s1 = _s1()
        encodings = [_sparse_encoding(i) for i in range(3)]
        tracker = CortexStabilityTracker(s1, encodings)
        overlaps = tracker.measure()
        assert len(overlaps) == 3
        for o in overlaps:
            assert o == pytest.approx(1.0, abs=1e-9)

    def test_drift_collapses_overlap(self):
        """After training S1 on many distinct inputs, the L2/3 response
        to a fixed reference encoding should drift. Overlap drops below 1."""
        s1 = _s1()
        encodings = [_sparse_encoding(i) for i in range(3)]
        tracker = CortexStabilityTracker(s1, encodings)

        # Train S1 aggressively on unrelated inputs to force drift.
        for i in range(200):
            s1.process(_sparse_encoding(1000 + i))

        overlaps = tracker.measure()
        # At least one of the references should show non-trivial drift.
        assert min(overlaps) < 1.0

    def test_measure_is_idempotent(self):
        """Two consecutive measures should return identical overlaps."""
        s1 = _s1()
        encodings = [_sparse_encoding(i) for i in range(3)]
        tracker = CortexStabilityTracker(s1, encodings)
        # Train a bit so overlaps aren't trivially 1.0.
        for i in range(50):
            s1.process(_sparse_encoding(500 + i))
        a = tracker.measure()
        b = tracker.measure()
        assert a == b
