"""Tests for Hebbian learning rule invariants.

Validates the behavioral contracts of _learn_ff_hebbian and
_learn_column_weights: LTP/LTD directions, bounds, masking,
neuromodulation gating, and edge cases.
"""

import numpy as np

from arbor.cortex import CorticalRegion

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_region(seed=42, n_columns=8, n_l4=4, n_l23=4, ff_sparsity=0.4):
    """Create a region with known random state."""
    return CorticalRegion(
        input_dim=32,
        n_columns=n_columns,
        n_l4=n_l4,
        n_l23=n_l23,
        k_columns=3,
        ff_sparsity=ff_sparsity,
        seed=seed,
    )


def _random_hebbian_inputs(region, rng):
    """Generate random but realistic inputs for _learn_ff_hebbian."""
    input_dim = region.input_dim
    n_neurons = region.n_columns * region.n_l4

    flat_input = np.zeros(input_dim)
    active_inputs = rng.choice(input_dim, size=input_dim // 4, replace=False)
    flat_input[active_inputs] = 1.0

    ltp_signal = np.zeros(input_dim)
    trace_inputs = rng.choice(input_dim, size=input_dim // 3, replace=False)
    ltp_signal[trace_inputs] = rng.uniform(0.1, 1.0, size=len(trace_inputs))

    n_winners = region.k_columns
    winner_indices = np.sort(rng.choice(n_neurons, size=n_winners, replace=False))

    return flat_input, ltp_signal, winner_indices


# ---------------------------------------------------------------------------
# _learn_ff_hebbian
# ---------------------------------------------------------------------------


class TestLearnFfHebbian:
    def test_ltp_increases_active_input_weights(self):
        """Winner weights from active inputs should increase."""
        region = _make_region(seed=10)
        rng = np.random.default_rng(10)
        flat_input, ltp_signal, winners = _random_hebbian_inputs(region, rng)

        before = region.ff_weights[:, winners].copy()
        region._learn_ff_hebbian(flat_input, ltp_signal, winners)
        after = region.ff_weights[:, winners]

        # Where ltp_signal > 0 and weight was not already at 1 or masked,
        # weights should generally increase (LTP dominates for active inputs)
        active_mask = ltp_signal > 0.1
        mask = region.ff_mask[:, winners]
        relevant = active_mask[:, np.newaxis] & mask & (before < 0.95)
        if relevant.any():
            assert (after[relevant] >= before[relevant] - 1e-10).all(), (
                "LTP should increase weights from active inputs to winners"
            )

    def test_ltd_decreases_inactive_input_weights(self):
        """Winner weights from inactive inputs should decrease."""
        region = _make_region(seed=20, ff_sparsity=0.0)
        # Use high LTD rate to make effect visible
        region.ltd_rate = 0.1

        flat_input = np.zeros(region.input_dim)
        flat_input[:4] = 1.0  # only 4 inputs active
        ltp_signal = flat_input.copy()
        winners = np.array([0, region.n_l4, 2 * region.n_l4])

        # Set weights to 0.5 so there's room to decrease
        region.ff_weights[:] = 0.5
        region.ff_weights *= region.ff_mask

        before = region.ff_weights[:, winners].copy()
        region._learn_ff_hebbian(flat_input, ltp_signal, winners)
        after = region.ff_weights[:, winners]

        # Inactive inputs (indices >= 4) should see weight decrease
        inactive_slice = slice(4, None)
        mask = region.ff_mask[inactive_slice, :][:, winners]
        decreased = after[inactive_slice][mask] < before[inactive_slice][mask]
        assert decreased.any(), "LTD should decrease weights from inactive inputs"

    def test_subthreshold_ltp_on_non_winners(self):
        """Non-winner neurons should get weak subthreshold LTP."""
        region = _make_region(seed=30, ff_sparsity=0.0)
        rng = np.random.default_rng(30)
        flat_input, ltp_signal, winners = _random_hebbian_inputs(region, rng)

        # Set all weights to 0.3 so there's room to grow
        region.ff_weights[:] = 0.3
        region.ff_weights *= region.ff_mask
        before = region.ff_weights.copy()

        region._learn_ff_hebbian(flat_input, ltp_signal, winners)

        # Non-winner neurons in active input dimensions should increase
        non_winner_mask = np.ones(region.ff_weights.shape[1], dtype=bool)
        non_winner_mask[winners] = False
        active_dims = np.flatnonzero(ltp_signal > 0.01)

        if len(active_dims) > 0:
            diff = (
                region.ff_weights[np.ix_(active_dims, non_winner_mask)]
                - before[np.ix_(active_dims, non_winner_mask)]
            )
            mask = region.ff_mask[np.ix_(active_dims, non_winner_mask)]
            assert (diff[mask] >= -1e-10).all(), (
                "Subthreshold LTP should not decrease non-winner weights"
            )

    def test_weights_clamped_to_unit_interval(self):
        """Weights must remain in [0, 1] after repeated updates."""
        rng = np.random.default_rng(7)
        region = _make_region(seed=7)

        for _ in range(50):
            flat_input, ltp_signal, winners = _random_hebbian_inputs(region, rng)
            region._learn_ff_hebbian(flat_input, ltp_signal, winners)

        assert region.ff_weights.min() >= 0.0
        assert region.ff_weights.max() <= 1.0

    def test_masked_connections_stay_zero(self):
        """Structurally pruned connections must never become nonzero."""
        region = _make_region(ff_sparsity=0.5, seed=3)
        rng = np.random.default_rng(3)
        zero_mask = ~region.ff_mask

        for _ in range(20):
            flat_input, ltp_signal, winners = _random_hebbian_inputs(region, rng)
            region._learn_ff_hebbian(flat_input, ltp_signal, winners)

        np.testing.assert_array_equal(
            region.ff_weights[zero_mask],
            0.0,
            err_msg="Masked connections should stay zero",
        )

    def test_zero_neuromodulation_no_change(self):
        """With surprise_modulator=0, weights should not change."""
        region = _make_region(seed=40)
        region.surprise_modulator = 0.0
        rng = np.random.default_rng(40)
        flat_input, ltp_signal, winners = _random_hebbian_inputs(region, rng)

        before = region.ff_weights.copy()
        region._learn_ff_hebbian(flat_input, ltp_signal, winners)

        np.testing.assert_array_equal(
            region.ff_weights,
            before,
            err_msg="Zero neuromodulation should freeze weights",
        )

    def test_empty_winners_only_subthreshold(self):
        """With no winners, only subthreshold LTP should apply."""
        region = _make_region(seed=50, ff_sparsity=0.0)
        rng = np.random.default_rng(50)
        flat_input, ltp_signal, _ = _random_hebbian_inputs(region, rng)

        # Set weights to 0.3 so sub-LTP is visible
        region.ff_weights[:] = 0.3
        region.ff_weights *= region.ff_mask
        before = region.ff_weights.copy()

        region._learn_ff_hebbian(flat_input, ltp_signal, np.array([], dtype=np.intp))

        # Only active dims should change (subthreshold LTP)
        active_dims = np.flatnonzero(ltp_signal > 0.01)
        inactive_dims = np.flatnonzero(ltp_signal <= 0.01)

        if len(inactive_dims) > 0:
            np.testing.assert_array_equal(
                region.ff_weights[inactive_dims],
                before[inactive_dims],
                err_msg="Inactive input dims should not change without winners",
            )
        if len(active_dims) > 0:
            diff = region.ff_weights[active_dims] - before[active_dims]
            assert (diff >= -1e-10).all(), "Subthreshold should only increase"


# ---------------------------------------------------------------------------
# _learn_column_weights
# ---------------------------------------------------------------------------


class TestLearnColumnWeights:
    def test_active_src_active_tgt_increases(self):
        """w[src, tgt] increases when both src and tgt are active."""
        region = _make_region(seed=60)
        n_src, n_tgt = 4, 4
        n_col = region.n_columns

        active_cols = np.array([0, 2])
        weights = np.full((n_col, n_src, n_tgt), 0.5)
        src_fr = np.zeros(n_col * n_src)
        tgt_active = np.zeros(n_col * n_tgt)

        # Col 0: src[0] active, tgt[0] active
        src_fr[0] = 1.0
        tgt_active[0] = 1.0

        before = weights.copy()
        region._learn_column_weights(
            active_cols, weights, src_fr, tgt_active, n_src, n_tgt, 0.1, 0.01
        )

        assert weights[0, 0, 0] > before[0, 0, 0], (
            "Active src + active tgt should increase weight"
        )

    def test_inactive_src_active_tgt_decreases(self):
        """w[src, tgt] decreases when src is inactive but tgt is active."""
        region = _make_region(seed=61)
        n_src, n_tgt = 4, 4
        n_col = region.n_columns

        active_cols = np.array([0])
        weights = np.full((n_col, n_src, n_tgt), 0.5)
        src_fr = np.zeros(n_col * n_src)
        tgt_active = np.zeros(n_col * n_tgt)

        # Col 0: src[0] inactive (0.0), tgt[0] active
        tgt_active[0] = 1.0

        before = weights.copy()
        region._learn_column_weights(
            active_cols, weights, src_fr, tgt_active, n_src, n_tgt, 0.05, 0.1
        )

        assert weights[0, 0, 0] < before[0, 0, 0], (
            "Inactive src + active tgt should decrease weight"
        )

    def test_inactive_tgt_no_change(self):
        """Weights to inactive targets should not change (pre-clamp)."""
        region = _make_region(seed=62)
        n_src, n_tgt = 4, 4
        n_col = region.n_columns

        active_cols = np.array([0])
        weights = np.full((n_col, n_src, n_tgt), 0.5)
        src_fr = np.zeros(n_col * n_src)
        tgt_active = np.zeros(n_col * n_tgt)

        # Col 0: src active, but NO targets active
        src_fr[0] = 1.0

        before = weights.copy()
        region._learn_column_weights(
            active_cols, weights, src_fr, tgt_active, n_src, n_tgt, 0.05, 0.01
        )

        np.testing.assert_array_equal(
            weights,
            np.clip(before, 0, 1),
            err_msg="No active targets should mean no weight changes (except clamp)",
        )

    def test_empty_active_cols_no_change(self):
        """No active columns should leave weights unchanged."""
        region = _make_region(seed=63)
        n_src, n_tgt = 4, 4
        n_col = region.n_columns

        weights = np.full((n_col, n_src, n_tgt), 0.5)
        src_fr = np.ones(n_col * n_src)
        tgt_active = np.ones(n_col * n_tgt)

        before = weights.copy()
        region._learn_column_weights(
            np.array([], dtype=np.intp),
            weights,
            src_fr,
            tgt_active,
            n_src,
            n_tgt,
            0.05,
            0.01,
        )
        np.testing.assert_array_equal(weights, before)

    def test_weights_clamped_to_unit_interval(self):
        """Weights must remain in [0, 1] after repeated updates."""
        rng = np.random.default_rng(5)
        region = _make_region(seed=5)
        n_src, n_tgt = 4, 4

        weights = rng.uniform(0.0, 1.0, size=(region.n_columns, n_src, n_tgt))
        for _ in range(50):
            n_col = region.n_columns
            active_cols = np.sort(
                rng.choice(n_col, size=region.k_columns, replace=False)
            )
            src_fr = rng.uniform(0.0, 1.0, size=n_col * n_src)
            tgt_active = np.zeros(n_col * n_tgt)
            for c in active_cols:
                tgt_active[c * n_tgt + rng.choice(n_tgt)] = 1.0

            region._learn_column_weights(
                active_cols,
                weights,
                src_fr,
                tgt_active,
                n_src,
                n_tgt,
                0.05,
                0.01,
            )

        assert weights.min() >= 0.0
        assert weights.max() <= 1.0

    def test_inactive_cols_untouched(self):
        """Columns not in active_cols should not have weights modified."""
        region = _make_region(seed=64)
        n_src, n_tgt = 4, 4
        n_col = region.n_columns

        active_cols = np.array([0, 2])
        weights = np.full((n_col, n_src, n_tgt), 0.5)
        src_fr = np.ones(n_col * n_src)
        tgt_active = np.ones(n_col * n_tgt)

        before = weights.copy()
        region._learn_column_weights(
            active_cols,
            weights,
            src_fr,
            tgt_active,
            n_src,
            n_tgt,
            0.05,
            0.01,
        )

        # Inactive columns (1, 3, 4, ...) should be unchanged
        inactive = [c for c in range(n_col) if c not in active_cols]
        np.testing.assert_array_equal(
            weights[inactive],
            np.clip(before[inactive], 0, 1),
            err_msg="Inactive columns should not be modified",
        )
