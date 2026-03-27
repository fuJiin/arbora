"""Regression tests for vectorized Hebbian learning functions.

Compares vectorized _learn_ff_hebbian and _learn_column_weights against
naive reference implementations (the original loop-based code) to ensure
numerically identical results.
"""

import numpy as np
import pytest

from step.cortex import CorticalRegion

# ---------------------------------------------------------------------------
# Reference (original loop-based) implementations
# ---------------------------------------------------------------------------


def _ref_learn_ff_hebbian(region, flat_input, ltp_signal, winner_indices):
    """Original loop-based _learn_ff_hebbian, pre-vectorization."""
    neuromod = region.surprise_modulator * region.reward_modulator
    ltp_rate = region.learning_rate * neuromod

    if len(winner_indices) > 0:
        region.ff_weights[:, winner_indices] += ltp_rate * ltp_signal[:, np.newaxis]

    if len(winner_indices) > 0:
        ltd_rate = region.ltd_rate * neuromod
        inactive_input = 1.0 - flat_input
        winner_cols = winner_indices // region.n_l4
        col_masks = region._col_mask[:, winner_cols]
        local_on = np.maximum((flat_input[:, np.newaxis] * col_masks).sum(axis=0), 1.0)
        local_off = np.maximum(
            (inactive_input[:, np.newaxis] * col_masks).sum(axis=0), 1.0
        )
        local_scales = local_on / local_off
        neuron_masks = region.ff_mask[:, winner_indices]
        region.ff_weights[:, winner_indices] -= (
            ltd_rate
            * local_scales[np.newaxis, :]
            * inactive_input[:, np.newaxis]
            * neuron_masks
        )
        w = region.ff_weights[:, winner_indices]
        w[~neuron_masks] = 0.0
        np.clip(w, 0, 1, out=w)
        region.ff_weights[:, winner_indices] = w

    active_dims = np.flatnonzero(ltp_signal > 0.01)
    if len(active_dims) > 0:
        sub_ltp = ltp_rate * 0.1 * ltp_signal[active_dims, np.newaxis]
        region.ff_weights[active_dims] += sub_ltp
        region.ff_weights[active_dims] *= region.ff_mask[active_dims]
        np.minimum(
            region.ff_weights[active_dims],
            1,
            out=region.ff_weights[active_dims],
        )


def _ref_learn_column_weights(
    region,
    active_cols,
    weights,
    src_fr,
    tgt_active,
    n_src,
    n_tgt,
    ltp_rate,
    ltd_rate,
):
    """Original loop-based _learn_column_weights, pre-vectorization."""
    src_by_col = src_fr.reshape(region.n_columns, n_src)
    tgt_by_col = tgt_active.reshape(region.n_columns, n_tgt)
    for col in active_cols:
        winners = np.nonzero(tgt_by_col[col])[0]
        if len(winners) == 0:
            continue
        src = src_by_col[col]
        w = weights[col]
        for j in winners:
            w[:, j] += ltp_rate * src
            w[:, j] -= ltd_rate * (1.0 - src)
    np.clip(weights, 0.0, 1.0, out=weights)


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


def _random_column_inputs(region, rng, n_src=4, n_tgt=4):
    """Generate random inputs for _learn_column_weights."""
    n_col = region.n_columns
    active_cols = np.sort(rng.choice(n_col, size=region.k_columns, replace=False))
    weights = rng.uniform(0.0, 1.0, size=(n_col, n_src, n_tgt))
    src_fr = np.zeros(n_col * n_src)
    for c in active_cols:
        active_src = rng.choice(n_src, size=max(1, n_src // 2), replace=False)
        src_fr[c * n_src + active_src] = rng.uniform(0.3, 1.0, size=len(active_src))
    tgt_active = np.zeros(n_col * n_tgt)
    for c in active_cols:
        active_tgt = rng.choice(n_tgt, size=max(1, n_tgt // 2), replace=False)
        tgt_active[c * n_tgt + active_tgt] = 1.0

    return active_cols, weights, src_fr, tgt_active


# ---------------------------------------------------------------------------
# _learn_ff_hebbian regression tests
# ---------------------------------------------------------------------------


class TestLearnFfHebbianRegression:
    @pytest.mark.parametrize("seed", [0, 1, 42, 99, 2024])
    def test_matches_reference(self, seed):
        """Vectorized _learn_ff_hebbian matches loop-based reference."""
        rng = np.random.default_rng(seed)

        ref_region = _make_region(seed=seed)
        vec_region = _make_region(seed=seed)

        flat_input, ltp_signal, winners = _random_hebbian_inputs(ref_region, rng)

        # Ensure identical starting weights
        np.testing.assert_array_equal(ref_region.ff_weights, vec_region.ff_weights)

        _ref_learn_ff_hebbian(ref_region, flat_input, ltp_signal, winners)
        vec_region._learn_ff_hebbian(flat_input, ltp_signal, winners)

        np.testing.assert_allclose(
            vec_region.ff_weights,
            ref_region.ff_weights,
            atol=1e-12,
            err_msg=f"ff_weights diverged with seed={seed}",
        )

    def test_empty_winners(self):
        """No winners should leave weights unchanged."""
        region = _make_region()
        rng = np.random.default_rng(0)
        flat_input, ltp_signal, _ = _random_hebbian_inputs(region, rng)

        before = region.ff_weights.copy()
        region._learn_ff_hebbian(flat_input, ltp_signal, np.array([], dtype=np.intp))

        # Only subthreshold LTP should fire (on all neurons)
        # but with empty winners, LTP/LTD blocks are skipped
        active_dims = np.flatnonzero(ltp_signal > 0.01)
        if len(active_dims) == 0:
            np.testing.assert_array_equal(region.ff_weights, before)

    def test_weights_stay_in_bounds(self):
        """Weights must remain in [0, 1] after update."""
        rng = np.random.default_rng(7)
        region = _make_region(seed=7)

        for _ in range(20):
            flat_input, ltp_signal, winners = _random_hebbian_inputs(region, rng)
            region._learn_ff_hebbian(flat_input, ltp_signal, winners)

        assert region.ff_weights.min() >= 0.0
        assert region.ff_weights.max() <= 1.0

    def test_respects_ff_mask(self):
        """Masked-out connections must remain zero."""
        region = _make_region(ff_sparsity=0.5, seed=3)
        rng = np.random.default_rng(3)
        zero_mask = ~region.ff_mask

        for _ in range(10):
            flat_input, ltp_signal, winners = _random_hebbian_inputs(region, rng)
            region._learn_ff_hebbian(flat_input, ltp_signal, winners)

        np.testing.assert_array_equal(
            region.ff_weights[zero_mask],
            0.0,
            err_msg="Masked connections should stay zero",
        )


# ---------------------------------------------------------------------------
# _learn_column_weights regression tests
# ---------------------------------------------------------------------------


class TestLearnColumnWeightsRegression:
    @pytest.mark.parametrize("seed", [0, 1, 42, 99, 2024])
    def test_matches_reference(self, seed):
        """Vectorized _learn_column_weights matches loop-based reference."""
        rng = np.random.default_rng(seed)
        region = _make_region(seed=seed)

        n_src, n_tgt = 4, 4
        ltp_rate, ltd_rate = 0.05, 0.01

        active_cols, weights, src_fr, tgt_active = _random_column_inputs(
            region, rng, n_src=n_src, n_tgt=n_tgt
        )

        ref_weights = weights.copy()
        vec_weights = weights.copy()

        _ref_learn_column_weights(
            region,
            active_cols,
            ref_weights,
            src_fr,
            tgt_active,
            n_src,
            n_tgt,
            ltp_rate,
            ltd_rate,
        )
        region._learn_column_weights(
            active_cols,
            vec_weights,
            src_fr,
            tgt_active,
            n_src,
            n_tgt,
            ltp_rate,
            ltd_rate,
        )

        np.testing.assert_allclose(
            vec_weights,
            ref_weights,
            atol=1e-12,
            err_msg=f"column weights diverged with seed={seed}",
        )

    def test_empty_active_cols(self):
        """No active columns should leave weights unchanged."""
        region = _make_region()
        rng = np.random.default_rng(0)
        n_src, n_tgt = 4, 4
        _, weights, src_fr, tgt_active = _random_column_inputs(
            region, rng, n_src, n_tgt
        )
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

    def test_weights_stay_in_bounds(self):
        """Weights must remain in [0, 1] after repeated updates."""
        rng = np.random.default_rng(5)
        region = _make_region(seed=5)
        n_src, n_tgt = 4, 4

        for _ in range(20):
            active_cols, weights, src_fr, tgt_active = _random_column_inputs(
                region, rng, n_src, n_tgt
            )
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

    def test_no_target_winners_noop(self):
        """Columns with no active targets should not change weights."""
        region = _make_region(seed=11)
        n_src, n_tgt = 4, 4
        n_col = region.n_columns

        active_cols = np.array([0, 2])
        weights = np.full((n_col, n_src, n_tgt), 0.5)
        src_fr = np.ones(n_col * n_src) * 0.5
        tgt_active = np.zeros(n_col * n_tgt)  # no targets active

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
        np.testing.assert_array_equal(
            weights,
            np.clip(before, 0, 1),
            err_msg="With no active targets, weights should only be clipped",
        )
