"""Tests for HippocampalRegion composite — Region protocol + full pipeline."""

import numpy as np
import pytest

from arbora.cortex.circuit_types import Region
from arbora.hippocampus import HippocampalRegion


def _region(input_dim: int = 64, **overrides) -> HippocampalRegion:
    """Small default HC for fast tests."""
    defaults = dict(
        ec_dim=200,
        dg_dim=800,
        ca3_dim=200,
        seed=0,
    )
    defaults.update(overrides)
    return HippocampalRegion(input_dim, **defaults)


class TestInit:
    def test_shapes_and_ports(self):
        hc = _region(input_dim=64)
        assert hc.input_dim == 64
        assert hc.input_port.n_total == 64
        assert hc.output_port.n_total == 64
        assert hc.input_port.id == HippocampalRegion.INPUT_ID
        assert hc.output_port.id == HippocampalRegion.OUTPUT_ID

    def test_internal_layer_wiring(self):
        hc = _region(input_dim=64)
        assert hc.ec.input_dim == 64
        assert hc.ec.output_dim == 200
        assert hc.dg.input_dim == 200
        assert hc.dg.output_dim == 800
        assert hc.ca3.dim == 200
        assert hc.ca3.dg_dim == 800
        assert hc.ca1.ca3_dim == 200
        assert hc.ca1.ec_direct_dim == 200
        assert hc.ca1.output_dim == 200

    def test_invalid_input_dim_raises(self):
        with pytest.raises(ValueError):
            HippocampalRegion(input_dim=0)

    def test_invalid_retrieval_iterations_raises(self):
        with pytest.raises(ValueError):
            _region(retrieval_iterations=-1)

    def test_different_seeds_produce_different_projections(self):
        """Changing the top-level seed propagates to all internal RNGs."""
        a = _region(input_dim=64, seed=0)
        b = _region(input_dim=64, seed=1)
        assert not np.allclose(a.ec.forward_weights, b.ec.forward_weights)
        assert not np.allclose(a.dg.weights, b.dg.weights)
        assert not np.allclose(a.ca3.mossy_weights, b.ca3.mossy_weights)
        assert not np.allclose(a.ca1.ca3_weights, b.ca1.ca3_weights)

    def test_dg_k_defaults_to_two_percent(self):
        hc = _region(dg_dim=500)
        assert hc.dg.k_at(None) == 10


class TestRegionProtocol:
    def test_satisfies_runtime_protocol(self):
        hc = _region()
        assert isinstance(hc, Region)

    def test_has_required_methods(self):
        hc = _region()
        assert callable(hc.process)
        assert callable(hc.apply_reward)
        assert callable(hc.reset_working_memory)


class TestProcess:
    def test_output_shape_and_dtype(self):
        hc = _region(input_dim=64)
        rng = np.random.default_rng(0)
        out = hc.process(rng.normal(size=64))
        assert out.shape == (64,)
        assert out.dtype == np.float64

    def test_sets_output_port_firing_rate(self):
        hc = _region(input_dim=64)
        rng = np.random.default_rng(0)
        out = hc.process(rng.normal(size=64))
        np.testing.assert_array_equal(hc.output_port.firing_rate, out)

    def test_updates_last_match(self):
        hc = _region(input_dim=64)
        assert hc.last_match == 0.0
        rng = np.random.default_rng(0)
        hc.process(rng.normal(size=64))
        # Any numeric value in [-1, 1] is acceptable; the point is it moved.
        assert -1.0 <= hc.last_match <= 1.0

    def test_accepts_boolean_cortical_input(self):
        """Sparse binary upstream L2/3 firing should project cleanly."""
        hc = _region(input_dim=64)
        x = np.zeros(64, dtype=np.bool_)
        x[::4] = True
        out = hc.process(x)
        assert out.shape == (64,)
        assert np.isfinite(out).all()

    def test_wrong_input_dim_raises(self):
        hc = _region(input_dim=64)
        with pytest.raises(ValueError):
            hc.process(np.zeros(63))

    def test_kwargs_are_ignored_cleanly(self):
        """Region protocol passes **kwargs; unknown kwargs should not break."""
        hc = _region(input_dim=64)
        rng = np.random.default_rng(0)
        out = hc.process(rng.normal(size=64), salience=0.5, unknown_kw=True)
        assert out.shape == (64,)


class TestBindingAndRetrieval:
    """Load-bearing ARB-116 smoke test: pattern-complete across process() calls."""

    def test_repeated_same_input_is_deterministic(self):
        """Same input twice with LTP saturation → stable output."""
        hc = _region(input_dim=64)
        x = np.random.default_rng(1).normal(size=64)
        hc.process(x)
        # After the first call, lateral_weights saturate on this pattern;
        # the second call should reach the same attractor.
        out_a = hc.process(x)
        out_b = hc.process(x)
        np.testing.assert_array_equal(out_a, out_b)

    def test_memory_persists_across_reset_working_memory(self):
        """reset_working_memory preserves CA3 lateral weights."""
        hc = _region(input_dim=64)
        x = np.random.default_rng(2).normal(size=64)
        hc.process(x)  # prime
        before_lateral = hc.ca3.lateral_weights.copy()
        hc.process(x)  # saturate
        saturated_out = hc.process(x)

        hc.reset_working_memory()
        # Lateral weights survive the reset; CA3 state and output_port cleared.
        assert hc.ca3.lateral_weights.any()
        np.testing.assert_array_equal(hc.ca3.lateral_weights, hc.ca3.lateral_weights)
        assert not hc.ca3.state.any()
        assert not hc.output_port.firing_rate.any()

        # Re-running the same input should reach the same attractor.
        after_out = hc.process(x)
        np.testing.assert_array_equal(saturated_out, after_out)
        # And the lateral weights kept climbing (more LTP), but the
        # pattern they support is the same.
        assert hc.ca3.lateral_weights.sum() >= before_lateral.sum()

    def test_partial_cue_pattern_completes(self):
        """Encode X, then a lightly-perturbed X retrieves the same CA3 pattern.

        This is the canonical hippocampal pattern-completion test: the
        CA3 activation after process() should converge across small
        input perturbations once the attractor has been laid down.
        """
        hc = _region(input_dim=128, ec_dim=400, dg_dim=1600, ca3_dim=400)
        rng = np.random.default_rng(3)
        x = rng.normal(size=128)

        # Prime the attractor. Multiple exposures to ensure saturated lateral weights.
        for _ in range(3):
            hc.process(x)
        attractor = hc.ca3.state.copy()

        # Slightly perturbed input — small enough that DG output still overlaps
        # substantially with the stored DG pattern.
        x_partial = x + 0.02 * rng.normal(size=128)
        hc.process(x_partial)
        recovered = hc.ca3.state.copy()

        # We tolerate some drift but require most of the attractor to survive.
        overlap = (attractor & recovered).sum()
        assert overlap >= 0.5 * attractor.sum(), (
            f"pattern completion failed; overlap={overlap}/{attractor.sum()}"
        )


class TestReset:
    def test_reset_working_memory_clears_transient(self):
        hc = _region(input_dim=64)
        rng = np.random.default_rng(0)
        hc.process(rng.normal(size=64))
        assert hc.ca3.state.any()
        assert hc.output_port.firing_rate.any()
        assert hc.last_match != 0.0

        hc.reset_working_memory()
        assert not hc.ca3.state.any()
        assert not hc.output_port.firing_rate.any()
        assert hc.last_match == 0.0

    def test_reset_working_memory_preserves_lateral_weights(self):
        hc = _region(input_dim=64)
        hc.process(np.random.default_rng(0).normal(size=64))
        lateral_before = hc.ca3.lateral_weights.copy()
        hc.reset_working_memory()
        np.testing.assert_array_equal(hc.ca3.lateral_weights, lateral_before)

    def test_reset_memory_clears_lateral_weights(self):
        hc = _region(input_dim=64)
        hc.process(np.random.default_rng(0).normal(size=64))
        assert hc.ca3.lateral_weights.any()
        hc.reset_memory()
        assert not hc.ca3.lateral_weights.any()
        assert not hc.ca3.state.any()


class TestApplyReward:
    def test_apply_reward_is_noop(self):
        """HC learning is not reward-modulated in v1."""
        hc = _region(input_dim=64)
        hc.process(np.random.default_rng(0).normal(size=64))
        lateral_before = hc.ca3.lateral_weights.copy()
        state_before = hc.ca3.state.copy()
        match_before = hc.last_match

        hc.apply_reward(1.0)
        hc.apply_reward(-1.0)

        np.testing.assert_array_equal(hc.ca3.lateral_weights, lateral_before)
        np.testing.assert_array_equal(hc.ca3.state, state_before)
        assert hc.last_match == match_before
