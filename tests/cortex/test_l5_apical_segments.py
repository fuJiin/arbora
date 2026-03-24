"""Tests for L5 apical dendritic segments (BAC firing model)."""

import numpy as np

from step.cortex.region import CorticalRegion


def _make_region(use_l5_apical=True, **kwargs):
    defaults = dict(
        input_dim=16,
        n_columns=8,
        n_l4=4,
        n_l23=4,
        n_l5=4,
        k_columns=2,
        n_apical_segments=4,
        n_synapses_per_segment=8,
        seg_activation_threshold=2,
        use_l5_apical_segments=use_l5_apical,
        seed=42,
    )
    defaults.update(kwargs)
    return CorticalRegion(**defaults)


class TestL5ApicalSegmentInit:
    """Segment data structures are allocated correctly."""

    def test_segment_arrays_allocated(self):
        r = _make_region()
        r.init_apical_context(source_dim=32, source_name="S2")
        src = r._apical_sources["S2"]
        assert "seg_indices" in src
        assert "seg_perm" in src
        assert src["seg_indices"].shape == (r.n_l5_total, 4, 8)
        assert src["seg_perm"].shape == (r.n_l5_total, 4, 8)

    def test_no_weights_in_segment_mode(self):
        r = _make_region()
        r.init_apical_context(source_dim=32, source_name="S2")
        src = r._apical_sources["S2"]
        assert "weights" not in src

    def test_linear_mode_no_segments(self):
        r = _make_region(use_l5_apical=False)
        r.init_apical_context(source_dim=32, source_name="S2")
        src = r._apical_sources["S2"]
        assert "weights" in src
        assert "seg_indices" not in src

    def test_multiple_sources(self):
        r = _make_region()
        r.init_apical_context(source_dim=32, source_name="S2")
        r.init_apical_context(source_dim=16, source_name="M1")
        assert len(r._apical_sources) == 2
        assert r._apical_sources["S2"]["seg_indices"].shape[0] == r.n_l5_total
        assert r._apical_sources["M1"]["seg_indices"].shape[0] == r.n_l5_total


class TestL5ApicalPrediction:
    """L5 neurons predicted when apical segment matches context."""

    def _setup_predicted_neuron(self, r, source_name="S2"):
        """Wire segment 0 of L5 neuron 0 to fire for a specific context."""
        src = r._apical_sources[source_name]
        # Set all synapses in segment 0 of neuron 0 to source index 0
        src["seg_indices"][0, 0, :] = 0
        src["seg_perm"][0, 0, :] = 1.0
        # Set context: source neuron 0 is active
        src["context"][:] = 0.0
        src["context"][0] = 1.0

    def test_predicted_when_segment_matches(self):
        r = _make_region()
        r.init_apical_context(source_dim=32, source_name="S2")
        self._setup_predicted_neuron(r)
        predicted = r._predict_l5_from_segments()
        assert predicted[0], "L5 neuron 0 should be predicted"

    def test_not_predicted_without_context(self):
        r = _make_region()
        r.init_apical_context(source_dim=32, source_name="S2")
        src = r._apical_sources["S2"]
        src["seg_indices"][0, 0, :] = 0
        src["seg_perm"][0, 0, :] = 1.0
        # No context set
        predicted = r._predict_l5_from_segments()
        assert not predicted.any()

    def test_compute_predictions_includes_l5(self):
        r = _make_region()
        r.init_apical_context(source_dim=32, source_name="S2")
        self._setup_predicted_neuron(r)
        r._compute_predictions()
        assert r.predicted_l5[0]


class TestL5ApicalBACFiring:
    """Predicted L5 neurons get boosted in competitive selection."""

    def test_apical_boost_biases_winner(self):
        r = _make_region(fb_boost=0.5)
        r.init_apical_context(source_dim=32, source_name="S2")

        # Run a few steps to build up firing rates
        rng = np.random.default_rng(123)
        for _ in range(10):
            r.step(rng.random(r.n_l4_total))

        # Wire segment so L5 neuron 1 in column 0 is predicted
        src = r._apical_sources["S2"]
        target_neuron = 0 * r.n_l5 + 1  # col 0, neuron 1
        src["seg_indices"][target_neuron, 0, :] = 0
        src["seg_perm"][target_neuron, 0, :] = 1.0
        src["context"][:] = 0.0
        src["context"][0] = 1.0

        # Run prediction
        r._compute_predictions()
        assert r.predicted_l5[target_neuron]

        # Activate L5 with column 0 active and not bursting
        r.active_columns[:] = False
        r.active_columns[0] = True
        r.bursting_columns[:] = False
        r._activate_l5(np.array([0]))

        # The predicted neuron should be the winner (or at least active)
        assert r.active_l5[target_neuron], (
            "Predicted L5 neuron should win with apical boost"
        )

    def test_no_boost_without_segments(self):
        """Linear gain mode should not set predicted_l5."""
        r = _make_region(use_l5_apical=False)
        r.init_apical_context(source_dim=32, source_name="S2")
        rng = np.random.default_rng(0)
        r.step(rng.random(r.n_l4_total))
        assert not r.predicted_l5.any()


class TestL5ApicalLearning:
    """Segment growth and reinforcement on L5 apical dendrites."""

    def test_growth_on_unpredicted_column(self):
        """Active L5 but not predicted → grow segment on winner."""
        r = _make_region()
        r.init_apical_context(source_dim=32, source_name="S2")
        src = r._apical_sources["S2"]
        # Set context
        src["context"][:] = 0.0
        src["context"][0] = 1.0
        src["context"][1] = 1.0

        # Run a step to get columns active
        rng = np.random.default_rng(0)
        r.step(rng.random(r.n_l4_total))

        # Get initial perm state
        initial_perm_sum = src["seg_perm"].sum()

        # Run learning
        r._learn_l5_apical()

        # Permanences should have changed (growth happened)
        assert src["seg_perm"].sum() != initial_perm_sum

    def test_reinforce_on_predicted_column(self):
        """Active L5 AND predicted → reinforce matching segments."""
        r = _make_region(fb_boost=0.5)
        r.init_apical_context(source_dim=32, source_name="S2")
        src = r._apical_sources["S2"]

        # Set up: wire neuron 0 to be predicted
        src["seg_indices"][0, 0, :] = 0
        src["seg_perm"][0, 0, :] = 0.6  # Above threshold
        src["context"][:] = 0.0
        src["context"][0] = 1.0

        # Make neuron 0 active and predicted
        r.active_columns[0] = True
        r.active_l5[0] = True
        r.predicted_l5[0] = True

        initial_perm = src["seg_perm"][0, 0, 0]
        r._learn_l5_apical()

        # Permanence should increase (reinforcement)
        assert src["seg_perm"][0, 0, 0] > initial_perm

    def test_punish_false_positive(self):
        """Predicted but not active → punish segment."""
        r = _make_region()
        r.init_apical_context(source_dim=32, source_name="S2")
        src = r._apical_sources["S2"]

        # Wire neuron 0 predicted but make it inactive
        src["seg_indices"][0, 0, :] = 0
        src["seg_perm"][0, 0, :] = 0.6
        src["context"][:] = 0.0
        src["context"][0] = 1.0

        r.predicted_l5[0] = True
        r.active_l5[0] = False
        r.active_columns[:] = False  # Column not active

        initial_perm = src["seg_perm"][0, 0, 0]
        r._learn_l5_apical()

        # Permanence should decrease (punishment)
        assert src["seg_perm"][0, 0, 0] < initial_perm

    def test_reset_clears_predicted_l5(self):
        r = _make_region()
        r.predicted_l5[0] = True
        r.reset_working_memory()
        assert not r.predicted_l5.any()


class TestBackwardCompat:
    """Default (use_l5_apical_segments=False) preserves old behavior."""

    def test_linear_gain_still_works(self):
        r = _make_region(use_l5_apical=False)
        r.init_apical_context(source_dim=32, source_name="S2")
        src = r._apical_sources["S2"]
        src["context"][:] = 0.5  # Nonzero context
        rng = np.random.default_rng(0)
        # Should not crash — linear gain path runs
        for _ in range(5):
            r.step(rng.random(r.n_l4_total))
        assert r.active_l5.any()

    def test_default_is_linear_gain(self):
        r = CorticalRegion(
            input_dim=16,
            n_columns=8,
            n_l4=4,
            n_l23=4,
            k_columns=2,
        )
        assert not r.use_l5_apical_segments
