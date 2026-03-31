"""Tests for L5→L5 lateral segments (output-layer sequence prediction)."""

import numpy as np

from arbora.cortex.region import CorticalRegion


def _make_region(**kwargs):
    defaults = dict(
        input_dim=16,
        n_columns=8,
        n_l4=4,
        n_l23=4,
        n_l5=4,
        k_columns=2,
        n_l5_segments=4,
        n_synapses_per_segment=8,
        seg_activation_threshold=2,
        seed=42,
    )
    defaults.update(kwargs)
    return CorticalRegion(**defaults)


class TestL5LateralSegmentInit:
    def test_segment_arrays_allocated(self):
        r = _make_region()
        assert r.l5_seg_indices.shape == (r.n_l5_total, 4, 8)
        assert r.l5_seg_perm.shape == (r.n_l5_total, 4, 8)

    def test_source_pool(self):
        r = _make_region()
        assert len(r._l5_source_pool) == r.n_l5_total


class TestL5LateralPrediction:
    def test_prediction_from_active_l5(self):
        r = _make_region()
        # Wire segment 0 of L5 neuron 1 to fire when neuron 0 is active
        r.l5_seg_indices[1, 0, :] = 0
        r.l5_seg_perm[1, 0, :] = 1.0
        # Set neuron 0 active
        r.l5.active[0] = True
        predicted = r._predict_l5_lateral_from_segments()
        assert predicted[1], "L5 neuron 1 should be predicted by neuron 0"

    def test_no_prediction_without_activity(self):
        r = _make_region()
        r.l5_seg_indices[1, 0, :] = 0
        r.l5_seg_perm[1, 0, :] = 1.0
        predicted = r._predict_l5_lateral_from_segments()
        assert not predicted.any()

    def test_lateral_prediction_in_compute_predictions(self):
        r = _make_region()
        r.l5_seg_indices[1, 0, :] = 0
        r.l5_seg_perm[1, 0, :] = 1.0
        r.l5.active[0] = True
        r._compute_predictions()
        assert r.l5.predicted[1]

    def test_lateral_and_apical_predictions_combine(self):
        """Both lateral and apical predictions OR into predicted_l5."""
        r = _make_region()
        r.init_apical_context(source_dim=32, source_name="S2")
        # Lateral: neuron 0 predicts neuron 1
        r.l5_seg_indices[1, 0, :] = 0
        r.l5_seg_perm[1, 0, :] = 1.0
        r.l5.active[0] = True
        # Apical: context predicts neuron 2
        src = r._apical_sources["S2"]
        src["seg_indices"][2, 0, :] = 0
        src["seg_perm"][2, 0, :] = 1.0
        src["context"][0] = 1.0
        r._compute_predictions()
        assert r.l5.predicted[1], "Lateral prediction"
        assert r.l5.predicted[2], "Apical prediction"


class TestL5LateralLearning:
    def test_growth_on_burst(self):
        r = _make_region()
        rng = np.random.default_rng(0)
        # Run steps to get L5 active
        for _ in range(5):
            r.step(rng.random(r.n_l4_total))
        initial = r.l5_seg_perm.sum()
        r._learn_l5_lateral_segments()
        assert r.l5_seg_perm.sum() != initial

    def test_reinforce_on_predicted(self):
        r = _make_region()
        # Wire: neuron 0 predicts neuron 1 via lateral
        r.l5_seg_indices[1, 0, :] = 0
        r.l5_seg_perm[1, 0, :] = 0.6
        r.l5.active[0] = True
        r.l5.active[1] = True
        r.l5.predicted[1] = True
        r.active_columns[0] = True
        r.bursting_columns[0] = False
        initial = r.l5_seg_perm[1, 0, 0]
        r._learn_l5_lateral_segments()
        assert r.l5_seg_perm[1, 0, 0] > initial

    def test_punish_false_prediction(self):
        r = _make_region()
        r.l5_seg_indices[1, 0, :] = 0
        r.l5_seg_perm[1, 0, :] = 0.6
        r.l5.active[0] = True
        r.l5.predicted[1] = True
        r.l5.active[1] = False  # predicted but not active
        r.active_columns[0] = True  # need at least one active column
        initial = r.l5_seg_perm[1, 0, 0]
        r._learn_l5_lateral_segments()
        assert r.l5_seg_perm[1, 0, 0] < initial


class TestL5SegmentTrace:
    def test_trace_initialized_when_decay_enabled(self):
        r = _make_region(pre_trace_decay=0.8)
        assert r._seg_trace_l5 is not None
        assert r._seg_trace_l5.shape == (r.n_l5_total,)

    def test_trace_not_initialized_when_decay_zero(self):
        r = _make_region(pre_trace_decay=0.0)
        assert r._seg_trace_l5 is None

    def test_trace_updated_on_step(self):
        r = _make_region(pre_trace_decay=0.8)
        rng = np.random.default_rng(0)
        r.step(rng.random(r.n_l4_total))
        # Trace is presynaptic — records previous step's L5 state.
        # Run a second step so the first step's L5 activations
        # get recorded in the trace.
        r.step(rng.random(r.n_l4_total))
        assert r._seg_trace_l5.sum() > 0

    def test_trace_decays(self):
        r = _make_region(pre_trace_decay=0.8)
        r._seg_trace_l5[0] = 1.0
        rng = np.random.default_rng(0)
        r.step(rng.random(r.n_l4_total))
        # Should have decayed to ~0.8 (or 0.8 + 1.0 if neuron 0 is active)
        assert r._seg_trace_l5[0] <= 1.8

    def test_trace_reset(self):
        r = _make_region(pre_trace_decay=0.8)
        r._seg_trace_l5[0] = 1.0
        r.reset_working_memory()
        assert r._seg_trace_l5.sum() == 0.0


class TestL5LateralBoost:
    def test_predicted_l5_boosts_winner(self):
        r = _make_region(fb_boost=5.0)  # high boost to override firing rate
        # Zero out firing rates so boost is decisive
        r.l23.firing_rate[:] = 0.01  # uniform small baseline
        r.l5.predicted[0 * r.n_l5 + 1] = True
        r.active_columns[0] = True
        r.bursting_columns[0] = False
        r._activate_l5(np.array([0]))
        assert r.l5.active[0 * r.n_l5 + 1]
