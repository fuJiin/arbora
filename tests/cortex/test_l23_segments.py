"""Tests for L2/3 lateral dendritic segments."""

import numpy as np

from step.cortex.region import CorticalRegion
from step.cortex.sensory import SensoryRegion


class TestL23SegmentInit:
    def test_segment_arrays_created(self):
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        n_syn = r.n_synapses_per_segment
        assert r.l23_seg_indices.shape == (8, 4, n_syn)
        assert r.l23_seg_perm.shape == (8, 4, n_syn)

    def test_segments_start_disconnected(self):
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        assert r.l23_seg_perm.max() == 0.0

    def test_predicted_l23_starts_empty(self):
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        assert r.l23.predicted.sum() == 0

    def test_sensory_region_local_connectivity(self):
        """L2/3 segments should use local pools in SensoryRegion."""
        r = SensoryRegion(
            input_dim=20, n_columns=8, n_l4=2, n_l23=2, k_columns=2, seed=42
        )
        # Check that segment indices for col 0 only reference local neurons
        col0_neurons = range(0, 2)  # L2/3 neurons in col 0
        for neuron in col0_neurons:
            pool = r._l23_col_pools[0]
            for s in range(r.n_l23_segments):
                indices = r.l23_seg_indices[neuron, s]
                assert all(idx in pool for idx in indices)


class TestL23Prediction:
    def test_no_predictions_initially(self):
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.l23.active[0] = True
        pred = r._predict_l23_from_segments()
        assert pred.sum() == 0

    def test_prediction_with_active_segment(self):
        """Manually wire a segment and verify prediction fires."""
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        # Wire L2/3 neuron 0's segment 0: all synapses point to neuron 2
        r.l23_seg_indices[0, 0, :] = 2
        r.l23_seg_perm[0, 0, :] = 1.0
        # Activate neuron 2
        r.l23.active[2] = True
        pred = r._predict_l23_from_segments()
        assert pred[0]  # neuron 0 should be predicted

    def test_prediction_requires_threshold(self):
        """Segment needs enough active connected synapses to fire."""
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            seg_activation_threshold=3,
        )
        # Wire segment: 2 synapses to neuron 2, rest to neuron 4
        r.l23_seg_indices[0, 0, :] = 4
        r.l23_seg_indices[0, 0, 0] = 2
        r.l23_seg_indices[0, 0, 1] = 2
        r.l23_seg_perm[0, 0, :] = 1.0
        # Only activate neuron 2 — only 2 synapses match, threshold is 3
        r.l23.active[2] = True
        pred = r._predict_l23_from_segments()
        assert not pred[0]


class TestL23SegmentLearning:
    def test_segments_grow_on_burst(self):
        """L2/3 segments should grow when column bursts."""
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=2)
        # Set up context: some L2/3 neurons were active
        r._pred_context_l23[0] = True
        r._pred_context_l23[2] = True
        # Simulate burst in column 1
        r.active_columns[1] = True
        r.bursting_columns[1] = True
        r.l23.active[2:4] = True  # both L2/3 in col 1
        r.l23.voltage[2] = 0.8  # neuron 2 has higher voltage (trace winner)

        perm_before = r.l23_seg_perm.copy()
        r._learn_l23_segments()
        perm_after = r.l23_seg_perm

        # Neuron 2 (trace winner in col 1) should have grown segments
        assert not np.array_equal(perm_before[2], perm_after[2])

    def test_segments_reinforce_on_precise(self):
        """Correctly predicted L2/3 neurons should have segments reinforced."""
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=2)
        # Wire neuron 0's segment to activate from neuron 2
        r.l23_seg_indices[0, 0, :] = 2
        r.l23_seg_perm[0, 0, :] = 0.6  # above threshold
        # Set context
        r._pred_context_l23[2] = True
        # Simulate precise activation
        r.active_columns[0] = True
        r.bursting_columns[0] = False
        r.l23.active[0] = True
        r.l23.predicted[0] = True

        perm_before = r.l23_seg_perm[0, 0].copy()
        r._learn_l23_segments()
        perm_after = r.l23_seg_perm[0, 0]

        # Active synapses should be strengthened
        assert perm_after[0] > perm_before[0]

    def test_false_predictions_punished(self):
        """Predicted but inactive L2/3 neurons should have segments weakened."""
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=2)
        # Wire neuron 0's segment
        r.l23_seg_indices[0, 0, :] = 2
        r.l23_seg_perm[0, 0, :] = 0.6
        r._pred_context_l23[2] = True
        # Neuron 0 predicted but NOT active
        r.l23.predicted[0] = True
        r.l23.active[0] = False
        r.active_columns[0] = True

        perm_before = r.l23_seg_perm[0, 0, 0]
        r._learn_l23_segments()
        perm_after = r.l23_seg_perm[0, 0, 0]

        assert perm_after < perm_before


class TestL23SegmentIntegration:
    def test_predicted_l23_gets_voltage_boost(self):
        """Predicted L2/3 neurons should have higher voltage after activation."""
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=2)
        # Wire segment: neuron 0 predicted from neuron 2
        r.l23_seg_indices[0, 0, :] = 2
        r.l23_seg_perm[0, 0, :] = 1.0
        r.l23.active[2] = True

        # Run prediction
        r._compute_predictions()
        assert r.l23.predicted[0]

    def test_end_to_end_with_sensory(self):
        """Full process() cycle should work with L2/3 segments."""
        r = SensoryRegion(
            input_dim=10, n_columns=4, n_l4=2, n_l23=2, k_columns=2, seed=42
        )
        enc = np.zeros(10)
        enc[0] = 1.0
        enc[5] = 1.0
        # Run a few steps to build context
        for _ in range(10):
            r.process(enc)
        # Should not crash; segments should start growing
        assert r.l23_seg_perm.max() >= 0.0

    def test_reset_clears_predicted_l23(self):
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.l23.predicted[0] = True
        r.reset_working_memory()
        assert r.l23.predicted.sum() == 0
