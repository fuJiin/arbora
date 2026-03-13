import numpy as np
import pytest

from step.cortex import SensoryRegion
from step.cortex.region import CorticalRegion
from step.data import STORY_BOUNDARY
from step.encoders.charbit import CharbitEncoder
from step.runner import run_hierarchy

# ---------------------------------------------------------------------------
# Apical segment initialization
# ---------------------------------------------------------------------------


class TestApicalInit:
    def test_no_apical_by_default(self):
        """Regions start without apical segments."""
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        assert not r.has_apical
        assert r.apical_seg_indices is None

    def test_init_apical_segments(self):
        """init_apical_segments creates arrays with correct shape."""
        r = CorticalRegion(
            n_columns=4, n_l4=2, n_l23=2, k_columns=1,
            n_apical_segments=3, n_synapses_per_segment=8,
        )
        r.init_apical_segments(source_dim=16)
        assert r.has_apical
        assert r.apical_seg_indices.shape == (8, 3, 8)  # n_l4_total, n_seg, n_syn
        assert r.apical_seg_perm.shape == (8, 3, 8)
        assert r._apical_context.shape == (16,)

    def test_apical_indices_in_range(self):
        """All apical synapse indices should be within source_dim."""
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.init_apical_segments(source_dim=10)
        assert r.apical_seg_indices.max() < 10
        assert r.apical_seg_indices.min() >= 0

    def test_apical_perm_starts_zero(self):
        """Apical permanences start at zero (disconnected)."""
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.init_apical_segments(source_dim=10)
        assert r.apical_seg_perm.sum() == 0.0


# ---------------------------------------------------------------------------
# Apical prediction
# ---------------------------------------------------------------------------


class TestApicalPrediction:
    def test_no_prediction_without_context(self):
        """No apical predictions when context is all zero."""
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.init_apical_segments(source_dim=8)
        r._predict_apical_columns()
        assert not r.apical_predicted_cols.any()

    def test_no_prediction_without_connected_synapses(self):
        """No predictions when permanences are below threshold."""
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.init_apical_segments(source_dim=8)
        r._apical_context[:] = 1.0  # all active
        r._predict_apical_columns()
        assert not r.apical_predicted_cols.any()

    def test_prediction_with_active_segment(self):
        """Column predicted when apical segment has enough active connected synapses."""
        r = CorticalRegion(
            n_columns=4, n_l4=2, n_l23=2, k_columns=1,
            seg_activation_threshold=2,
        )
        r.init_apical_segments(source_dim=8)

        # Wire neuron 0 (column 0), segment 0: all synapses → source 0
        r.apical_seg_indices[0, 0, :] = 0
        r.apical_seg_perm[0, 0, :] = 1.0

        # Set context: source 0 active
        r._apical_context[0] = 0.5

        r._predict_apical_columns()
        assert r.apical_predicted_cols[0]  # column 0 predicted


# ---------------------------------------------------------------------------
# Prediction gain integration
# ---------------------------------------------------------------------------


class TestPredictionGain:
    def test_gain_boosts_predicted_columns(self):
        """prediction_gain > 1 should amplify voltage in apical-predicted columns."""
        r = CorticalRegion(
            n_columns=4, n_l4=2, n_l23=2, k_columns=2,
            prediction_gain=2.0,
        )
        r.init_apical_segments(source_dim=8)

        # Wire column 0 for apical prediction
        r.apical_seg_indices[0, 0, :] = 0
        r.apical_seg_perm[0, 0, :] = 1.0
        r._apical_context[0] = 1.0

        # Give equal drive to all neurons
        drive = np.ones(r.n_l4_total) * 0.5
        r.step(drive)

        # Column 0 should have been gain-boosted, making it more likely to activate
        assert r.active_columns[0]

    def test_no_gain_without_apical(self):
        """prediction_gain has no effect without apical segments."""
        r = CorticalRegion(
            n_columns=4, n_l4=2, n_l23=2, k_columns=2,
            prediction_gain=2.0,
        )
        # No init_apical_segments called
        drive = np.ones(r.n_l4_total) * 0.5
        r.step(drive)
        # Should still work without error
        assert r.active_columns.sum() > 0


# ---------------------------------------------------------------------------
# Apical segment learning
# ---------------------------------------------------------------------------


class TestApicalLearning:
    def test_grow_on_burst(self):
        """Apical segments should grow when a column bursts."""
        r = CorticalRegion(
            n_columns=4, n_l4=2, n_l23=2, k_columns=1,
            perm_init=0.6,
        )
        r.init_apical_segments(source_dim=8)
        initial_perm = r.apical_seg_perm.copy()

        # Set apical context (will be saved as pred context)
        r._apical_context[:4] = 1.0

        # Drive column 0 to activate (will burst since no predictions)
        drive = np.zeros(r.n_l4_total)
        drive[0:2] = 1.0
        r.step(drive)

        # Permanences should have changed
        assert not np.array_equal(r.apical_seg_perm, initial_perm)

    def test_reinforce_on_precise(self):
        """Active apical segments should be reinforced on precise activation."""
        r = CorticalRegion(
            n_columns=4, n_l4=2, n_l23=2, k_columns=1,
            seg_activation_threshold=2,
        )
        r.init_apical_segments(source_dim=8)

        # Wire apical segment on neuron 0 with connected synapses
        r.apical_seg_indices[0, 0, :] = 0
        r.apical_seg_perm[0, 0, :] = 0.6  # above threshold

        # Set context to make segment active
        r._apical_context[0] = 1.0

        # Need fb prediction for precise activation
        r.fb_seg_indices[0, 0, :] = 0
        r.fb_seg_perm[0, 0, :] = 1.0
        r.active_l23[0] = True

        initial_perm = r.apical_seg_perm[0, 0, 0]

        drive = np.zeros(r.n_l4_total)
        drive[0:2] = 1.0
        r.step(drive)

        # Permanence should have increased
        assert r.apical_seg_perm[0, 0, 0] >= initial_perm


# ---------------------------------------------------------------------------
# Reset clears apical state
# ---------------------------------------------------------------------------


class TestApicalReset:
    def test_reset_clears_apical(self):
        """reset_working_memory clears apical context and predictions."""
        r = CorticalRegion(n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.init_apical_segments(source_dim=8)
        r._apical_context[:] = 1.0
        r.apical_predicted_cols[:] = True

        r.reset_working_memory()

        assert not r._apical_context.any()
        assert not r.apical_predicted_cols.any()


# ---------------------------------------------------------------------------
# Hierarchy integration with apical feedback
# ---------------------------------------------------------------------------


class TestHierarchyApical:
    @pytest.fixture()
    def encoder(self):
        return CharbitEncoder(length=4, width=5, chars="abcd")

    @pytest.fixture()
    def regions(self):
        r1 = SensoryRegion(
            input_dim=4 * 5,
            encoding_width=5,
            n_columns=8,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            prediction_gain=1.5,
            seed=42,
        )
        r2 = SensoryRegion(
            input_dim=r1.n_l23_total,
            encoding_width=0,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            seed=123,
        )
        # Initialize apical feedback: R2 L2/3 → R1
        r1.init_apical_segments(source_dim=r2.n_l23_total)
        return r1, r2

    def test_hierarchy_with_apical_runs(self, regions, encoder):
        """Hierarchy with apical feedback runs without error."""
        r1, r2 = regions
        tokens = [
            (0, "a"), (1, "b"), (2, "c"),
            (STORY_BOUNDARY, ""),
            (0, "a"), (1, "b"),
        ]
        metrics = run_hierarchy(
            r1, r2, encoder, tokens,
            enable_apical_feedback=True, log_interval=1000,
        )
        assert metrics.elapsed_seconds > 0

    def test_apical_context_flows(self, regions, encoder):
        """After processing tokens, R1 apical context should be non-zero."""
        r1, r2 = regions
        tokens = [(i % 3, chr(ord("a") + i % 3)) for i in range(20)]
        run_hierarchy(
            r1, r2, encoder, tokens,
            enable_apical_feedback=True, log_interval=1000,
        )
        # R2 should have produced some firing rate, which flows to R1 apical
        assert r1._apical_context.any()

    def test_apical_segments_grow(self, regions, encoder):
        """Apical segments should grow over a training sequence."""
        r1, r2 = regions
        tokens = [(i % 3, chr(ord("a") + i % 3)) for i in range(200)]
        run_hierarchy(
            r1, r2, encoder, tokens,
            enable_apical_feedback=True, log_interval=1000,
        )
        # Some apical permanences should have changed from zero
        assert r1.apical_seg_perm.sum() > 0

    def test_no_apical_without_init(self, encoder):
        """Hierarchy works without apical init (backward compatible)."""
        r1 = SensoryRegion(
            input_dim=4 * 5, encoding_width=5,
            n_columns=8, n_l4=2, n_l23=2, k_columns=2, seed=42,
        )
        r2 = SensoryRegion(
            input_dim=r1.n_l23_total, encoding_width=0,
            n_columns=4, n_l4=2, n_l23=2, k_columns=1, seed=123,
        )
        tokens = [(0, "a"), (1, "b"), (2, "c")]
        metrics = run_hierarchy(r1, r2, encoder, tokens, log_interval=1000)
        assert metrics.elapsed_seconds > 0
        assert not r1.has_apical
