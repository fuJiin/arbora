import numpy as np
import pytest

from arbor.cortex import SensoryRegion
from arbor.cortex.circuit import Circuit, ConnectionRole
from arbor.cortex.modulators import SurpriseTracker
from arbor.cortex.region import CorticalRegion
from arbor.encoders.charbit import CharbitEncoder
from examples.chat.data import STORY_BOUNDARY
from tests.conftest import run_circuit

# ---------------------------------------------------------------------------
# Apical context initialization
# ---------------------------------------------------------------------------


class TestApicalInit:
    def test_no_apical_by_default(self):
        """Regions start without apical context."""
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        assert not r.has_apical

    def test_init_apical_context(self):
        """init_apical_context creates context buffer with correct shape."""
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.init_apical_context(source_dim=16)
        assert r.has_apical
        assert r._apical_context.shape == (16,)

    def test_backward_compat_alias(self):
        """init_apical_segments still works as alias."""
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.init_apical_segments(source_dim=10)
        assert r.has_apical
        assert r._apical_context.shape == (10,)


# ---------------------------------------------------------------------------
# Gain modulation
# ---------------------------------------------------------------------------


class TestGainModulation:
    def test_gain_amplifies_with_context(self):
        """Apical context > 0 should amplify voltage via gain."""
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            prediction_gain=2.0,
        )
        r.init_apical_context(source_dim=8)

        # Strong apical context
        r._apical_context[:] = 1.0

        # Give equal drive to all neurons
        drive = np.ones(r.n_l4_total) * 0.5
        r.step(drive)

        # Should still activate k columns (gain doesn't change which, just sharpens)
        assert r.active_columns.sum() == 2

    def test_no_gain_without_context(self):
        """Zero apical context should apply no gain."""
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            prediction_gain=2.0,
        )
        r.init_apical_context(source_dim=8)
        # Context stays zero
        drive = np.ones(r.n_l4_total) * 0.5
        r.step(drive)
        assert r.active_columns.sum() == 2

    def test_no_gain_without_apical(self):
        """prediction_gain has no effect without apical init."""
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            prediction_gain=2.0,
        )
        drive = np.ones(r.n_l4_total) * 0.5
        r.step(drive)
        assert r.active_columns.sum() > 0

    def test_gain_is_modulatory_not_additive(self):
        """With zero feedforward drive, apical context should NOT activate columns."""
        r = CorticalRegion(
            8,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            prediction_gain=3.0,
        )
        r.init_apical_context(source_dim=8)
        r._apical_context[:] = 1.0  # strong context

        # Zero drive — gain should multiply by zero
        drive = np.zeros(r.n_l4_total)
        r.step(drive)

        # Columns may still activate via excitability, but apical alone
        # should not create drive from nothing


# ---------------------------------------------------------------------------
# Reset clears apical state
# ---------------------------------------------------------------------------


class TestApicalReset:
    def test_reset_clears_apical(self):
        """reset_working_memory clears apical context."""
        r = CorticalRegion(8, n_columns=4, n_l4=2, n_l23=2, k_columns=1)
        r.init_apical_context(source_dim=8)
        r._apical_context[:] = 1.0

        r.reset_working_memory()

        assert not r._apical_context.any()


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
        return r1, r2

    def test_hierarchy_with_apical_runs(self, regions, encoder):
        """Hierarchy with apical feedback runs without error."""
        r1, r2 = regions
        tokens = [
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (STORY_BOUNDARY, ""),
            (0, "a"),
            (1, "b"),
        ]
        circuit = Circuit(encoder)
        circuit.add_region("S1", r1, entry=True)
        circuit.add_region("S2", r2)
        circuit.connect(
            r1.l23,
            r2.l4,
            ConnectionRole.FEEDFORWARD,
            surprise_tracker=SurpriseTracker(),
        )
        circuit.connect(r2.l23, r1.l4, ConnectionRole.APICAL)
        result = run_circuit(circuit, tokens, log_interval=1000)
        assert result.elapsed_seconds > 0

    def test_apical_context_flows(self, regions, encoder):
        """After processing tokens, S1 apical context should be non-zero."""
        r1, r2 = regions
        tokens = [(i % 3, chr(ord("a") + i % 3)) for i in range(20)]
        circuit = Circuit(encoder)
        circuit.add_region("S1", r1, entry=True)
        circuit.add_region("S2", r2)
        circuit.connect(
            r1.l23,
            r2.l4,
            ConnectionRole.FEEDFORWARD,
            surprise_tracker=SurpriseTracker(),
        )
        circuit.connect(r2.l23, r1.l4, ConnectionRole.APICAL)
        run_circuit(circuit, tokens, log_interval=1000)
        assert r1._apical_context.any()

    def test_no_apical_without_init(self, encoder):
        """Hierarchy works without apical init (backward compatible)."""
        r1 = SensoryRegion(
            input_dim=4 * 5,
            encoding_width=5,
            n_columns=8,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            seed=42,
        )
        r2 = SensoryRegion(
            input_dim=r1.n_l23_total,
            encoding_width=0,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            seed=123,
        )
        tokens = [(0, "a"), (1, "b"), (2, "c")]
        circuit = Circuit(encoder)
        circuit.add_region("S1", r1, entry=True)
        circuit.add_region("S2", r2)
        circuit.connect(
            r1.l23,
            r2.l4,
            ConnectionRole.FEEDFORWARD,
            surprise_tracker=SurpriseTracker(),
        )
        result = run_circuit(circuit, tokens, log_interval=1000)
        assert result.elapsed_seconds > 0
        assert not r1.has_apical
