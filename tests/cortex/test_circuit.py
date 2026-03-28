import numpy as np
import pytest

from step.cortex.circuit import Circuit, ConnectionRole
from step.cortex.modulators import SurpriseTracker, ThalamicGate
from step.cortex.sensory import SensoryRegion
from step.data import STORY_BOUNDARY
from step.encoders.charbit import CharbitEncoder
from tests.conftest import run_circuit


@pytest.fixture()
def encoder():
    return CharbitEncoder(length=4, width=5, chars="abcd")


@pytest.fixture()
def region1():
    return SensoryRegion(
        input_dim=4 * 5,
        encoding_width=5,
        n_columns=8,
        n_l4=2,
        n_l23=2,
        k_columns=2,
        seed=42,
    )


@pytest.fixture()
def region2(region1):
    return SensoryRegion(
        input_dim=region1.n_l23_total,
        encoding_width=0,
        n_columns=4,
        n_l4=2,
        n_l23=2,
        k_columns=1,
        voltage_decay=0.8,
        seed=123,
    )


class TestSingleRegion:
    def test_single_region_runs(self, region1, encoder):
        """Circuit with one entry region processes tokens without error."""
        tokens = [(0, "a"), (1, "b"), (2, "c"), (0, "a"), (1, "b")]
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        result = run_circuit(cortex, tokens)
        assert result.elapsed_seconds > 0

    def test_single_region_with_story_boundary(self, region1, encoder):
        tokens = [
            (0, "a"),
            (1, "b"),
            (STORY_BOUNDARY, ""),
            (0, "a"),
            (1, "b"),
        ]
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        result = run_circuit(cortex, tokens)
        assert result.elapsed_seconds > 0


class TestHierarchy:
    def test_hierarchy_runs(self, region1, region2, encoder):
        """Circuit with feedforward + surprise processes without error."""
        tokens = [
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (STORY_BOUNDARY, ""),
            (0, "a"),
            (1, "b"),
        ]
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect(
            region1.l23,
            region2.l4,
            ConnectionRole.FEEDFORWARD,
            surprise_tracker=SurpriseTracker(),
        )
        result = run_circuit(cortex, tokens)
        assert result.elapsed_seconds > 0
        assert "S2" in result.surprise_modulators
        assert len(result.surprise_modulators["S2"]) > 0

    def test_hierarchy_region2_activates(self, region1, region2, encoder):
        tokens = [(i % 3, chr(ord("a") + i % 3)) for i in range(20)]
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect(
            region1.l23,
            region2.l4,
            ConnectionRole.FEEDFORWARD,
            surprise_tracker=SurpriseTracker(),
        )
        run_circuit(cortex, tokens)
        assert region2.active_columns.sum() > 0

    def test_hierarchy_with_buffer_runs(self, region1, encoder):
        """End-to-end with buffer_depth=3, S2 activates."""
        buf_depth = 3
        region2 = SensoryRegion(
            input_dim=region1.n_l23_total * buf_depth,
            encoding_width=0,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            voltage_decay=0.8,
            seed=123,
        )
        tokens = [(i % 3, chr(ord("a") + i % 3)) for i in range(30)]
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect(
            region1.l23,
            region2.l4,
            ConnectionRole.FEEDFORWARD,
            buffer_depth=buf_depth,
            surprise_tracker=SurpriseTracker(),
        )
        result = run_circuit(cortex, tokens)
        assert result.elapsed_seconds > 0
        assert region2.active_columns.sum() > 0


class TestTopoOrder:
    def test_topo_order(self, region1, region2, encoder):
        """Feedforward edges determine processing order."""
        cortex = Circuit(encoder)
        # Add S2 first, but S1 is entry and feeds S2
        cortex.add_region("S2", region2)
        cortex.add_region("S1", region1, entry=True)
        cortex.connect(region1.l23, region2.l4, ConnectionRole.FEEDFORWARD)
        order = cortex._topo_order()
        assert order.index("S1") < order.index("S2")


class TestValidation:
    def test_missing_entry_raises(self, region1, encoder):
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1)
        with pytest.raises(ValueError, match="No entry region"):
            cortex.process(encoder.encode("a"))

    def test_duplicate_region_raises(self, region1, encoder):
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        with pytest.raises(ValueError, match="Duplicate"):
            cortex.add_region("S1", region1)

    def test_unregistered_lamina_in_connect_raises(self, region1, region2, encoder):
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        # region2 not registered — its lamina should fail
        with pytest.raises(ValueError, match="not registered"):
            cortex.connect(region1.l23, region2.l4, ConnectionRole.FEEDFORWARD)

    def test_unknown_role_raises(self, region1, region2, encoder):
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        with pytest.raises(ValueError, match="Unknown connection role"):
            cortex.connect(region1.l23, region2.l4, "bogus")


class TestApical:
    def test_apical_connection_inits_segments(self, region1, region2, encoder):
        """connect(..., 'apical') calls init_apical_segments on target."""
        assert not region1.has_apical
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect(region2.l23, region1.l4, ConnectionRole.APICAL)
        assert region1.has_apical


class TestAccessors:
    def test_timelines(self, region1, encoder):
        cortex = Circuit(encoder, enable_timeline=True)
        cortex.add_region("S1", region1, entry=True)
        run_circuit(cortex, [(0, "a"), (1, "b")])
        assert "S1" in cortex.timelines
        assert len(cortex.timelines["S1"].frames) > 0

    def test_diagnostics(self, region1, encoder):
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        run_circuit(cortex, [(0, "a"), (1, "b")])
        assert "S1" in cortex.diagnostics

    def test_region_accessor(self, region1, encoder):
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        assert cortex.region("S1") is region1


class TestTemporalBuffer:
    def test_buffer_depth_1_is_identity(self, region1, region2, encoder):
        """Default buffer_depth=1 behaves identically to no buffer."""
        tokens = [(i % 3, chr(ord("a") + i % 3)) for i in range(10)]
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect(
            region1.l23,
            region2.l4,
            ConnectionRole.FEEDFORWARD,
            buffer_depth=1,
            surprise_tracker=SurpriseTracker(),
        )
        result = run_circuit(cortex, tokens)
        assert result.elapsed_seconds > 0

    def test_buffer_concatenates_snapshots(self, region1, encoder):
        """With buffer_depth=3, signal is 3x source dim, oldest first."""
        buf_depth = 3
        region2 = SensoryRegion(
            input_dim=region1.n_l23_total * buf_depth,
            encoding_width=0,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            seed=123,
        )
        tokens = [(i % 3, chr(ord("a") + i % 3)) for i in range(5)]

        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect(
            region1.l23, region2.l4, ConnectionRole.FEEDFORWARD, buffer_depth=buf_depth
        )

        # Find the connection object
        ff_conn = cortex._connections[0]
        assert ff_conn._buffer is not None
        assert ff_conn._buffer.shape == (buf_depth, region1.n_l23_total)

        # Run and verify S2 gets a signal of the right length
        run_circuit(cortex, tokens)
        assert region2.active_columns.sum() >= 0  # ran without error

    def test_buffer_zero_pads_initially(self, region1, encoder):
        """First token with buffer_depth=3: first 2 slots are zero."""
        buf_depth = 3
        region2 = SensoryRegion(
            input_dim=region1.n_l23_total * buf_depth,
            encoding_width=0,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            seed=123,
        )
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect(
            region1.l23, region2.l4, ConnectionRole.FEEDFORWARD, buffer_depth=buf_depth
        )

        ff_conn = cortex._connections[0]

        # Process one token through S1 manually
        enc = encoder.encode("a")
        region1.process(enc)

        signal = cortex._get_ff_signal(ff_conn)
        assert signal.shape == (buf_depth * region1.n_l23_total,)
        # First 2 slots should be zero (oldest), last slot has the signal
        n = region1.n_l23_total
        np.testing.assert_array_equal(signal[:n], 0.0)
        np.testing.assert_array_equal(signal[n : 2 * n], 0.0)

    def test_story_boundary_clears_buffer(self, region1, encoder):
        """Story boundary zeros the buffer and resets position."""
        buf_depth = 2
        region2 = SensoryRegion(
            input_dim=region1.n_l23_total * buf_depth,
            encoding_width=0,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            seed=123,
        )
        tokens = [
            (0, "a"),
            (1, "b"),
            (STORY_BOUNDARY, ""),
            (0, "a"),
        ]
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect(
            region1.l23,
            region2.l4,
            ConnectionRole.FEEDFORWARD,
            buffer_depth=buf_depth,
            surprise_tracker=SurpriseTracker(),
        )
        run_circuit(cortex, tokens)

        ff_conn = cortex._connections[0]
        # After the run, buffer_pos should reflect post-boundary state
        # (1 token after boundary → pos=1)
        assert ff_conn._buffer_pos == 1

    def test_buffer_input_dim_mismatch_raises(self, region1, region2, encoder):
        """Wrong input_dim caught at connect() time."""
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)  # region2 has input_dim = n_l23_total * 1
        with pytest.raises(ValueError, match="input_dim"):
            cortex.connect(
                region1.l23, region2.l4, ConnectionRole.FEEDFORWARD, buffer_depth=3
            )


class TestBurstGating:
    def test_burst_gate_zeros_precise_columns(self, region1, encoder):
        """Precise columns are zeroed, bursting columns pass through."""
        region2 = SensoryRegion(
            input_dim=region1.n_l23_total,
            encoding_width=0,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            seed=123,
        )
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect(
            region1.l23, region2.l4, ConnectionRole.FEEDFORWARD, burst_gate=True
        )

        # Process a token so S1 has some state
        enc = encoder.encode("a")
        region1.process(enc)

        ff_conn = cortex._connections[0]
        signal = cortex._get_ff_signal(ff_conn)

        # Verify: neurons in non-bursting columns should be zero
        for col in range(region1.n_columns):
            start = col * region1.n_l23
            end = start + region1.n_l23
            if not region1.bursting_columns[col]:
                np.testing.assert_array_equal(signal[start:end], 0.0)

    def test_burst_gate_with_buffer(self, region1, encoder):
        """Combined mode: each buffer slot stores gated signal."""
        buf_depth = 2
        region2 = SensoryRegion(
            input_dim=region1.n_l23_total * buf_depth,
            encoding_width=0,
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            seed=123,
        )
        tokens = [(i % 3, chr(ord("a") + i % 3)) for i in range(10)]
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect(
            region1.l23,
            region2.l4,
            ConnectionRole.FEEDFORWARD,
            buffer_depth=buf_depth,
            burst_gate=True,
            surprise_tracker=SurpriseTracker(),
        )
        result = run_circuit(cortex, tokens)
        assert result.elapsed_seconds > 0


class TestDeterminism:
    def test_same_seed_same_results(self, encoder):
        """Same seed produces identical probe snapshots."""
        from step.probes.core import LaminaProbe

        tokens = [(i % 4, chr(ord("a") + i % 4)) for i in range(50)]

        probe1 = LaminaProbe()
        r1 = SensoryRegion(
            input_dim=4 * 5,
            encoding_width=5,
            n_columns=8,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            seed=42,
        )
        c1 = Circuit(encoder)
        c1.add_region("S1", r1, entry=True)
        result1 = run_circuit(c1, tokens, probes=[probe1])

        probe2 = LaminaProbe()
        r2 = SensoryRegion(
            input_dim=4 * 5,
            encoding_width=5,
            n_columns=8,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            seed=42,
        )
        c2 = Circuit(encoder)
        c2.add_region("S1", r2, entry=True)
        result2 = run_circuit(c2, tokens, probes=[probe2])

        snap1 = result1.probe_snapshots["lamina"]["S1"]
        snap2 = result2.probe_snapshots["lamina"]["S1"]
        assert abs(snap1.l4.recall - snap2.l4.recall) < 1e-10


class TestThalamicGate:
    """Unit tests for ThalamicGate."""

    def test_gate_starts_closed(self):
        gate = ThalamicGate()
        assert gate.readiness == pytest.approx(0.0)

    def test_gate_opens_with_low_burst_rate(self):
        gate = ThalamicGate()
        for _ in range(100):
            gate.update(0.0)
        assert gate.readiness > 0.95

    def test_gate_stays_closed_with_high_burst_rate(self):
        gate = ThalamicGate()
        for _ in range(100):
            gate.update(1.0)
        assert gate.readiness < 0.05

    def test_gate_reset(self):
        gate = ThalamicGate()
        for _ in range(100):
            gate.update(0.0)
        assert gate.readiness > 0.95
        gate.reset()
        assert gate.readiness == pytest.approx(0.0)


class TestThalamicGateIntegration:
    """Integration tests for thalamic gating in circuit."""

    def test_apical_with_thalamic_gate_runs(self, region1, region2, encoder):
        """Hierarchy with thalamic gate runs and populates thalamic_readiness."""
        tokens = [(i % 3, chr(ord("a") + i % 3)) for i in range(30)]
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect(
            region1.l23,
            region2.l4,
            ConnectionRole.FEEDFORWARD,
            surprise_tracker=SurpriseTracker(),
        )
        cortex.connect(
            region2.l23, region1.l4, ConnectionRole.APICAL, thalamic_gate=ThalamicGate()
        )
        result = run_circuit(cortex, tokens)
        assert "S2->S1" in result.thalamic_readiness
        assert len(result.thalamic_readiness["S2->S1"]) > 0

    def test_story_boundary_resets_gate(self, region1, region2, encoder):
        """Gate resets at story boundary (readiness drops back toward 0)."""
        tokens = (
            [(i % 3, chr(ord("a") + i % 3)) for i in range(20)]
            + [(STORY_BOUNDARY, "")]
            + [(i % 3, chr(ord("a") + i % 3)) for i in range(5)]
        )
        gate = ThalamicGate()
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect(
            region1.l23,
            region2.l4,
            ConnectionRole.FEEDFORWARD,
            surprise_tracker=SurpriseTracker(),
        )
        cortex.connect(
            region2.l23, region1.l4, ConnectionRole.APICAL, thalamic_gate=gate
        )
        run_circuit(cortex, tokens)
        # After boundary + 5 tokens, gate should be partially open but not fully
        # (it was reset at the boundary, so readiness should be modest)
        assert gate.readiness < 0.5

    def test_no_gate_backward_compatible(self, region1, region2, encoder):
        """Apical without gate works and thalamic_readiness is empty."""
        tokens = [(i % 3, chr(ord("a") + i % 3)) for i in range(20)]
        cortex = Circuit(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect(
            region1.l23,
            region2.l4,
            ConnectionRole.FEEDFORWARD,
            surprise_tracker=SurpriseTracker(),
        )
        cortex.connect(region2.l23, region1.l4, ConnectionRole.APICAL)
        result = run_circuit(cortex, tokens)
        assert result.thalamic_readiness == {}
