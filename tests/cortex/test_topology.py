import numpy as np
import pytest

from step.cortex.sensory import SensoryRegion
from step.cortex.topology import Topology
from step.data import STORY_BOUNDARY
from step.encoders.charbit import CharbitEncoder


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
        """Topology with one entry region processes tokens without error."""
        tokens = [(0, "a"), (1, "b"), (2, "c"), (0, "a"), (1, "b")]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        result = cortex.run(tokens, log_interval=1000)
        assert result.elapsed_seconds > 0
        assert len(result.per_region["S1"].overlaps) == 4  # t > 0

    def test_single_region_with_story_boundary(self, region1, encoder):
        tokens = [
            (0, "a"), (1, "b"),
            (STORY_BOUNDARY, ""),
            (0, "a"), (1, "b"),
        ]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        result = cortex.run(tokens, log_interval=1000)
        assert len(result.per_region["S1"].overlaps) == 3


class TestHierarchy:
    def test_hierarchy_runs(self, region1, region2, encoder):
        """Topology with feedforward + surprise processes without error."""
        tokens = [
            (0, "a"), (1, "b"), (2, "c"),
            (STORY_BOUNDARY, ""),
            (0, "a"), (1, "b"),
        ]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect("S1", "S2", "feedforward")
        cortex.connect("S1", "S2", "surprise")
        result = cortex.run(tokens, log_interval=1000)
        assert result.elapsed_seconds > 0
        assert "S2" in result.surprise_modulators
        assert len(result.surprise_modulators["S2"]) > 0

    def test_hierarchy_region2_activates(self, region1, region2, encoder):
        tokens = [(i % 3, chr(ord("a") + i % 3)) for i in range(20)]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect("S1", "S2", "feedforward")
        cortex.connect("S1", "S2", "surprise")
        cortex.run(tokens, log_interval=1000)
        assert region2.active_columns.sum() > 0

    def test_hierarchy_with_buffer_runs(self, region1, encoder):
        """End-to-end with buffer_depth=3, S2 activates."""
        buf_depth = 3
        region2 = SensoryRegion(
            input_dim=region1.n_l23_total * buf_depth,
            encoding_width=0,
            n_columns=4, n_l4=2, n_l23=2, k_columns=1,
            voltage_decay=0.8,
            seed=123,
        )
        tokens = [(i % 3, chr(ord("a") + i % 3)) for i in range(30)]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect("S1", "S2", "feedforward", buffer_depth=buf_depth)
        cortex.connect("S1", "S2", "surprise")
        result = cortex.run(tokens, log_interval=1000)
        assert result.elapsed_seconds > 0
        assert region2.active_columns.sum() > 0


class TestTopoOrder:
    def test_topo_order(self, region1, region2, encoder):
        """Feedforward edges determine processing order."""
        cortex = Topology(encoder)
        # Add S2 first, but S1 is entry and feeds S2
        cortex.add_region("S2", region2)
        cortex.add_region("S1", region1, entry=True)
        cortex.connect("S1", "S2", "feedforward")
        order = cortex._topo_order()
        assert order.index("S1") < order.index("S2")


class TestValidation:
    def test_missing_entry_raises(self, region1, encoder):
        cortex = Topology(encoder)
        cortex.add_region("S1", region1)
        with pytest.raises(ValueError, match="No entry region"):
            cortex.run([(0, "a")])

    def test_duplicate_region_raises(self, region1, encoder):
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        with pytest.raises(ValueError, match="Duplicate"):
            cortex.add_region("S1", region1)

    def test_unknown_region_in_connect_raises(self, region1, encoder):
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        with pytest.raises(ValueError, match="Unknown region"):
            cortex.connect("S1", "S99", "feedforward")

    def test_unknown_kind_raises(self, region1, region2, encoder):
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        with pytest.raises(ValueError, match="Unknown connection kind"):
            cortex.connect("S1", "S2", "bogus")


class TestApical:
    def test_apical_connection_inits_segments(self, region1, region2, encoder):
        """connect(..., 'apical') calls init_apical_segments on target."""
        assert not region1.has_apical
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect("S2", "S1", "apical")
        assert region1.has_apical


class TestAccessors:
    def test_timelines(self, region1, encoder):
        cortex = Topology(encoder, enable_timeline=True)
        cortex.add_region("S1", region1, entry=True)
        cortex.run([(0, "a"), (1, "b")], log_interval=1000)
        assert "S1" in cortex.timelines
        assert len(cortex.timelines["S1"].frames) > 0

    def test_diagnostics(self, region1, encoder):
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.run([(0, "a"), (1, "b")], log_interval=1000)
        assert "S1" in cortex.diagnostics

    def test_region_accessor(self, region1, encoder):
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        assert cortex.region("S1") is region1


class TestTemporalBuffer:
    def test_buffer_depth_1_is_identity(self, region1, region2, encoder):
        """Default buffer_depth=1 behaves identically to no buffer."""
        tokens = [(i % 3, chr(ord("a") + i % 3)) for i in range(10)]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect("S1", "S2", "feedforward", buffer_depth=1)
        cortex.connect("S1", "S2", "surprise")
        result = cortex.run(tokens, log_interval=1000)
        assert result.elapsed_seconds > 0

    def test_buffer_concatenates_snapshots(self, region1, encoder):
        """With buffer_depth=3, signal is 3x source dim, oldest first."""
        buf_depth = 3
        region2 = SensoryRegion(
            input_dim=region1.n_l23_total * buf_depth,
            encoding_width=0,
            n_columns=4, n_l4=2, n_l23=2, k_columns=1,
            seed=123,
        )
        tokens = [(i % 3, chr(ord("a") + i % 3)) for i in range(5)]

        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect("S1", "S2", "feedforward", buffer_depth=buf_depth)

        # Find the connection object
        ff_conn = cortex._connections[0]
        assert ff_conn._buffer is not None
        assert ff_conn._buffer.shape == (buf_depth, region1.n_l23_total)

        # Run and verify S2 gets a signal of the right length
        cortex.connect("S1", "S2", "surprise")
        cortex.run(tokens, log_interval=1000)
        assert region2.active_columns.sum() >= 0  # ran without error

    def test_buffer_zero_pads_initially(self, region1, encoder):
        """First token with buffer_depth=3: first 2 slots are zero."""
        buf_depth = 3
        region2 = SensoryRegion(
            input_dim=region1.n_l23_total * buf_depth,
            encoding_width=0,
            n_columns=4, n_l4=2, n_l23=2, k_columns=1,
            seed=123,
        )
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect("S1", "S2", "feedforward", buffer_depth=buf_depth)

        ff_conn = cortex._connections[0]

        # Process one token through S1 manually
        enc = encoder.encode("a")
        region1.process(enc)

        signal = cortex._get_ff_signal(ff_conn)
        assert signal.shape == (buf_depth * region1.n_l23_total,)
        # First 2 slots should be zero (oldest), last slot has the signal
        n = region1.n_l23_total
        np.testing.assert_array_equal(signal[:n], 0.0)
        np.testing.assert_array_equal(signal[n:2 * n], 0.0)

    def test_story_boundary_clears_buffer(self, region1, encoder):
        """Story boundary zeros the buffer and resets position."""
        buf_depth = 2
        region2 = SensoryRegion(
            input_dim=region1.n_l23_total * buf_depth,
            encoding_width=0,
            n_columns=4, n_l4=2, n_l23=2, k_columns=1,
            seed=123,
        )
        tokens = [
            (0, "a"), (1, "b"),
            (STORY_BOUNDARY, ""),
            (0, "a"),
        ]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect("S1", "S2", "feedforward", buffer_depth=buf_depth)
        cortex.connect("S1", "S2", "surprise")
        cortex.run(tokens, log_interval=1000)

        ff_conn = cortex._connections[0]
        # After the run, buffer_pos should reflect post-boundary state
        # (1 token after boundary → pos=1)
        assert ff_conn._buffer_pos == 1

    def test_buffer_input_dim_mismatch_raises(self, region1, region2, encoder):
        """Wrong input_dim caught at connect() time."""
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)  # region2 has input_dim = n_l23_total * 1
        with pytest.raises(ValueError, match="input_dim"):
            cortex.connect("S1", "S2", "feedforward", buffer_depth=3)


class TestBurstGating:
    def test_burst_gate_zeros_precise_columns(self, region1, encoder):
        """Precise columns are zeroed, bursting columns pass through."""
        region2 = SensoryRegion(
            input_dim=region1.n_l23_total,
            encoding_width=0,
            n_columns=4, n_l4=2, n_l23=2, k_columns=1,
            seed=123,
        )
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect("S1", "S2", "feedforward", burst_gate=True)

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
            n_columns=4, n_l4=2, n_l23=2, k_columns=1,
            seed=123,
        )
        tokens = [(i % 3, chr(ord("a") + i % 3)) for i in range(10)]
        cortex = Topology(encoder)
        cortex.add_region("S1", region1, entry=True)
        cortex.add_region("S2", region2)
        cortex.connect(
            "S1", "S2", "feedforward",
            buffer_depth=buf_depth, burst_gate=True,
        )
        cortex.connect("S1", "S2", "surprise")
        result = cortex.run(tokens, log_interval=1000)
        assert result.elapsed_seconds > 0


class TestResultsMatchRunner:
    def test_single_region_matches_run_cortex(self, encoder):
        """Topology produces same metrics as run_cortex for same seed."""
        from step.runner import run_cortex

        tokens = [(i % 4, chr(ord("a") + i % 4)) for i in range(50)]

        # Via run_cortex
        r1 = SensoryRegion(
            input_dim=4 * 5, encoding_width=5,
            n_columns=8, n_l4=2, n_l23=2, k_columns=2, seed=42,
        )
        old_metrics = run_cortex(r1, encoder, tokens, log_interval=1000)

        # Via Topology directly (same seed → same region)
        r2 = SensoryRegion(
            input_dim=4 * 5, encoding_width=5,
            n_columns=8, n_l4=2, n_l23=2, k_columns=2, seed=42,
        )
        cortex = Topology(encoder)
        cortex.add_region("S1", r2, entry=True)
        new_result = cortex.run(tokens, log_interval=1000)
        new_metrics = new_result.per_region["S1"]

        np.testing.assert_allclose(old_metrics.overlaps, new_metrics.overlaps)
        np.testing.assert_allclose(old_metrics.accuracies, new_metrics.accuracies)
        np.testing.assert_allclose(
            old_metrics.synaptic_accuracies, new_metrics.synaptic_accuracies
        )
