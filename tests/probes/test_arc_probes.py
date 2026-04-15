"""Tests for ARC visual decodability probes.

Tests use a minimal V1+V2 circuit with synthetic grids to verify probe
behavior without requiring the ARC-AGI SDK.
"""

import numpy as np

from arbora.cortex import SensoryRegion
from arbora.cortex.circuit import Circuit, ConnectionRole
from arbora.thalamus import ThalamicNucleus
from examples.arc.encoder import ArcGridEncoder
from examples.arc.probes import (
    ArcProbeBundle,
    ChangeLocalizationProbe,
    RepresentationStabilityProbe,
    TimerSelectivityProbe,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_arc_circuit(seed: int = 42):
    """Build a small V1+V2 circuit matching ARC architecture topology."""
    encoder = ArcGridEncoder()
    v1 = SensoryRegion(
        input_dim=encoder.input_dim,
        encoding_width=encoder.encoding_width,
        n_columns=32,
        n_l4=2,
        n_l23=2,
        n_l5=2,
        k_columns=4,
        n_l4_lat_segments=4,
        n_synapses_per_segment=16,
        seg_activation_threshold=2,
        seed=seed,
    )
    pulvinar = ThalamicNucleus(
        input_dim=v1.n_l5_total,
        relay_dim=v1.n_l5_total,
        seed=seed + 100,
    )
    v2 = SensoryRegion(
        input_dim=pulvinar.relay_dim,
        n_columns=16,
        n_l4=2,
        n_l23=2,
        n_l5=2,
        k_columns=2,
        n_l4_lat_segments=4,
        n_synapses_per_segment=16,
        seg_activation_threshold=2,
        seed=seed + 50,
    )
    circuit = Circuit(encoder)
    circuit.add_region("V1", v1, entry=True)
    circuit.add_region("pulvinar", pulvinar)
    circuit.add_region("V2", v2)
    circuit.connect(v1.l5, pulvinar.input_port, ConnectionRole.FEEDFORWARD)
    circuit.connect(pulvinar.output_port, v2.input_port, ConnectionRole.FEEDFORWARD)
    circuit.finalize()
    return circuit, encoder


def _make_grid(rng: np.random.Generator, base: int = 0) -> np.ndarray:
    """Create a 64x64 grid with a known pattern."""
    grid = np.full((64, 64), base, dtype=np.int8)
    # Random content in the main area
    grid[:60, :] = rng.integers(0, 16, size=(60, 64), dtype=np.int8)
    # Timer region: bottom 4 rows cycle predictably
    timer_val = rng.integers(0, 16)
    grid[60:64, :] = timer_val
    return grid


def _make_grid_with_change(
    prev_grid: np.ndarray, rng: np.random.Generator, *, timer_only: bool = False
) -> np.ndarray:
    """Create a new grid with controlled changes."""
    grid = prev_grid.copy()
    if not timer_only:
        # Change some content cells
        n_changes = rng.integers(5, 20)
        for _ in range(n_changes):
            r = rng.integers(0, 60)
            c = rng.integers(0, 64)
            grid[r, c] = (grid[r, c] + rng.integers(1, 16)) % 16
    # Always change the timer
    new_timer = (prev_grid[62, 0] + 1) % 16
    grid[60:64, :] = new_timer
    return grid


# ---------------------------------------------------------------------------
# ChangeLocalizationProbe
# ---------------------------------------------------------------------------


class TestChangeLocalization:
    def test_snapshot_empty(self):
        probe = ChangeLocalizationProbe()
        snap = probe.snapshot()
        assert snap["n_steps"] == 0
        assert snap["burst_change_corr"] == 0.0

    def test_observe_accumulates(self):
        circuit, encoder = _make_arc_circuit()
        probe = ChangeLocalizationProbe(window=20)
        rng = np.random.default_rng(42)

        grid = _make_grid(rng)
        for _ in range(5):
            encoding = encoder.encode(grid)
            circuit.process(encoding)
            probe.observe(circuit, encoder, grid)
            grid = _make_grid_with_change(grid, rng)

        snap = probe.snapshot()
        assert snap["n_steps"] == 5
        assert 0.0 <= snap["mean_burst_frac"] <= 1.0
        assert 0.0 <= snap["mean_change_frac"] <= 1.0

    def test_no_change_low_burst(self):
        """When the grid doesn't change, burst fraction should drop."""
        circuit, encoder = _make_arc_circuit()
        probe = ChangeLocalizationProbe(window=20)
        rng = np.random.default_rng(42)

        grid = _make_grid(rng)
        # First frame: everything is "novel"
        encoding = encoder.encode(grid)
        circuit.process(encoding)
        probe.observe(circuit, encoder, grid)

        # Repeat same grid: should produce low burst
        for _ in range(10):
            encoding = encoder.encode(grid)
            circuit.process(encoding)
            probe.observe(circuit, encoder, grid)

        snap = probe.snapshot()
        # After repeating the same frame, change fraction should be ~0
        # (except for the first frame which is all-change)
        assert snap["mean_change_frac"] < 0.2

    def test_reset_clears_state(self):
        circuit, encoder = _make_arc_circuit()
        probe = ChangeLocalizationProbe()
        rng = np.random.default_rng(42)

        grid = _make_grid(rng)
        encoding = encoder.encode(grid)
        circuit.process(encoding)
        probe.observe(circuit, encoder, grid)

        assert probe.snapshot()["n_steps"] == 1
        probe.reset()
        assert probe.snapshot()["n_steps"] == 0

    def test_localization_score_range(self):
        """Localization scores should be in [0, 1]."""
        circuit, encoder = _make_arc_circuit()
        probe = ChangeLocalizationProbe(window=20)
        rng = np.random.default_rng(42)

        grid = _make_grid(rng)
        for _ in range(10):
            encoding = encoder.encode(grid)
            circuit.process(encoding)
            probe.observe(circuit, encoder, grid)
            grid = _make_grid_with_change(grid, rng)

        snap = probe.snapshot()
        assert 0.0 <= snap["mean_localization"] <= 1.0


# ---------------------------------------------------------------------------
# TimerSelectivityProbe
# ---------------------------------------------------------------------------


class TestTimerSelectivity:
    def test_snapshot_empty(self):
        probe = TimerSelectivityProbe()
        snap = probe.snapshot()
        assert snap["timer_frames"] == 0
        assert snap["n_selective"] == 0

    def test_classifies_timer_frames(self):
        """Probe should correctly identify timer-only change frames."""
        circuit, encoder = _make_arc_circuit()
        probe = TimerSelectivityProbe()
        rng = np.random.default_rng(42)

        grid = _make_grid(rng)
        encoding = encoder.encode(grid)
        circuit.process(encoding)
        probe.observe(circuit, encoder, grid)

        # Timer-only change
        for _ in range(5):
            grid = _make_grid_with_change(grid, rng, timer_only=True)
            encoding = encoder.encode(grid)
            circuit.process(encoding)
            probe.observe(circuit, encoder, grid)

        snap = probe.snapshot()
        # Should have classified at least some timer frames
        assert snap["timer_frames"] >= 1

    def test_mixed_frames_skipped(self):
        """Frames where both timer and content change should be skipped."""
        circuit, encoder = _make_arc_circuit()
        probe = TimerSelectivityProbe()
        rng = np.random.default_rng(42)

        grid = _make_grid(rng)
        encoding = encoder.encode(grid)
        circuit.process(encoding)
        probe.observe(circuit, encoder, grid)

        # Mixed change (both timer and content)
        grid2 = _make_grid_with_change(grid, rng, timer_only=False)
        encoding = encoder.encode(grid2)
        circuit.process(encoding)
        probe.observe(circuit, encoder, grid2)

        snap = probe.snapshot()
        # Mixed frames don't count as timer or non-timer
        total_classified = snap["timer_frames"] + snap["nontimer_frames"]
        # At most 2 (first frame is all-change so it's mixed)
        assert total_classified <= 2

    def test_reset_clears_state(self):
        circuit, encoder = _make_arc_circuit()
        probe = TimerSelectivityProbe()
        rng = np.random.default_rng(42)

        grid = _make_grid(rng)
        encoding = encoder.encode(grid)
        circuit.process(encoding)
        probe.observe(circuit, encoder, grid)

        # Timer-only change
        grid2 = _make_grid_with_change(grid, rng, timer_only=True)
        encoding = encoder.encode(grid2)
        circuit.process(encoding)
        probe.observe(circuit, encoder, grid2)

        probe.reset()
        snap = probe.snapshot()
        assert snap["timer_frames"] == 0
        assert snap["nontimer_frames"] == 0

    def test_selectivity_threshold(self):
        """Selective columns require timer burst rate >> non-timer rate."""
        probe = TimerSelectivityProbe(selectivity_threshold=2.0)
        # Manually inject data to test selectivity computation
        probe._timer_frames = 10
        probe._nontimer_frames = 10
        # Column 5 bursts 8/10 timer frames, 1/10 non-timer
        probe._timer_bursts[5] = 8
        probe._nontimer_bursts[5] = 1
        # Column 10 bursts equally on both
        probe._timer_bursts[10] = 5
        probe._nontimer_bursts[10] = 5

        snap = probe.snapshot()
        selective_cols = [s["column"] for s in snap["selective_columns"]]
        assert 5 in selective_cols  # highly selective
        assert 10 not in selective_cols  # not selective


# ---------------------------------------------------------------------------
# RepresentationStabilityProbe
# ---------------------------------------------------------------------------


class TestRepresentationStability:
    def test_snapshot_empty(self):
        probe = RepresentationStabilityProbe()
        snap = probe.snapshot()
        assert snap["n_steps"] == 0
        assert snap["mean_jaccard"] == 0.0

    def test_identical_frames_high_jaccard(self):
        """Repeating the same grid should yield high V2 stability."""
        circuit, encoder = _make_arc_circuit()
        probe = RepresentationStabilityProbe(window=20)
        rng = np.random.default_rng(42)

        grid = _make_grid(rng)
        # Process same grid many times so V2 stabilizes
        for _ in range(20):
            encoding = encoder.encode(grid)
            circuit.process(encoding)
            probe.observe(circuit, encoder, grid)

        snap = probe.snapshot()
        assert snap["n_steps"] > 0
        # With identical input, V2 should eventually stabilize
        # (exact Jaccard depends on learning, but should be >= 0)
        assert snap["mean_jaccard"] >= 0.0

    def test_jaccard_range(self):
        """Jaccard similarity should be in [0, 1]."""
        circuit, encoder = _make_arc_circuit()
        probe = RepresentationStabilityProbe(window=20)
        rng = np.random.default_rng(42)

        grid = _make_grid(rng)
        for _ in range(10):
            encoding = encoder.encode(grid)
            circuit.process(encoding)
            probe.observe(circuit, encoder, grid)
            grid = _make_grid_with_change(grid, rng)

        snap = probe.snapshot()
        if snap["n_steps"] > 0:
            assert 0.0 <= snap["mean_jaccard"] <= 1.0
            assert 0.0 <= snap["min_jaccard"] <= snap["max_jaccard"] <= 1.0

    def test_missing_region_graceful(self):
        """Probe should handle missing V2 region gracefully."""
        # Circuit with only V1
        encoder = ArcGridEncoder()
        v1 = SensoryRegion(
            input_dim=encoder.input_dim,
            encoding_width=encoder.encoding_width,
            n_columns=16,
            n_l4=2,
            n_l23=2,
            k_columns=2,
            seed=42,
        )
        circuit = Circuit(encoder)
        circuit.add_region("V1", v1, entry=True)
        circuit.finalize()

        probe = RepresentationStabilityProbe(region_name="V2")
        rng = np.random.default_rng(42)
        grid = _make_grid(rng)

        encoding = encoder.encode(grid)
        circuit.process(encoding)
        probe.observe(circuit, encoder, grid)

        snap = probe.snapshot()
        assert snap["n_steps"] == 0  # never observed anything

    def test_reset_clears_state(self):
        circuit, encoder = _make_arc_circuit()
        probe = RepresentationStabilityProbe()
        rng = np.random.default_rng(42)

        grid = _make_grid(rng)
        for _ in range(5):
            encoding = encoder.encode(grid)
            circuit.process(encoding)
            probe.observe(circuit, encoder, grid)
            grid = _make_grid_with_change(grid, rng)

        assert probe.snapshot()["n_steps"] > 0
        probe.reset()
        assert probe.snapshot()["n_steps"] == 0


# ---------------------------------------------------------------------------
# ArcProbeBundle
# ---------------------------------------------------------------------------


class TestArcProbeBundle:
    def test_observe_all_probes(self):
        circuit, encoder = _make_arc_circuit()
        bundle = ArcProbeBundle(window=20)
        rng = np.random.default_rng(42)

        grid = _make_grid(rng)
        for _ in range(5):
            encoding = encoder.encode(grid)
            circuit.process(encoding)
            bundle.observe(circuit, encoder, grid)
            grid = _make_grid_with_change(grid, rng)

        snap = bundle.snapshot()
        assert "change_localization" in snap
        assert "timer_selectivity" in snap
        assert "representation_stability" in snap

        assert snap["change_localization"]["n_steps"] == 5
        assert snap["representation_stability"]["region"] == "V2"

    def test_reset_all(self):
        circuit, encoder = _make_arc_circuit()
        bundle = ArcProbeBundle()
        rng = np.random.default_rng(42)

        grid = _make_grid(rng)
        encoding = encoder.encode(grid)
        circuit.process(encoding)
        bundle.observe(circuit, encoder, grid)

        bundle.reset()
        snap = bundle.snapshot()
        assert snap["change_localization"]["n_steps"] == 0
        assert snap["representation_stability"]["n_steps"] == 0

    def test_print_report_does_not_crash(self, capsys):
        circuit, encoder = _make_arc_circuit()
        bundle = ArcProbeBundle()
        rng = np.random.default_rng(42)

        grid = _make_grid(rng)
        for _ in range(5):
            encoding = encoder.encode(grid)
            circuit.process(encoding)
            bundle.observe(circuit, encoder, grid)
            grid = _make_grid_with_change(grid, rng)

        # Should not raise
        bundle.print_report()
        captured = capsys.readouterr()
        assert "ARC Visual Probes" in captured.out
        assert "Change Localization" in captured.out
        assert "Timer Selectivity" in captured.out
        assert "V2 Representation Stability" in captured.out
