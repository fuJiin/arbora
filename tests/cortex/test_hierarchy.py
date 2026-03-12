import numpy as np
import pytest

from step.cortex import SensoryRegion, SurpriseTracker
from step.cortex.runner import STORY_BOUNDARY, run_hierarchy
from step.encoders.charbit import CharbitEncoder

# ---------------------------------------------------------------------------
# SurpriseTracker
# ---------------------------------------------------------------------------


class TestSurpriseTracker:
    def test_baseline_adaptation(self):
        """At steady-state burst rate, modulator should converge near 1.0."""
        tracker = SurpriseTracker()
        for _ in range(1000):
            mod = tracker.update(0.3)
        assert mod == pytest.approx(1.0, abs=0.05)

    def test_spike(self):
        """A sudden burst rate spike should produce modulator > 1.0."""
        tracker = SurpriseTracker()
        # Establish steady baseline at 0.1
        for _ in range(1000):
            tracker.update(0.1)
        # Spike: several high-burst steps to overcome EMA smoothing
        for _ in range(10):
            mod = tracker.update(0.9)
        assert mod > 1.0

    def test_clamp(self):
        """Modulator should never exceed 2.0."""
        tracker = SurpriseTracker()
        # Very low baseline
        for _ in range(200):
            tracker.update(0.01)
        # Huge spike
        mod = tracker.update(1.0)
        assert mod <= 2.0

    def test_zero_burst_rate(self):
        """Zero burst rate should not produce NaN or error."""
        tracker = SurpriseTracker()
        mod = tracker.update(0.0)
        assert np.isfinite(mod)
        assert mod >= 0.0


# ---------------------------------------------------------------------------
# Surprise modulator scales learning
# ---------------------------------------------------------------------------


class TestSurpriseModulatorScalesLearning:
    def test_region_learning_modulated(self):
        """Higher surprise_modulator should produce larger weight changes."""
        def run_one_step(modulator: float) -> float:
            r = SensoryRegion(
                input_dim=10,
                n_columns=4,
                n_l4=2,
                n_l23=2,
                k_columns=1,
                learning_rate=0.1,
                synapse_decay=1.0,
                seed=42,
            )
            r.surprise_modulator = modulator
            enc = np.zeros(10)
            enc[0] = 1.0
            enc[5] = 1.0
            r.process(enc)
            return float(r.l23_lateral_weights.sum())

        w_normal = run_one_step(1.0)
        w_high = run_one_step(2.0)
        # Higher modulator should produce larger or equal weight updates
        assert w_high >= w_normal


# ---------------------------------------------------------------------------
# Hierarchy integration
# ---------------------------------------------------------------------------


class TestHierarchyRuns:
    @pytest.fixture()
    def encoder(self):
        return CharbitEncoder(length=4, width=5, chars="abcd")

    @pytest.fixture()
    def region1(self):
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
    def region2(self, region1):
        return SensoryRegion(
            input_dim=region1.n_l23_total,
            encoding_width=0,  # sliding window for inter-region input
            n_columns=4,
            n_l4=2,
            n_l23=2,
            k_columns=1,
            voltage_decay=0.8,
            eligibility_decay=0.98,
            synapse_decay=0.9999,
            seed=123,
        )

    def test_hierarchy_runs(self, region1, region2, encoder):
        """Two regions process a short sequence without error."""
        tokens = [
            (0, "a"),
            (1, "b"),
            (2, "c"),
            (STORY_BOUNDARY, ""),
            (0, "a"),
            (1, "b"),
        ]
        metrics = run_hierarchy(
            region1, region2, encoder, tokens, log_interval=1000
        )
        assert metrics.elapsed_seconds > 0
        assert len(metrics.surprise_modulators) > 0

    def test_region2_receives_l23_output(self, region1, region2, encoder):
        """Region 2 input_dim must match Region 1 L2/3 total neurons."""
        assert region2.input_dim == region1.n_l23_total

    def test_region2_activates(self, region1, region2, encoder):
        """Region 2 should produce activations after Region 1."""
        tokens = [(i % 3, chr(ord("a") + i % 3)) for i in range(20)]
        run_hierarchy(region1, region2, encoder, tokens, log_interval=1000)
        # Region 2 should have activated at some point
        assert region2.active_columns.sum() > 0
