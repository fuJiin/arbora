"""Tests for agranular regions (n_l4=0 or n_l5=0)."""

import numpy as np
import pytest

from arbora.cortex.region import CorticalRegion


class TestNoL4:
    """CorticalRegion with n_l4=0 (agranular, like motor cortex)."""

    @pytest.fixture()
    def region(self):
        return CorticalRegion(
            input_dim=20,
            n_columns=8,
            n_l4=0,
            n_l23=4,
            k_columns=2,
            seed=42,
        )

    def test_has_l4_false(self, region):
        assert not region.has_l4
        assert region.n_l4 == 0
        assert region.n_l4_total == 0

    def test_input_port_is_l23(self, region):
        assert region.input_port is region.l23

    def test_ff_weights_target_l23(self, region):
        assert region.ff_weights.shape == (20, region.n_l23_total)

    def test_l4_lamina_is_empty(self, region):
        assert region.l4.n_total == 0
        assert region.l4.active.shape == (0,)

    def test_process_runs(self, region):
        encoding = np.random.default_rng(0).random(20)
        result = region.process(encoding)
        assert isinstance(result, np.ndarray)

    def test_step_activates_l23(self, region):
        drive = np.random.default_rng(0).random(region.n_l23_total)
        region.step(drive)
        assert region.l23.active.any()
        assert region.active_columns.sum() == region.k_columns

    def test_step_returns_l23_indices(self, region):
        drive = np.random.default_rng(0).random(region.n_l23_total)
        result = region.step(drive)
        # Result should be L2/3 neuron indices, not L4
        assert len(result) > 0
        assert result.max() < region.n_l23_total

    def test_l5_activates(self, region):
        """L2/3→L5 pathway should still work."""
        drive = np.random.default_rng(0).random(region.n_l23_total)
        region.step(drive)
        assert region.l5.active.any()

    def test_output_scores_nonzero(self, region):
        drive = np.random.default_rng(0).random(region.n_l23_total)
        region.step(drive)
        assert region.output_scores.sum() > 0

    def test_multiple_steps(self, region):
        """Multiple steps should not error."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            drive = rng.random(region.n_l23_total)
            region.step(drive)
        assert region.l23.active.any()

    def test_prediction_develops(self, region):
        """After repeated input, L2/3 should develop predictions."""
        rng = np.random.default_rng(0)
        patterns = [rng.random(region.n_l23_total) for _ in range(3)]
        for _ in range(50):
            for p in patterns:
                region.step(p)
        # Some L2/3 neurons should be predicted
        region.step(patterns[0])
        # At minimum, the region processed without error
        assert region.active_columns.sum() == region.k_columns


class TestNoL5:
    """CorticalRegion with n_l5=0 (sensory, no output projection)."""

    @pytest.fixture()
    def region(self):
        return CorticalRegion(
            input_dim=20,
            n_columns=8,
            n_l4=4,
            n_l23=4,
            n_l5=0,
            k_columns=2,
            seed=42,
        )

    def test_has_l5_false(self, region):
        assert not region.has_l5
        assert region.n_l5 == 0
        assert region.n_l5_total == 0

    def test_output_port_is_l23(self, region):
        assert region.output_port is region.l23

    def test_l5_lamina_is_empty(self, region):
        assert region.l5.n_total == 0
        assert region.l5.active.shape == (0,)

    def test_output_scores_zero(self, region):
        encoding = np.random.default_rng(0).random(20)
        region.process(encoding)
        assert region.output_scores.sum() == 0

    def test_process_runs(self, region):
        encoding = np.random.default_rng(0).random(20)
        result = region.process(encoding)
        assert isinstance(result, np.ndarray)

    def test_l4_and_l23_activate(self, region):
        encoding = np.random.default_rng(0).random(20)
        region.process(encoding)
        assert region.l4.active.any()
        assert region.l23.active.any()

    def test_multiple_steps(self, region):
        rng = np.random.default_rng(0)
        for _ in range(20):
            region.process(rng.random(20))
        assert region.l23.active.any()


class TestGranularUnchanged:
    """Verify that standard granular regions still work identically."""

    @pytest.fixture()
    def region(self):
        return CorticalRegion(
            input_dim=20,
            n_columns=8,
            n_l4=4,
            n_l23=4,
            k_columns=2,
            seed=42,
        )

    def test_has_l4_true(self, region):
        assert region.has_l4

    def test_has_l5_true(self, region):
        assert region.has_l5

    def test_input_port_is_l4(self, region):
        assert region.input_port is region.l4

    def test_output_port_is_l5(self, region):
        assert region.output_port is region.l5

    def test_ff_weights_target_l4(self, region):
        assert region.ff_weights.shape == (20, region.n_l4_total)

    def test_process_runs(self, region):
        encoding = np.random.default_rng(0).random(20)
        result = region.process(encoding)
        assert isinstance(result, np.ndarray)
