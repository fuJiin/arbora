"""Tests for CA1 — dual-input (CA3 + direct EC) comparator."""

import numpy as np
import pytest

from arbora.hippocampus import CA1


class TestInit:
    def test_shapes(self):
        ca1 = CA1(ca3_dim=500, ec_direct_dim=1000, output_dim=1000, seed=0)
        assert ca1.ca3_weights.shape == (500, 1000)
        assert ca1.ec_weights.shape == (1000, 1000)

    def test_invalid_dims_raise(self):
        with pytest.raises(ValueError):
            CA1(ca3_dim=0, ec_direct_dim=100, output_dim=100)
        with pytest.raises(ValueError):
            CA1(ca3_dim=100, ec_direct_dim=0, output_dim=100)
        with pytest.raises(ValueError):
            CA1(ca3_dim=100, ec_direct_dim=100, output_dim=0)


class TestForward:
    def test_output_shape_and_types(self):
        ca1 = CA1(ca3_dim=500, ec_direct_dim=1000, output_dim=1000, seed=0)
        rng = np.random.default_rng(0)
        ca3 = rng.normal(size=500)
        ec = rng.normal(size=1000)
        output, match = ca1.forward(ca3, ec)
        assert output.shape == (1000,)
        assert output.dtype == np.float64
        assert isinstance(match, float)
        assert -1.0 <= match <= 1.0

    def test_deterministic(self):
        ca1 = CA1(ca3_dim=500, ec_direct_dim=1000, output_dim=1000, seed=0)
        rng = np.random.default_rng(0)
        ca3 = rng.normal(size=500)
        ec = rng.normal(size=1000)
        o1, m1 = ca1.forward(ca3, ec)
        o2, m2 = ca1.forward(ca3, ec)
        np.testing.assert_array_equal(o1, o2)
        assert m1 == m2

    def test_wrong_dims_raise(self):
        ca1 = CA1(ca3_dim=100, ec_direct_dim=200, output_dim=150, seed=0)
        with pytest.raises(ValueError):
            ca1.forward(np.zeros(99), np.zeros(200))
        with pytest.raises(ValueError):
            ca1.forward(np.zeros(100), np.zeros(199))

    def test_accepts_boolean_inputs(self):
        """Sparse binary CA3 and EC patterns should project cleanly."""
        ca1 = CA1(ca3_dim=500, ec_direct_dim=1000, output_dim=1000, seed=0)
        ca3 = np.zeros(500, dtype=np.bool_)
        ca3[::10] = True
        ec = np.zeros(1000, dtype=np.bool_)
        ec[::20] = True
        output, _match = ca1.forward(ca3, ec)
        assert output.shape == (1000,)


class TestMatchSignal:
    def test_zero_inputs_produce_zero_match(self):
        """No NaN, no DivisionByZero — zero vectors yield match=0."""
        ca1 = CA1(ca3_dim=500, ec_direct_dim=1000, output_dim=1000, seed=0)
        _, match = ca1.forward(np.zeros(500), np.zeros(1000))
        assert match == 0.0

    def test_dissimilar_random_inputs_have_low_match(self):
        """Unrelated random inputs produce near-zero match on average."""
        ca1 = CA1(ca3_dim=500, ec_direct_dim=1000, output_dim=1000, seed=0)
        rng = np.random.default_rng(0)
        matches = []
        for _ in range(20):
            ca3 = rng.normal(size=500)
            ec = rng.normal(size=1000)
            _, m = ca1.forward(ca3, ec)
            matches.append(m)
        # Mean match across unrelated random inputs should hover near 0.
        assert abs(np.mean(matches)) < 0.1

    def test_correlated_drives_produce_high_match(self):
        """When CA3 and EC drives are engineered to align, match > 0.5."""
        ca1 = CA1(ca3_dim=500, ec_direct_dim=1000, output_dim=1000, seed=0)
        # Solve for inputs that produce parallel drive vectors:
        # target_drive = any vector; CA3 input solved via least-squares on
        # ca3_weights; EC input solved likewise on ec_weights.
        rng = np.random.default_rng(0)
        target = rng.normal(size=1000)
        ca3_input, *_ = np.linalg.lstsq(ca1.ca3_weights.T, target, rcond=None)
        ec_input, *_ = np.linalg.lstsq(ca1.ec_weights.T, target, rcond=None)
        _, match = ca1.forward(ca3_input, ec_input)
        assert match > 0.9, f"expected near-identical drives; got match={match:.3f}"

    def test_anticorrelated_drives_produce_negative_match(self):
        """Oppositely-aligned drives produce negative match."""
        ca1 = CA1(ca3_dim=500, ec_direct_dim=1000, output_dim=1000, seed=0)
        rng = np.random.default_rng(0)
        target = rng.normal(size=1000)
        ca3_input, *_ = np.linalg.lstsq(ca1.ca3_weights.T, target, rcond=None)
        # EC aligned to -target.
        ec_input, *_ = np.linalg.lstsq(ca1.ec_weights.T, -target, rcond=None)
        _, match = ca1.forward(ca3_input, ec_input)
        assert match < -0.9
